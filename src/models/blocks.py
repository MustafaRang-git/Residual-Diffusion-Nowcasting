"""
blocks.py
---------
Reusable PyTorch layers and blocks for the Nowcasting architecture.
Contains the CBAM Attention module and the standard Residual Convolution Block.
"""

import torch
import torch.nn as nn

# ==========================================
# 1. CBAM: Channel Attention (What to look at?)
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=8):
        super(ChannelAttention, self).__init__()
        # 1. The "Squashers" (Pooling)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 2. The tiny "Brain" (MLP)
        # It shrinks the channels down by a 'reduction' factor, then expands them back
        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        )
        
        # 3. The Sigmoid function forces the final multipliers between 0.0 and 1.0
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pass the input through both squashing methods
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        
        # Add them together and apply Sigmoid to get the final multipliers
        multipliers = self.sigmoid(avg_out + max_out)
        
        # Multiply the original input by these channel multipliers
        return x * multipliers

# ==========================================
# 2. CBAM: Spatial Attention (Where to look?)
# ==========================================
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # The 7x7 Convolution to scan for storm edges
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. Squash the channels (take the Average and the Max across the channel axis)
        # dim=1 is the Channel axis. keepdim=True keeps it as a 2D map.
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # 2. Stack the two maps together
        stacked = torch.cat([avg_out, max_out], dim=1)
        
        # 3. Blend them into a single mask and apply Sigmoid (0.0 to 1.0)
        mask = self.sigmoid(self.conv(stacked))
        
        # 4. Multiply the input by the Spatial Mask
        return x * mask

# ==========================================
# 3. CBAM: The Full Module
# ==========================================
class CBAM(nn.Module):
    """Combines Channel and Spatial attention sequentially."""
    def __init__(self, in_planes, reduction=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        # Step 1: Channel Refinement
        x = self.ca(x)
        # Step 2: Spatial Refinement
        x = self.sa(x)
        return x

# ==========================================
# 4. Residual Convolution Block
# ==========================================
class ResidualConvBlock(nn.Module):
    """
    A standard Convolutional Block that extracts features, but includes a 
    'skip connection' (adding the input to the output) to prevent math gradients 
    from dying out in deep networks.
    """
    def __init__(self, in_channels, out_channels, apply_cbam=True):
        super(ResidualConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # The CBAM module acts as the final "highlighter" for the extracted features
        self.cbam = CBAM(out_channels) if apply_cbam else nn.Identity()

        # If the number of channels changes, we need a mathematical bridge to allow 
        # the input to be added to the output. We use a 1x1 Convolution for this.
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity() # Does absolutely nothing, just passes the input through

    def forward(self, x):
        # Save the original input for the skip connection
        shortcut = self.shortcut(x)
        
        # Pass through the convolutions to extract features
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Highlight the important features
        out = self.cbam(out)
        
        # THE RESIDUAL CONNECTION: Add the original input back to the extracted features
        out = out + shortcut 
        
        return self.act(out)
