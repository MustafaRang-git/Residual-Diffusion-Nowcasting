"""
ra_unet.py
----------
The Residual Attention U-Net. 
This acts as 'Stage 1' of our pipeline. It takes past radar frames and attempts 
to predict the blurry, average future trajectory of the storm.
"""

import torch
import torch.nn as nn
from .blocks import ResidualConvBlock

class RAUNet(nn.Module):
    def __init__(self, in_seq=13, out_seq=12, base_channels=32, hint_channels=16):
        super(RAUNet, self).__init__()
        
        self.out_seq = out_seq
        self.hint_channels = hint_channels

        # ====================================================
        # 1. THE ENCODER (Extracting deep meaning, shrinking size)
        # ====================================================
        # Input goes from 13 channels (past frames) to 32 channels. 
        self.enc1 = ResidualConvBlock(in_seq, base_channels)
        self.pool1 = nn.MaxPool2d(2) # Shrinks image by half (e.g., 128 -> 64)

        self.enc2 = ResidualConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2) # Shrinks image by half (e.g., 64 -> 32)

        self.enc3 = ResidualConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2) # Shrinks image by half (e.g., 32 -> 16)

        # ====================================================
        # 2. THE BOTTLENECK (The deepest understanding of the storm)
        # ====================================================
        # Here, the image is very small (16x16), but has 256 channels of deep math!
        self.bottleneck = ResidualConvBlock(base_channels * 4, base_channels * 8)

        # ====================================================
        # 3. THE DECODER (Expanding back to full image size)
        # ====================================================
        # Up-Convolutions mathematically stretch the image back out (16 -> 32)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        # Notice the input to dec3 is (base_channels * 8). That's because we glue the Encoder's 
        # high-res features directly to the Up-Conv features (The Skip Connection!)
        self.dec3 = ResidualConvBlock(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ResidualConvBlock(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ResidualConvBlock(base_channels * 2, base_channels)

        # ====================================================
        # 4. THE OUTPUT HEADS
        # ====================================================
        # Head A: Outputs the 12 future frames (The blurry deterministic forecast)
        self.final_conv = nn.Conv2d(base_channels, out_seq, kernel_size=1)
        
        # Head B: The "Hint" Extractor. This outputs a lightweight map to pass to the Diffusion model!
        self.hint_conv = nn.Conv2d(base_channels, hint_channels, kernel_size=1)

    def forward(self, x):
        # --- ENCODER ---
        e1 = self.enc1(x)               # Feature map 1
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)              # Feature map 2
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)              # Feature map 3
        p3 = self.pool3(e3)

        # --- BOTTLENECK ---
        b = self.bottleneck(p3)

        # --- DECODER & SKIP CONNECTIONS ---
        d3 = self.up3(b)
        # SKIP CONNECTION: We literally concatenate (glue together) the deep 'd3' with the early 'e3'.
        # This gives the network both deep mathematical meaning AND sharp edge boundaries.
        d3 = torch.cat([d3, e3], dim=1) 
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1) # SKIP CONNECTION
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1) # SKIP CONNECTION
        d1 = self.dec1(d1)

        # --- OUTPUT ---
        # The main forecast
        mu = self.final_conv(d1)
        
        # The bridge to the Stochastic model
        hint = self.hint_conv(d1)
        
        return mu, hint
