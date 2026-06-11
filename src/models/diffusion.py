"""
diffusion.py
------------
The Conditional Denoising Diffusion Probabilistic Model (DDPM).
This acts as 'Stage 2' of our pipeline. It learns to generate chaotic, realistic 
storm details out of pure static noise, guided by the 'hint' from the RA-UNet.
"""

import torch
import torch.nn as nn
import math
from .blocks import ResidualConvBlock

# ====================================================
# 1. TIME EMBEDDING (Knowing "When" we are)
# ====================================================
class SinusoidalPositionEmbeddings(nn.Module):
    """
    Because the Diffusion network uses the exact same weights for all 1,000 steps,
    we must mathematically inject a "clock" into the data so the network knows 
    if it is looking at Step 10 (almost clear weather) or Step 900 (pure static).
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        # Generates a unique barcode of sine and cosine waves for every timestep 't'
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# ====================================================
# 2. THE DENOISING U-NET (The Neural Network)
# ====================================================
class DiffusionUNet(nn.Module):
    """
    This is the network that looks at a noisy image and guesses what the static looks like.
    """
    def __init__(self, out_seq=12, hint_channels=16, base_channels=32):
        super().__init__()
        
        # The input is the Noisy Residual (out_seq) PLUS the Hint from RA-UNet (hint_channels)
        in_channels = out_seq + hint_channels
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, base_channels * 4),
            nn.GELU(),
            nn.Linear(base_channels * 4, base_channels * 4)
        )
        
        # A simple Encoder-Decoder structure to process the noisy image
        self.enc1 = ResidualConvBlock(in_channels, base_channels)
        self.enc2 = ResidualConvBlock(base_channels, base_channels * 2)
        
        self.bottleneck = ResidualConvBlock(base_channels * 2, base_channels * 4)
        
        self.dec2 = ResidualConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = ResidualConvBlock(base_channels * 2 + base_channels, base_channels)
        
        # The output must be EXACTLY the same size as the noise we are trying to predict
        self.final_conv = nn.Conv2d(base_channels, out_seq, kernel_size=1)

    def forward(self, x_noisy, t, hint):
        # 1. Condition the network with the Hint Stencil
        # We mathematically glue the Noisy Image and the RA-UNet Hint together!
        x = torch.cat([x_noisy, hint], dim=1)
        
        # 2. Get the "Time Barcode" so the network knows what step it is on
        t_emb = self.time_mlp(t)
        # Reshape the time barcode so we can add it to our image channels
        t_emb = t_emb[:, :, None, None] 

        # 3. Extract Features (Encoder)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        
        # 4. Process deep meaning (Bottleneck)
        b = self.bottleneck(e2)
        # INJECT THE TIME: We add the time barcode directly into the deepest math
        b = b + t_emb 
        
        # 5. Rebuild the image (Decoder with Skip Connections)
        d2 = self.dec2(torch.cat([b, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        
        # 6. Output the predicted static noise!
        predicted_noise = self.final_conv(d1)
        return predicted_noise

# ====================================================
# 3. THE DIFFUSION WRAPPER (The Math Engine)
# ====================================================
class ResidualDiffusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = DiffusionUNet(
            out_seq=cfg['data']['out_seq'],
            hint_channels=cfg['model']['hint_channels'],
            base_channels=cfg['model']['base_channels']
        )
        
        self.timesteps = cfg['diffusion']['timesteps']
        beta_start = cfg['diffusion']['beta_start']
        beta_end = cfg['diffusion']['beta_end']
        
        # Calculate the math variables needed to add noise (Forward Process)
        # torch.linspace generates a steady ramp of numbers from start to end
        betas = torch.linspace(beta_start, beta_end, self.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0) # Cumulative multiplication
        
        # We register these as "buffers". They are constants saved inside the model 
        # so they get automatically moved to the GPU, but the model doesn't try to train them.
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(self, x_start, t, noise=None):
        """
        THE FORWARD PROCESS (Destroying Data):
        Adds the mathematically perfect amount of Gaussian noise to the true residual at step 't'.
        """
        if noise is None:
            noise = torch.randn_like(x_start) # Generate pure static

        # Reshape the alpha constants so they can be multiplied against our 4D image tensor
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        # The DDPM Forward Equation: scales the image down slightly, and adds scaled noise!
        return sqrt_alpha_t * x_start + sqrt_one_minus_alpha_t * noise

    def forward(self, true_residual, hint):
        """
        TRAINING LOOP:
        1. Pick a random timestep 't'
        2. Destroy the true_residual to that timestep
        3. Ask the network to predict the noise that was just added.
        """
        batch_size = true_residual.shape[0]
        # Pick a completely random step between 0 and 999 for every image in the batch
        t = torch.randint(0, self.timesteps, (batch_size,), device=true_residual.device).long()
        
        # Generate pure static noise
        noise = torch.randn_like(true_residual)
        
        # Mathematically add the noise to the true image
        x_noisy = self.q_sample(true_residual, t, noise)
        
        # Ask the UNet to predict the noise, guided by the hint stencil
        predicted_noise = self.model(x_noisy, t, hint)
        
        return predicted_noise, noise

    @torch.no_grad()
    def p_sample(self, x_t, t, hint, t_index):
        """
        Takes an image at step 't', guesses the static noise, and mathematically 
        subtracts it to step 't-1'.
        """
        # 1. Ask the network to guess the static
        predicted_noise = self.model(x_t, t, hint)
        
        # Extract the mathematical constants for this specific timestep
        betas_t = self.cfg['diffusion']['beta_start'] + (self.cfg['diffusion']['beta_end'] - self.cfg['diffusion']['beta_start']) * (t_index / self.timesteps)
        alphas_t = 1.0 - betas_t
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_index]

        # 2. The core DDPM Reverse Equation:
        # We scale the image, and subtract the scaled predicted noise.
        model_mean = (1.0 / math.sqrt(alphas_t)) * (x_t - (betas_t / sqrt_one_minus_alphas_cumprod_t) * predicted_noise)

        # 3. If we are at step 0, we are done! Otherwise, we inject a tiny bit of 
        # random variance back in to prevent the math from collapsing into a blur.
        if t_index == 0:
            return model_mean
        else:
            posterior_variance = betas_t
            noise = torch.randn_like(x_t)
            return model_mean + math.sqrt(posterior_variance) * noise

    @torch.no_grad()
    def sample(self, hint, shape):
        """
        The full 1,000-step generation loop!
        """
        device = hint.device
        # 1. Start with pure TV static
        x = torch.randn(shape, device=device)
        
        # 2. Loop backwards from 999 down to 0
        for i in reversed(range(0, self.timesteps)):
            # Create a tensor filled with the current time step 'i'
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            
            # Step backward one unit of time!
            x = self.p_sample(x, t, hint, i)
            
        # x is now the fully generated, sharp, chaotic Residual!
        return x
