"""
03_train_diff.py (Kaggle Notebook 3)
------------------------------------
Trains the DDPM (Stage 2) to generate chaotic storm residuals.

"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW

from src.config.config_loader import load_config
from src.data.datamodule import SEVIRDataModule
from src.models.ra_unet import RAUNet
from src.models.diffusion import ResidualDiffusion

def train_diffusion():
    # 1. Setup Configuration & Device
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Stage 2 on device: {device}")

    # 2. Load the Data
    data_path = "../../data/raw/sevir/sevir_128_processed.h5" 
    if os.path.exists("/kaggle/input"):
        import glob
        h5_files = glob.glob("/kaggle/input/**/*.h5", recursive=True)
        if h5_files:
            data_path = h5_files[0]
            print(f"Auto-detected Kaggle data path: {data_path}")
    datamodule = SEVIRDataModule([data_path], cfg)
    train_loader = datamodule.get_train_dataloader()
    val_loader = datamodule.get_val_dataloader()

    # 3. Load AND FREEZE the Deterministic Backbone (Stage 1)
    print("Loading pre-trained RA-UNet...")
    ra_unet = RAUNet(
        in_seq=cfg['data']['in_seq'], 
        out_seq=cfg['data']['out_seq'], 
        base_channels=cfg['model']['base_channels']
    ).to(device)
    
    # Locate the weights from Notebook 2
    unet_weight_path = os.path.join(cfg['paths']['output_dir'], "ra_unet_best.pth")
    if os.path.exists("/kaggle/input"):
        import glob
        pth_files = glob.glob("/kaggle/input/**/*.pth", recursive=True)
        for f in pth_files:
            if 'unet' in f:
                unet_weight_path = f
                print(f"Auto-detected RA-UNet weights at: {unet_weight_path}")
                break
    
    ra_unet.load_state_dict(torch.load(unet_weight_path, map_location=device))
    
    # ---------------------------------------------------------
    # Turning off learning for the RA-UNet completely
    # ---------------------------------------------------------
    ra_unet.eval() 
    for param in ra_unet.parameters():
        param.requires_grad = False 
    
    # 4. Initialize Diffusion Model, Cost Function, and Optimizer
    diffusion = ResidualDiffusion(cfg).to(device)
    
    # Diffusion predicts STATIC NOISE. Using MSE (Mean Squared Error) because 
    # The predicted noise to exactly match the true generated static is wanted.
    criterion = nn.MSELoss() 
    
    optimizer = AdamW(diffusion.parameters(), lr=cfg['training']['diff']['lr'])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg['training']['use_amp'])

    # 5. The Training Loop
    epochs = cfg['training']['diff']['epochs']
    best_val_loss = float('inf')
    os.makedirs(cfg['paths']['output_dir'], exist_ok=True)

    print("\nStarting Stage 2: Stochastic Diffusion Training...")
    for epoch in range(epochs):
        diffusion.train()
        train_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=cfg['training']['use_amp']):
                # Step A: Getting the deterministic forecast and the HINT from the RA-UNet
                mu, hint = ra_unet(x)
                
                # Step B: Calculating the TRUE residual 
                true_residual = y - mu
                
                # Step C: Passing true residual and hint to Diffusion
                # It will automatically pick a random time 't', destroy the residual, 
                # and return its guess of the static noise.
                predicted_noise, true_noise = diffusion(true_residual, hint)
                
                # Step D: Calculating the Mistake (How close was its guess to the true static?)
                loss = criterion(predicted_noise, true_noise)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx} | Noise MSE Loss: {loss.item():.4f}")

        # 6. Validation
        diffusion.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.cuda.amp.autocast(enabled=cfg['training']['use_amp']):
                    mu, hint = ra_unet(x)
                    true_residual = y - mu
                    predicted_noise, true_noise = diffusion(true_residual, hint)
                    loss = criterion(predicted_noise, true_noise)
                val_loss += loss.item()
                
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"=== Epoch {epoch+1} Complete | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} ===")

        # 7. Save the Best Weights
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            save_path = os.path.join(cfg['paths']['output_dir'], "diffusion_best.pth")
            torch.save(diffusion.state_dict(), save_path)
            print(f"--> Saved new best diffusion model to {save_path}!")

if __name__ == "__main__":
    train_diffusion()
