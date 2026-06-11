"""
02_train_det.py (Kaggle Notebook 2)
-----------------------------------
Trains the RA-UNet (Stage 1) to predict the deterministic storm trajectory.
"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from src.config.config_loader import load_config
from src.data.datamodule import SEVIRDataModule
from src.models.ra_unet import RAUNet

def train_deterministic():
    # 1. Setup Configuration & Device
    cfg = load_config() # Loads default.yaml
    # Check if a GPU is available, otherwise use CPU (very slow!)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 2. Load the Data
    data_path = "../../data/raw/sevir/sevir_128_processed.h5" # Fallback local path
    
    # Auto-detect the exact path in Kaggle (Kaggle sometimes nests folders under your username)
    if os.path.exists("/kaggle/input"):
        import glob
        h5_files = glob.glob("/kaggle/input/**/*.h5", recursive=True)
        if h5_files:
            data_path = h5_files[0]
            print(f"Auto-detected Kaggle data path: {data_path}")
        
    datamodule = SEVIRDataModule([data_path], cfg)
    train_loader = datamodule.get_train_dataloader()
    val_loader = datamodule.get_val_dataloader()

    # 3. Initialize Model, Cost Function, and Optimizer
    model = RAUNet(
        in_seq=cfg['data']['in_seq'], 
        out_seq=cfg['data']['out_seq'], 
        base_channels=cfg['model']['base_channels']
    ).to(device) # Move the model to the GPU

    criterion = nn.L1Loss() # Mean Absolute Error (Keeps edges sharp)
    optimizer = AdamW(model.parameters(), lr=cfg['training']['det']['lr'], weight_decay=1e-5)

    # 4. AMP (Automatic Mixed Precision) Setup
    # This doubles GPU speed by using 16-bit math instead of 32-bit math
    scaler = torch.cuda.amp.GradScaler(enabled=cfg['training']['use_amp'])

    # 5. The Training Loop
    epochs = cfg['training']['det']['epochs']
    best_val_loss = float('inf') # Starting with infinity, try to get lower
    os.makedirs(cfg['paths']['output_dir'], exist_ok=True)

    print("\nStarting Stage 1: Deterministic Training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # Loop through every batch in the dataset
        for batch_idx, (x, y) in enumerate(train_loader):
            # Move the Tensors from RAM to the GPU
            x, y = x.to(device), y.to(device)

            # Step A: Clear old math
            optimizer.zero_grad()

            # Step B: Forward Pass
            # autocast ensures the math is done in fast 16-bit
            with torch.cuda.amp.autocast(enabled=cfg['training']['use_amp']):
                mu, _ = model(x) 
                loss = criterion(mu, y) # Calculate the L1 Loss

            # Step C: Backward Pass 
            scaler.scale(loss).backward()
            
            # Preventing math explosions (Gradient Clipping)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Step D: Updating the Neural Network weights
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx} | Loss: {loss.item():.4f}")

        # 6. Validation (Testing on unseen data)
        model.eval() 
        val_loss = 0.0
        with torch.no_grad(): 
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.cuda.amp.autocast(enabled=cfg['training']['use_amp']):
                    mu, _ = model(x)
                    loss = criterion(mu, y)
                val_loss += loss.item()
                
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"=== Epoch {epoch+1} Complete | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} ===")

        # 7. Save the Best Weights
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            save_path = os.path.join(cfg['paths']['output_dir'], "ra_unet_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved new best model to {save_path}!")

if __name__ == "__main__":
    train_deterministic()
