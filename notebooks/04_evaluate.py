"""
04_evaluate.py (Kaggle Notebook 4)
----------------------------------
Loads the fully trained system, generates forecasts on unseen validation data,
and visualizes the results.
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from src.config.config_loader import load_config
from src.data.datamodule import SEVIRDataModule
from src.models.ra_unet import RAUNet
from src.models.diffusion import ResidualDiffusion

def evaluate_system():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data Initialization
    data_path = "../../data/raw/sevir/sevir_128_processed.h5"
    if os.path.exists("/kaggle/input"):
        import glob
        h5_files = glob.glob("/kaggle/input/**/*.h5", recursive=True)
        if h5_files:
            data_path = h5_files[0]
            print(f"Auto-detected Kaggle data path: {data_path}")
    datamodule = SEVIRDataModule([data_path], cfg)
    val_loader = datamodule.get_val_dataloader()
    
    # Model Initialization
    print("Loading Models...")
    ra_unet = RAUNet(in_seq=cfg['data']['in_seq'], out_seq=cfg['data']['out_seq'], base_channels=cfg['model']['base_channels']).to(device)
    
    unet_weight_path = "../../outputs/ra_unet_best.pth"
    if os.path.exists("/kaggle/input"):
        import glob
        pth_files = glob.glob("/kaggle/input/**/*.pth", recursive=True)
        for f in pth_files:
            if 'unet' in f: unet_weight_path = f
                
    ra_unet.load_state_dict(torch.load(unet_weight_path, map_location=device))
    ra_unet.eval()

    diffusion = ResidualDiffusion(cfg).to(device)
    
    diff_weight_path = "../../outputs/diffusion_best.pth"
    if os.path.exists("/kaggle/input"):
        import glob
        pth_files = glob.glob("/kaggle/input/**/*.pth", recursive=True)
        for f in pth_files:
            if 'diffusion' in f: diff_weight_path = f
                
    diffusion.load_state_dict(torch.load(diff_weight_path, map_location=device))
    diffusion.eval()
    
    # Fetch validation batch
    x, y_true = next(iter(val_loader))
    x, y_true = x.to(device), y_true.to(device)
    
    print("Generating forecasts...")
    with torch.no_grad():
        # Deterministic trajectory and conditioning hint
        mu, hint = ra_unet(x)
        
        # Stochastic residual generation
        # Shape: (Batch, 12_frames, 128, 128)
        generated_residual = diffusion.sample(hint=hint, shape=y_true.shape)
        
        # Composite prediction
        y_pred = mu + generated_residual
        
        # Range truncation
        y_pred = torch.clamp(y_pred, 0.0, 1.0)
        
    # 4. Calculate Metrics
    def calculate_csi(pred, target, threshold):
        pred_b = pred >= threshold
        target_b = target >= threshold
        hits = torch.sum(pred_b & target_b).float()
        misses = torch.sum(~pred_b & target_b).float()
        false_alarms = torch.sum(pred_b & ~target_b).float()
        return (hits / (hits + misses + false_alarms + 1e-8)).item()

    print("\n" + "="*50)
    print("=== PERFORMANCE METRICS ===")
    print("="*50)
    
    unet_mae = torch.nn.functional.l1_loss(mu, y_true).item()
    diff_mae = torch.nn.functional.l1_loss(y_pred, y_true).item()
    print(f"RA-UNet MAE:   {unet_mae:.4f}")
    print(f"Diffusion MAE: {diff_mae:.4f}\n")
    
    for t in [0.1, 0.3, 0.5]:
        unet_csi = calculate_csi(mu, y_true, t)
        diff_csi = calculate_csi(y_pred, y_true, t)
        print(f"Threshold > {t} (Intensity)")
        print(f"  RA-UNet CSI:   {unet_csi:.4f}")
        print(f"  Diffusion CSI: {diff_csi:.4f}")
        if diff_csi > unet_csi:
            print("  🏆 Diffusion Wins! (Synthesized structure better)")
        else:
            print("  🏆 RA-UNet Wins! (Safer blurry prediction)")
        print("-" * 30)
        
    # 5. Timeline Visualization Grid
    storm_idx = 2
    
    target_frames = [2, 5, 8, 11]
    time_labels = ["+15 mins", "+30 mins", "+45 mins", "+60 mins"]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Nowcasting Timeline: Ground Truth vs. Generative Forecast", fontsize=20, y=0.95)
    
    for col, frame_idx in enumerate(target_frames):
        
        # Ground Truth
        true_img = y_true[storm_idx, frame_idx].cpu().numpy()
        ax_true = axes[0, col]
        im_true = ax_true.imshow(true_img, cmap='jet', vmin=0.0, vmax=1.0)
        ax_true.set_title(f"Ground Truth ({time_labels[col]})", fontsize=14)
        ax_true.axis('off')
        
        # Generative Prediction
        pred_img = y_pred[storm_idx, frame_idx].cpu().numpy()
        ax_pred = axes[1, col]
        im_pred = ax_pred.imshow(pred_img, cmap='jet', vmin=0.0, vmax=1.0)
        ax_pred.set_title(f"Diffusion Prediction ({time_labels[col]})", fontsize=14)
        ax_pred.axis('off')
        
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im_true, cax=cbar_ax, label="Normalized Intensity")
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()

if __name__ == "__main__":
    evaluate_system()
