"""
streamlit_app.py
----------------
The Real-Time Inference Application (Phase 7).
To run: `streamlit run app/streamlit_app.py` from the v2/ directory.
"""

import streamlit as st
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import sys
import h5py
import random
import tempfile

# Path configuration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config.config_loader import load_config
from src.models.ra_unet import RAUNet
from src.models.diffusion import ResidualDiffusion

# ==========================================
# 1. UI Setup
# ==========================================
st.set_page_config(
    page_title="NOWCASTING",
    page_icon="⛈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⛈️ High-Resolution Precipitation Nowcasting")
st.markdown("""
Welcome to the Nowcasting System. This application utilizes a **Deterministic RA-UNet** 
paired with a **Stochastic DDPM Diffusion Engine** to generate hyper-realistic, 
high-frequency radar forecasts for the next 60 minutes.
""")

# ==========================================
# 2. Model Loading Logic (Cached for speed)
# ==========================================
@st.cache_resource
def load_system():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Path validation
    unet_path = os.path.join(os.path.dirname(__file__), "..", "data", "models", "ra_unet_best.pth")
    diff_path = os.path.join(os.path.dirname(__file__), "..", "data", "models", "diffusion_best.pth")
    
    if not os.path.exists(unet_path) or not os.path.exists(diff_path):
        return None, None, cfg, device, False
        
    ra_unet = RAUNet(in_seq=cfg['data']['in_seq'], out_seq=cfg['data']['out_seq'], base_channels=cfg['model']['base_channels']).to(device)
    ra_unet.load_state_dict(torch.load(unet_path, map_location=device))
    ra_unet.eval()

    diffusion = ResidualDiffusion(cfg).to(device)
    diffusion.load_state_dict(torch.load(diff_path, map_location=device))
    diffusion.eval()
    
    return ra_unet, diffusion, cfg, device, True

ra_unet, diffusion, cfg, device, models_ready = load_system()

# ==========================================
# 3. Sidebar Controls
# ==========================================
st.sidebar.header("Control Center")
st.sidebar.info("Demo of the the System")

if not models_ready:
    st.sidebar.error("❌ Models not found!")
else:
    st.sidebar.success("✅ Models Loaded & Ready!")

# Initialize session state
if 'y_true' not in st.session_state:
    st.session_state.y_true = None
if 'y_pred' not in st.session_state:
    st.session_state.y_pred = None

generate_btn = st.sidebar.button("Fetch New Storm & Generate Forecast", type="primary", disabled=not models_ready)

# ==========================================
# 4. Data Fetching
# ==========================================
def fetch_offline_sample():
    """
    Extracts a randomized sequence from the HDF5 archive.
    """
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "sevir", "sevir_128_processed.h5")
    
    if not os.path.exists(data_path):
        st.error(f"Could not find the dataset at {data_path}. Please make sure you downloaded your Kaggle dataset here!")
        st.stop()
        
    with h5py.File(data_path, 'r') as f:
        num_events = f['vil'].shape[0]
        random_idx = random.randint(0, num_events - 1)
        raw_event = f['vil'][random_idx] # Shape: (128, 128, 49)
        
    # Transpose to (Time, Height, Width) and normalize
    seq = np.transpose(raw_event, (2, 0, 1)).astype(np.float32) / 255.0
    seq_tensor = torch.from_numpy(seq).float().unsqueeze(0) # Adding Batch Dimension
    
    # Split into X (Input) and Y 
    in_seq = cfg['data']['in_seq']
    out_seq = cfg['data']['out_seq']
    x = seq_tensor[:, :in_seq]
    y_true = seq_tensor[:, in_seq : in_seq + out_seq]
    
    return x, y_true

# ==========================================
# 5. Inference & Visualization
# ==========================================
if generate_btn:
    with st.spinner("Computing inference..."):
        
        # Data loading
        x, y_true = fetch_offline_sample()
        x = x.to(device)
        
        # Model inference
        with torch.no_grad():
            mu, hint = ra_unet(x)
            st.info("Computing diffusion reverse process...")
            generated_residual = diffusion.sample(hint=hint, shape=mu.shape)
            y_pred = torch.clamp(mu + generated_residual, 0.0, 1.0)
            
        st.session_state.y_true = y_true.cpu()
        st.session_state.y_pred = y_pred.cpu()
        st.success("Forecast Generation Complete!")
        
if st.session_state.y_true is not None and st.session_state.y_pred is not None:
    st.subheader("60-Minute Weather Simulation")
    st.markdown("This radar loop shows the full 1-hour forecast. **Right-click the video below to save it as a GIF for your presentation!**")
    
    y_true_np = st.session_state.y_true[0].numpy()
    y_pred_np = st.session_state.y_pred[0].numpy()
    
    with st.spinner("Compiling radar frames into animation..."):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        im_true = axes[0].imshow(y_true_np[0], cmap='jet', vmin=0, vmax=1)
        axes[0].set_title("Actual Future Radar")
        axes[0].axis('off')
        
        im_pred = axes[1].imshow(y_pred_np[0], cmap='jet', vmin=0, vmax=1)
        axes[1].set_title("System Prediction")
        axes[1].axis('off')
        
        # Add time text at the bottom
        time_text = fig.text(0.5, 0.05, '', ha='center', fontsize=14, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        
        def update(frame):
            im_true.set_array(y_true_np[frame])
            im_pred.set_array(y_pred_np[frame])
            time_text.set_text(f"Forecast Time: +{(frame+1)*5} mins")
            return [im_true, im_pred, time_text]
            
        ani = animation.FuncAnimation(fig, update, frames=12, interval=300, blit=True)
        
        # Save to temporary GIF file
        tmp_dir = tempfile.gettempdir()
        gif_path = os.path.join(tmp_dir, "radar_simulation.gif")
        ani.save(gif_path, writer='pillow', fps=3)
        plt.close(fig)
        
    st.image(gif_path, use_container_width=True)
