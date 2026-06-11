# ⛈️ Residual-Diffusion-Nowcasting

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Production_Ready-success)

> **Watch the live radar simulation loop below!** 
> <img width="1000" height="500" alt="demo" src="https://github.com/user-attachments/assets/23d1b683-1695-4f01-a1f8-90af6b64b980" />


## 📌 Overview
This repository contains a production-grade **Generative Precipitation Nowcasting System**. 

Traditional weather forecasting models (like standard CNNs or UNets) suffer from the "Double Penalty" problem - they tend to predict safe, blurry, low-intensity drizzle rather than accurately pinpointing high-intensity storm cores. To solve this, I engineered a dual-stage pipeline that uses a **Deterministic RA-UNet** to forecast the general storm trajectory, paired with a **Stochastic DDPM (Denoising Diffusion Probabilistic Model)** to hallucinate and inject high-frequency, high-intensity storm details back into the forecast.

This system accurately predicts the next 60 minutes of radar imagery based on the past 65 minutes of context.

## 🏗️ System Architecture
The pipeline is designed to be modular, scalable, and highly reproducible.

1. **Deterministic Backbone (RA-UNet):** A 64-channel Recurrent Attention U-Net that learns the temporal dynamics and spatial movement of storm cells.
2. **Generative Refiner (Residual DDPM):** A conditional Diffusion engine that takes the blurry RA-UNet output as a "hint" and runs a 1,000-step reverse diffusion process to synthesize realistic, high-intensity rainfall structures.
3. **Data Pipeline:** Custom PyTorch DataLoaders built to dynamically normalize, slice, and batch the massive 2018 SEVIR VIL (Vertically Integrated Liquid) HDF5 dataset.

## 📊 Performance Benchmarking
This architecture was evaluated quantitatively using the **Critical Success Index (CSI)** and **Mean Absolute Error (MAE)** against standard UNet baselines. 

By leveraging the Residual Diffusion process, the model successfully synthesizes high-intensity storm cores that traditional deterministic models fail to capture, resulting in superior CSI scores at high-intensity thresholds (e.g., `> 0.5` normalized intensity).

## 🚀 Reproducibility & Interactive Demo
I built a fully interactive Streamlit application to demonstrate the model's capabilities in real-time. The application pulls unseen storm events, runs the diffusion inference, and generates an automated 60-minute forecast simulation loop.

### Run it Locally
Ensure you have the required `data/` and model weights (`.pth`), then run:
```bash
# 1. Clone the repository
git clone https://github.com/MustafaRang-git/Residual-Diffusion-Nowcasting.git
cd Residual-Diffusion-Nowcasting

# 2. Activate your environment and install dependencies
pip install -r requirements.txt

# 3. Launch the Simulation App
streamlit run app/streamlit_app.py
