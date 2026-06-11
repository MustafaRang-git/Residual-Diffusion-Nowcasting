"""
01_data_prep.py (Kaggle Notebook 1)
-----------------------------------
"""

import os
import subprocess
import h5py
import numpy as np
import cv2
import gc

def refine_data():
    # 1. Downloading the raw AWS data to Kaggle's temporary drive
    print("Step 1: Downloading raw SEVIR data from AWS...")
    raw_dir = "/kaggle/working/raw_sevir"
    os.makedirs(raw_dir, exist_ok=True)
    
    # Downloading one specific, massive file directly using 'cp' (copy) instead of 'sync'
    cmd = ["aws", "s3", "cp", "--no-sign-request", 
           "s3://sevir/data/vil/2018/SEVIR_VIL_STORMEVENTS_2018_0101_0630.h5", 
           os.path.join(raw_dir, "sevir.h5")]
    subprocess.run(cmd, check=True)

    # 2. Finding the downloaded file
    downloaded_files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith('.h5')]
    if not downloaded_files:
        raise FileNotFoundError("AWS download failed to find any .h5 files.")
    raw_file = downloaded_files[0]
    
    # 3. Create a permanent, lightweight processed file
    processed_file = "/kaggle/working/sevir_128_processed.h5"
    print(f"\nStep 2: Resizing storms from 384x384 to 128x128...")
    
    with h5py.File(raw_file, 'r') as f_in:
        raw_vil = f_in['vil']
        num_events = raw_vil.shape[0]
        
        # We will process 4,000 events to ensure training is robust but fits in limits
        process_count = min(4000, num_events) 
        print(f"Processing {process_count} weather events...")

        with h5py.File(processed_file, 'w') as f_out:
            # Create a new dataset inside the lightweight file
            # Shape: (Events, 128, 128, 49 frames)
            dset = f_out.create_dataset('vil', shape=(process_count, 128, 128, 49), dtype=np.uint8)
            
            for i in range(process_count):
                event = raw_vil[i] # Shape: (384, 384, 49)
                resized_event = np.zeros((128, 128, 49), dtype=np.uint8)
                
                # Resize every single frame in the event
                for t in range(49):
                    resized_event[:, :, t] = cv2.resize(event[:, :, t], (128, 128), interpolation=cv2.INTER_LINEAR)
                
                dset[i] = resized_event
                
                if (i + 1) % 500 == 0:
                    print(f"  Processed {i + 1}/{process_count} events...")

    # 4. Clean up the massive raw file to free up Kaggle's 20GB limit!
    print("\nStep 3: Cleaning up raw files to free disk space...")
    os.remove(raw_file)
    gc.collect() # Force Python to clear RAM
    
    print("\nSUCCESS! The lightweight dataset is ready at:")
    print(processed_file)
    print("You can now click 'Save Version' in Kaggle and create your permanent Kaggle Dataset!")

if __name__ == "__main__":
    # In Kaggle, this will trigger automatically when the cell is run
    refine_data()
