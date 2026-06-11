"""
download_sevir.py
-----------------
The script acts as the orchestrator to download the raw SEVIR VIL radar data 
directly from the AWS Open Data Registry. It bypasses the need for manual downloads.
"""

import subprocess
from pathlib import Path
import sys

def download_sevir_data(target_dir: str, year: str = "2018"):
    """
    Downloads SEVIR VIL data for a specific year using the AWS CLI.
    
    Args:
        target_dir (str): The local directory where the data will be saved.
        year (str): The specific year of data to download (e.g., '2018', '2019').
    """
    # 1. Setup the paths
    # We use pathlib to ensure paths work on both Windows and Linux seamlessly.
    save_path = Path(target_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting download for SEVIR VIL ({year}) data...")
    print(f"Target directory: {save_path.absolute()}")
    
    # 2. Define the AWS S3 location
    # The --no-sign-request flag is crucial: it tells AWS that this is a public dataset 
    # and we don't need to provide a credit card or AWS account to access it.
    s3_uri = f"s3://sevir/data/vil/{year}/"
    
    # 3. Build the terminal command
    # This is exactly what would be run in terminal manually.
    command = [
        "aws", "s3", "sync", 
        "--no-sign-request", 
        s3_uri, 
        str(save_path)
    ]
    
    # 4. Execute the command
    try:
        # subprocess.run executes the terminal command from within Python.
        # check=True means Python will crash and show an error if the download fails.
        subprocess.run(command, check=True)
        print("\nDownload completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Failed to download data. Do you have the AWS CLI installed?")
        print(f"Error details: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("\n[ERROR] AWS CLI is not installed or not in your system PATH.")
        print("Please install it from: https://aws.amazon.com/cli/")
        sys.exit(1)

# This block allows the script to be run directly from the terminal
if __name__ == "__main__":
    # We will default to downloading a small subset of 2018 data into the v2/data folder
    download_sevir_data(target_dir="../../data/raw/sevir", year="2018")
