"""
sevir_dataset.py
----------------
Defines the PyTorch Dataset for SEVIR VIL radar imagery, mapping binary HDF5 
data into normalized spatiotemporal tensors.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class SEVIRDataset(Dataset):
    """
    PyTorch Dataset yielding (input_sequence, target_sequence) pairs from SEVIR HDF5 archives.
    
    x: Historical radar frames (e.g., 13 frames)
    y: Future radar frames (e.g., 12 frames)
    """

    def __init__(self, h5_paths, event_ids, in_seq=13, out_seq=12, img_size=128):
        """
        Args:
            h5_paths (list): List of paths to the downloaded .h5 files.
            event_ids (list): Which specific weather events this dataset should use 
                              (allows us to split train/val easily).
            in_seq (int): Number of input frames.
            out_seq (int): Number of output frames to predict.
            img_size (int): Size to resize the images to (e.g., 128x128).
        """
        super().__init__()
        self.h5_paths = h5_paths
        self.event_ids = event_ids
        self.in_seq = in_seq
        self.out_seq = out_seq
        self.img_size = img_size
        self.total_needed = in_seq + out_seq

        self.total_needed = in_seq + out_seq

        self._index = []
        for fp in h5_paths:
            with h5py.File(fp, "r") as f:
                num_events = f["vil"].shape[0]
            for i in event_ids:
                if i < num_events:
                    self._index.append((fp, i))

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        fp, ev_idx = self._index[idx]

        with h5py.File(fp, "r") as f:
            raw = f["vil"][ev_idx] 

        seq = np.transpose(raw, (2, 0, 1)).astype(np.float32)

        T = seq.shape[0]
        if T >= self.total_needed:
            seq = seq[:self.total_needed]
        else:
            pad = np.repeat(seq[-1:], self.total_needed - T, axis=0)
            seq = np.concatenate([seq, pad], axis=0)

        seq = seq / 255.0

        seq_tensor = torch.from_numpy(seq).float()

        seq_tensor = seq_tensor.unsqueeze(0) 
        seq_tensor = torch.nn.functional.interpolate(
            seq_tensor, size=(self.img_size, self.img_size), mode="bilinear"
        )
        seq_tensor = seq_tensor.squeeze(0)

        x = seq_tensor[:self.in_seq]
        y = seq_tensor[self.in_seq : self.in_seq + self.out_seq]

        return x, y
