"""
datamodule.py
-------------
Orchestrates PyTorch DataLoaders, handling data splitting and batching
for SEVIR radar sequences.
"""

from torch.utils.data import DataLoader
import numpy as np
import math

from .sevir_dataset import SEVIRDataset 

class SEVIRDataModule:
    def __init__(self, h5_paths, cfg):
        """
        Args:
            h5_paths (list): A list of absolute paths to the downloaded .h5 files.
            cfg (dict): The loaded configuration dictionary from our default.yaml.
        """
        self.h5_paths = h5_paths
        self.cfg = cfg
        
        self.total_events = self._count_total_events()
        self._split_data()

    def _count_total_events(self):
        """Aggregate total sequence count across provided HDF5 partitions."""
        import h5py
        total = 0
        for fp in self.h5_paths:
            with h5py.File(fp, "r") as f:
                total += f["vil"].shape[0]
        print(f"Total weather events discovered: {total}")
        return total

    def _split_data(self):
        """Perform deterministic train/val split based on configured seed and ratios."""
        rng = np.random.default_rng(self.cfg['project']['seed'])
        
        all_ids = list(range(self.total_events))
        rng.shuffle(all_ids)

        n_train = int(math.floor(self.total_events * self.cfg['data']['train_ratio']))
        n_val   = int(math.floor(self.total_events * self.cfg['data']['val_ratio']))

        self.train_ids = all_ids[:n_train]
        self.val_ids   = all_ids[n_train : n_train + n_val]
        
        print(f"Split -> Train: {len(self.train_ids)} | Val: {len(self.val_ids)}")

    def get_train_dataloader(self):
        """Return DataLoader for the training partition."""
        dataset = SEVIRDataset(
            self.h5_paths, 
            self.train_ids, 
            in_seq=self.cfg['data']['in_seq'],
            out_seq=self.cfg['data']['out_seq'],
            img_size=self.cfg['data']['img_size']
        )
        return DataLoader(
            dataset, 
            batch_size=self.cfg['training']['det']['batch_size'], 
            shuffle=True, 
            num_workers=2
        )

    def get_val_dataloader(self):
        """Return DataLoader for the validation partition."""
        dataset = SEVIRDataset(
            self.h5_paths, 
            self.val_ids, 
            in_seq=self.cfg['data']['in_seq'],
            out_seq=self.cfg['data']['out_seq'],
            img_size=self.cfg['data']['img_size']
        )
        return DataLoader(
            dataset, 
            batch_size=self.cfg['training']['det']['batch_size'], 
            shuffle=False, 
            num_workers=2
        )
