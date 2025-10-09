import os
import json
import random
import glob
from pathlib import Path
from typing import Dict, Any, List

import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
import numpy as np

def _list_feature_files(dir_path: str) -> List[str]:
    """List all .npy feature files in directory."""
    files = glob.glob(os.path.join(dir_path, "*.npy"))
    files.sort()
    return files

class SimpleVATEXDataset(Dataset):
    """Simple VATEX dataset for features."""
    
    def __init__(self, features_dir: str, ann_json: str, frames_per_clip: int, max_videos: int = None):
        super().__init__()
        self.features_dir = features_dir
        self.ann_json = ann_json
        self.frames_per_clip = frames_per_clip
        self.max_videos = max_videos
        
        # Load annotations
        if os.path.exists(self.ann_json):
            with open(self.ann_json, "r") as f:
                data = json.load(f)
        else:
            print(f"Warning: Annotations not found at {self.ann_json}")
            data = []
        
        # Create id to captions mapping
        id_to_caps: Dict[str, List[str]] = {}
        for item in data:
            vid = item.get("videoID") or item.get("video_id")
            caps = item.get("enCap") or item.get("en_captions") or item.get("captions") or []
            if isinstance(caps, dict):
                caps = caps.get("captions", [])
            if vid and caps:
                english_caps = [c for c in caps if isinstance(c, str) and c.strip()]
                if english_caps:
                    id_to_caps[vid] = english_caps
        
        # Get feature files
        self.records: List[Dict[str, Any]] = []
        for path in _list_feature_files(self.features_dir):
            vid = Path(path).stem.split('_')[0]  # Extract video ID from filename
            caps = id_to_caps.get(vid)
            if not caps:
                # Use dummy caption if no annotation found
                caps = [f"A person is performing an activity in video {vid}"]
            self.records.append({"path": path, "captions": caps, "video_id": vid})
        
        # Limit to max_videos
        if self.max_videos is not None and len(self.records) > self.max_videos:
            random.seed(42)
            self.records = random.sample(self.records, self.max_videos)
            print(f"SimpleVATEXDataset: Limited to {self.max_videos} features")
        
        if not self.records:
            raise RuntimeError(f"No usable VATEX features found in {self.features_dir}")
        
        print(f"SimpleVATEXDataset: {len(self.records)} features found in {self.features_dir}")
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        rec = self.records[idx]
        path = rec["path"]
        caps = rec["captions"]
        caption = random.choice(caps)
        
        # Load features
        features = np.load(path)  # Shape (1, T, 1024)
        features = features.squeeze(0)  # Shape (T, 1024)
        
        # Sample temporal steps
        total_temporal_steps = features.shape[0]
        if total_temporal_steps <= self.frames_per_clip:
            # If not enough steps, repeat or pad
            sampled_indices = np.linspace(0, max(total_temporal_steps - 1, 0), self.frames_per_clip).astype(int)
        else:
            sampled_indices = np.sort(np.random.choice(total_temporal_steps, self.frames_per_clip, replace=False)).astype(int)
        
        sampled_features = features[sampled_indices]
        
        return {
            "pixel_values": torch.from_numpy(sampled_features).float(),
            "caption": caption,
            "path": path
        }

def collate_fn_features(batch):
    """Collate function for features."""
    pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)  # (B, T_sampled, D)
    captions = [b["caption"] for b in batch]
    paths = [b["path"] for b in batch]
    return {"pixel_values": pixel_values, "captions": captions, "paths": paths}

class SimpleVATEXDataModule(L.LightningDataModule):
    """Simple VATEX datamodule that works directly with val directory."""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
    
    def setup(self, stage=None):
        data_cfg = self.cfg.data
        p = self.cfg.paths
        
        # Use the same directory for both train and val
        features_dir = p.vatex_features_dir
        ann_json = p.vatex_ann
        
        # Get all feature files
        all_feature_files = _list_feature_files(features_dir)
        if not all_feature_files:
            raise RuntimeError(f"No .npy files found in {features_dir}")
        
        # Split into train/val
        random.seed(42)
        random.shuffle(all_feature_files)
        
        # Use 80% for train, 20% for val
        split_idx = int(0.8 * len(all_feature_files))
        train_files = all_feature_files[:split_idx]
        val_files = all_feature_files[split_idx:]
        
        # Limit to max videos if specified
        max_train_videos = data_cfg.get("max_train_videos", len(train_files))
        max_val_videos = data_cfg.get("max_val_videos", len(val_files))
        
        train_files = train_files[:max_train_videos]
        val_files = val_files[:max_val_videos]
        
        print(f"Using {len(train_files)} train files and {len(val_files)} val files")
        
        # Create train dataset
        self.train_ds = SimpleVATEXDataset(
            features_dir=features_dir,
            ann_json=ann_json,
            frames_per_clip=data_cfg.frames_per_clip,
            max_videos=len(train_files)
        )
        
        # Create val dataset
        self.val_ds = SimpleVATEXDataset(
            features_dir=features_dir,
            ann_json=ann_json,
            frames_per_clip=data_cfg.frames_per_clip,
            max_videos=len(val_files)
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.cfg.data.batch_size, 
            shuffle=True,
            num_workers=self.cfg.num_workers, 
            pin_memory=False, 
            persistent_workers=bool(self.cfg.num_workers),
            collate_fn=collate_fn_features
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.cfg.data.batch_size, 
            shuffle=False,
            num_workers=self.cfg.num_workers, 
            pin_memory=False, 
            persistent_workers=bool(self.cfg.num_workers),
            collate_fn=collate_fn_features
        )
