# src/data/vatex_features_datamodule.py
import os, json, random, glob
from pathlib import Path
from typing import Dict, Any, List

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import lightning as L

def _list_feature_files(dir_path: str) -> List[str]:
    """List all .npy feature files in directory."""
    files = glob.glob(os.path.join(dir_path, "*.npy"))
    files.sort()
    return files

def sample_temporal_indices(num_frames: int, total_frames: int, strategy: str = "uniform"):
    """Sample temporal indices from preprocessed features."""
    if total_frames <= num_frames:
        return np.linspace(0, max(total_frames - 1, 0), num_frames).astype(int).tolist()
    if strategy == "uniform":
        return np.linspace(0, total_frames - 1, num_frames).astype(int).tolist()
    return np.sort(np.random.choice(total_frames, num_frames, replace=False)).astype(int).tolist()

class VATEXFeaturesDataset(Dataset):
    """
    VATEX dataset using preprocessed .npy features instead of video files.
    
    Features format: (1, T, 1024) where T is temporal dimension
    We'll sample 16 temporal steps and reshape to (16, 1024) for consistency
    """
    def __init__(self, features_dir: str, ann_json: str, frames_per_clip: int, sample_strategy: str, max_videos: int = None):
        super().__init__()
        self.features_dir = features_dir
        self.ann_json = ann_json
        self.frames_per_clip = frames_per_clip
        self.sample_strategy = sample_strategy
        self.max_videos = max_videos

        # Load annotations
        if os.path.exists(self.ann_json):
            with open(self.ann_json, "r") as f:
                data = json.load(f)
        else:
            print(f"Warning: Annotations not found at {self.ann_json}")
            print("Creating dummy annotations for available features...")
            data = self._create_dummy_annotations()

        # Build mapping: video_id -> captions
        id_to_caps: Dict[str, List[str]] = {}
        for item in data:
            vid = item.get("videoID") or item.get("video_id")
            # Use English captions if available, otherwise use any captions
            caps = item.get("enCap") or item.get("en_captions") or item.get("captions") or []
            if isinstance(caps, dict):
                caps = caps.get("captions", [])
            if vid and caps:
                english_caps = [c for c in caps if isinstance(c, str) and c.strip()]
                if english_caps:
                    id_to_caps[vid] = english_caps

        # Pair feature files to captions
        self.records: List[Dict[str, Any]] = []
        feature_files = _list_feature_files(self.features_dir)
        
        for feature_path in feature_files:
            # Extract video ID from filename (e.g., "--07WQ2iBlw_000001_000011.npy" -> "--07WQ2iBlw")
            filename = Path(feature_path).stem
            # Try to extract video ID (remove timestamp suffix)
            if "_" in filename:
                video_id = filename.rsplit("_", 2)[0]  # Remove last two parts (timestamp)
            else:
                video_id = filename
            
            # Try to find captions for this video
            caps = id_to_caps.get(video_id)
            if not caps:
                # If no specific captions, use a generic one
                caps = [f"A person is performing an activity in video {video_id}"]
            
            self.records.append({"path": feature_path, "captions": caps, "video_id": video_id})

        # Limit to max_videos if specified
        if self.max_videos is not None and len(self.records) > self.max_videos:
            random.seed(42)  # For reproducibility
            self.records = random.sample(self.records, self.max_videos)
            print(f"VATEXFeaturesDataset: Limited to {self.max_videos} features (English captions)")

        if not self.records:
            raise RuntimeError(f"No usable VATEX features found in {self.features_dir}")

        print(f"VATEXFeaturesDataset: {len(self.records)} feature files found in {self.features_dir}")

    def _create_dummy_annotations(self):
        """Create dummy annotations for available features."""
        feature_files = _list_feature_files(self.features_dir)
        annotations = []
        
        for feature_path in feature_files[:500]:  # Limit to 500 for dummy data
            filename = Path(feature_path).stem
            if "_" in filename:
                video_id = filename.rsplit("_", 2)[0]
            else:
                video_id = filename
            
            annotations.append({
                "videoID": video_id,
                "enCap": [f"A person is performing an activity in video {video_id}"]
            })
        
        return annotations

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        feature_path = rec["path"]
        caps = rec["captions"]
        caption = random.choice(caps)

        # Load preprocessed features
        features = np.load(feature_path)  # Shape: (1, T, 1024)
        
        # Remove batch dimension and get temporal dimension
        features = features.squeeze(0)  # Shape: (T, 1024)
        temporal_length = features.shape[0]
        
        # Sample temporal indices
        temporal_indices = sample_temporal_indices(
            self.frames_per_clip, temporal_length, self.sample_strategy
        )
        
        # Sample features at selected temporal indices
        sampled_features = features[temporal_indices]  # Shape: (16, 1024)
        
        # Convert to torch tensor and add channel dimension for consistency
        # We'll treat the 1024-dim features as "pixel values" for the model
        pixel_values = torch.from_numpy(sampled_features).float()  # Shape: (16, 1024)
        
        return {
            "pixel_values": pixel_values, 
            "caption": caption, 
            "path": feature_path,
            "video_id": rec["video_id"]
        }

def collate_fn(batch):
    """Collate function for feature-based dataset."""
    # Stack features instead of pixel values
    features = torch.stack([b["pixel_values"] for b in batch], dim=0)  # (B, T, 1024)
    captions = [b["caption"] for b in batch]
    paths = [b["path"] for b in batch]
    video_ids = [b["video_id"] for b in batch]
    return {
        "pixel_values": features, 
        "captions": captions, 
        "paths": paths,
        "video_ids": video_ids
    }

class VATEXFeaturesDataModule(L.LightningDataModule):
    """DataModule for VATEX preprocessed features."""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        data_cfg = self.cfg.data
        p = self.cfg.paths

        # Get max_videos from config
        max_train_videos = data_cfg.get("max_train_videos", None)
        max_val_videos = data_cfg.get("max_val_videos", None)

        # Use the same features directory for both train and val
        # (since we only have val features, we'll split them)
        features_dir = p.get("vatex_features_dir", "./val")
        ann_json = p.get("vatex_ann", "./data/vatex/ann/val_en.json")

        # Create train/val split from available features
        all_features = _list_feature_files(features_dir)
        random.seed(42)
        random.shuffle(all_features)
        
        # Split 80/20 for train/val
        split_idx = int(0.8 * len(all_features))
        train_features = all_features[:split_idx]
        val_features = all_features[split_idx:]
        
        # Create temporary directories for train/val split
        train_dir = "./data/vatex/features_train"
        val_dir = "./data/vatex/features_val"
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        
        # Copy files to train/val directories
        import shutil
        for i, feature_path in enumerate(train_features):
            filename = os.path.basename(feature_path)
            shutil.copy2(feature_path, os.path.join(train_dir, filename))
        
        for i, feature_path in enumerate(val_features):
            filename = os.path.basename(feature_path)
            shutil.copy2(feature_path, os.path.join(val_dir, filename))
        
        print(f"Created train/val split: {len(train_features)} train, {len(val_features)} val")

        self.train_ds = VATEXFeaturesDataset(
            features_dir=train_dir,
            ann_json=ann_json,
            frames_per_clip=data_cfg.frames_per_clip,
            sample_strategy=data_cfg.sample_strategy,
            max_videos=max_train_videos
        )
        
        self.val_ds = VATEXFeaturesDataset(
            features_dir=val_dir,
            ann_json=ann_json,
            frames_per_clip=data_cfg.frames_per_clip,
            sample_strategy="uniform",
            max_videos=max_val_videos
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.cfg.data.batch_size, 
            shuffle=True,
            num_workers=self.cfg.num_workers, 
            pin_memory=False, 
            persistent_workers=bool(self.cfg.num_workers),
            collate_fn=collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.cfg.data.batch_size, 
            shuffle=False,
            num_workers=self.cfg.num_workers, 
            pin_memory=False, 
            persistent_workers=bool(self.cfg.num_workers),
            collate_fn=collate_fn
        )
