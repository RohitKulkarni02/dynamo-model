# src/data/vatex_datamodule.py
import os, json, random, glob
from pathlib import Path
from typing import Dict, Any, List

import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import VideoMAEImageProcessor
import lightning as L
from PIL import Image
import numpy as np

def _list_videos(dir_path: str) -> List[str]:
    exts = ("*.mp4", "*.mov", "*.mkv", "*.avi")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(dir_path, e)))
    files.sort()
    return files

def sample_frame_indices(num_frames: int, total_frames: int, strategy: str = "uniform"):
    if total_frames <= num_frames:
        return np.linspace(0, max(total_frames - 1, 0), num_frames).astype(int).tolist()
    if strategy == "uniform":
        return np.linspace(0, total_frames - 1, num_frames).astype(int).tolist()
    return np.sort(np.random.choice(total_frames, num_frames, replace=False)).astype(int).tolist()

def load_frames_opencv(video_path: str, num_frames: int, strategy: str = "uniform") -> List[Image.Image]:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or num_frames
    idxs = sample_frame_indices(num_frames, total, strategy)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok: continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    cap.release()
    if not frames:
        raise RuntimeError(f"Failed to read frames from {video_path}")
    return frames

class VATEXLocalDataset(Dataset):
    """
    Local VATEX dataset:
      - videos_dir: folder of .mp4s, filenames like <videoID>.mp4
      - ann_json: JSON with entries containing 'videoID' and 'enCap' (list of captions)
    For each video we pick one caption at random at __getitem__ time.
    """
    def __init__(self, videos_dir: str, ann_json: str, frames_per_clip: int, frame_size: int, sample_strategy: str, max_videos: int = None):
        super().__init__()
        self.videos_dir = videos_dir
        self.ann_json = ann_json
        self.frames_per_clip = frames_per_clip
        self.sample_strategy = sample_strategy
        self.max_videos = max_videos

        with open(self.ann_json, "r") as f:
            data = json.load(f)

        # Build mapping: video_id -> English captions only
        id_to_caps: Dict[str, List[str]] = {}
        for item in data:
            vid = item.get("videoID") or item.get("video_id")
            # Explicitly use only English captions
            caps = item.get("enCap") or item.get("en_captions") or item.get("captions") or []
            if isinstance(caps, dict):  # sometimes { "captions": [...] }
                caps = caps.get("captions", [])
            if vid and caps:
                # Filter to ensure only English text captions
                english_caps = [c for c in caps if isinstance(c, str) and c.strip()]
                if english_caps:
                    id_to_caps[vid] = english_caps

        # Pair local files to captions
        self.records: List[Dict[str, Any]] = []
        for path in _list_videos(self.videos_dir):
            vid = Path(path).stem  # assume <videoID>.mp4
            caps = id_to_caps.get(vid)
            if not caps:
                continue
            self.records.append({"path": path, "captions": caps})

        # Limit to max_videos if specified (e.g., 500 videos)
        if self.max_videos is not None and len(self.records) > self.max_videos:
            random.seed(42)  # For reproducibility
            self.records = random.sample(self.records, self.max_videos)
            print(f"VATEXLocalDataset: Limited to {self.max_videos} videos (English captions only)")

        if not self.records:
            raise RuntimeError(f"No usable VATEX samples found. "
                               f"Check videos in {self.videos_dir} and annotations {self.ann_json}")

        self.processor = VideoMAEImageProcessor.from_pretrained(
            "MCG-NJU/videomae-base", size={"shortest_edge": frame_size}
        )

        print(f"VATEXLocalDataset: {len(self.records)} videos found in {self.videos_dir} (English captions)")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        path = rec["path"]
        caps = rec["captions"]
        caption = random.choice(caps)

        frames_pil = load_frames_opencv(path, self.frames_per_clip, self.sample_strategy)
        pixel_values = self.processor(list(frames_pil), return_tensors="pt").pixel_values.squeeze(0)  # (T,3,H,W)
        return {"pixel_values": pixel_values, "caption": caption, "path": path}

def collate_fn(batch):
    pixels = torch.stack([b["pixel_values"] for b in batch], dim=0)  # (B,T,3,H,W)
    captions = [b["caption"] for b in batch]
    paths = [b["path"] for b in batch]
    return {"pixel_values": pixels, "captions": captions, "paths": paths}

class VATEXDataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        data_cfg = self.cfg.data
        p = self.cfg.paths

        # Get max_videos from config (default to None for unlimited)
        max_train_videos = data_cfg.get("max_train_videos", None)
        max_val_videos = data_cfg.get("max_val_videos", None)

        self.train_ds = VATEXLocalDataset(
            videos_dir=p.vatex_train_videos,
            ann_json=p.vatex_train_ann,
            frames_per_clip=data_cfg.frames_per_clip,
            frame_size=data_cfg.frame_size,
            sample_strategy=data_cfg.sample_strategy,
            max_videos=max_train_videos
        )
        self.val_ds = VATEXLocalDataset(
            videos_dir=p.vatex_val_videos,
            ann_json=p.vatex_val_ann,
            frames_per_clip=data_cfg.frames_per_clip,
            frame_size=data_cfg.frame_size,
            sample_strategy="uniform",
            max_videos=max_val_videos
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.cfg.data.batch_size, shuffle=True,
                          num_workers=self.cfg.num_workers, pin_memory=False, persistent_workers=bool(self.cfg.num_workers),
                          collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.cfg.data.batch_size, shuffle=False,
                          num_workers=self.cfg.num_workers, pin_memory=False, persistent_workers=bool(self.cfg.num_workers),
                          collate_fn=collate_fn)
