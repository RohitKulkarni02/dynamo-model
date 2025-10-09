# src/data/msrvtt_datamodule.py
import os
import csv
import hashlib
import random
import subprocess
from pathlib import Path
from typing import Optional, Set, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import VideoMAEImageProcessor
import lightning as L

from ..utils.video import load_frames_as_pil


# ---------------------------
# Helpers
# ---------------------------

def _safe_stem(s: str) -> str:
    """Stable short filename stem from an arbitrary string."""
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def _ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def _load_manifest(path: str) -> Set[str]:
    """
    Read a CSV manifest with headers: url,status
    Keep only rows where status == 'ok'.
    """
    keep: Set[str] = set()
    if not path or not os.path.exists(path):
        return keep
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("status") == "ok" and row.get("url"):
                keep.add(row["url"])
    return keep


def _yt_dlp_cmd(url: str, out_path: str,
                cookie_file: Optional[str],
                cookies_from_browser: Optional[str]) -> list[str]:
    """
    Build yt-dlp command, preferring a cookie file if provided.
    """
    cmd = ["yt-dlp", "-f", "mp4", "-o", out_path, url, "--quiet", "--no-warnings"]
    if cookie_file and os.path.exists(cookie_file):
        cmd += ["--cookies", cookie_file]
    elif cookies_from_browser:
        cmd += ["--cookies-from-browser", cookies_from_browser]
    return cmd


def _download_youtube(url: str, out_path: str,
                      cookie_file: Optional[str],
                      cookies_from_browser: Optional[str]):
    """
    Download the full video. Raise if download fails.
    """
    cmd = _yt_dlp_cmd(url, out_path, cookie_file, cookies_from_browser)
    subprocess.run(cmd, check=True)


def _trim_ffmpeg(in_path: str, out_path: str, start: float, end: float):
    """
    Trim to [start, end]. Try stream copy; fall back to re-encode if needed.
    """
    duration = max(0.1, float(end) - float(start))
    # Try stream copy first
    cmd1 = [
        "ffmpeg", "-y",
        "-ss", f"{start:.2f}",
        "-i", in_path,
        "-t", f"{duration:.2f}",
        "-c", "copy",
        out_path,
        "-loglevel", "error",
    ]
    try:
        subprocess.run(cmd1, check=True)
    except subprocess.CalledProcessError:
        # Fall back to re-encode (slower but reliable)
        cmd2 = [
            "ffmpeg", "-y",
            "-ss", f"{start:.2f}",
            "-i", in_path,
            "-t", f"{duration:.2f}",
            "-c:v", "libx264",
            "-c:a", "aac",
            "-preset", "veryfast",
            out_path,
            "-loglevel", "error",
        ]
        subprocess.run(cmd2, check=True)


# ---------------------------
# Dataset
# ---------------------------

class MSRVTTDataset(Dataset):
    """
    MSR-VTT (AlexZigma/msr-vtt mirror) dataset wrapper that:
      - downloads YouTube videos on-demand with yt-dlp
      - trims segments to [start time, end time] with ffmpeg
      - loads frames and preprocesses for VideoMAE
      - skips dead/private URLs by resampling new indices
      - (optionally) filters to a pre-probed manifest of reachable URLs
    """

    def __init__(
        self,
        hf_split: str,
        dataset_name: str,
        frames_per_clip: int,
        frame_size: int,
        sample_strategy: str,
        cache_dir: str,
        cookies_from_browser: Optional[str] = None,
        cookie_file: Optional[str] = None,
        max_resample_tries: int = 10,
    ):
        super().__init__()

        self.ds = load_dataset(dataset_name, split=hf_split)
        self.frames_per_clip = frames_per_clip
        self.sample_strategy = sample_strategy
        self.cache_dir = cache_dir
        self.cookies_from_browser = cookies_from_browser
        self.cookie_file = cookie_file
        self.max_resample_tries = max_resample_tries

        _ensure_dir(self.cache_dir)

        # Preprocessor for VideoMAE
        self.processor = VideoMAEImageProcessor.from_pretrained(
            "MCG-NJU/videomae-base", size={"shortest_edge": frame_size}
        )

        # Optional: set by DataModule to limit to reachable URLs
        self.valid_indices: Optional[list[int]] = None

        # Track indices that already failed (e.g., private/blocked) to avoid retry storms
        self._bad_indices: Set[int] = set()

    def __len__(self) -> int:
        return len(self.valid_indices) if self.valid_indices is not None else len(self.ds)

    def _get_item_raw(self, idx: int) -> Dict[str, Any]:
        if self.valid_indices is not None:
            idx = self.valid_indices[idx]
        return self.ds[idx]

    def _seg_path(self, url: str, start: float, end: float, video_id: str) -> str:
        stem = f"{video_id}_{int(start)}_{int(end)}"
        fname = _safe_stem(stem) + ".mp4"
        return str(Path(self.cache_dir) / fname)

    def _prepare_segment(self, item: Dict[str, Any]) -> tuple[str, str]:
        url = item.get("url")
        if not url:
            raise RuntimeError("Item missing 'url' field")

        start = float(item.get("start time", 0.0))
        end = float(item.get("end time", start + 10.0))
        caption = item.get("caption") or item.get("sentence") or ""
        video_id = item.get("video_id") or "vid"

        seg_path = self._seg_path(url, start, end, video_id)
        if not os.path.exists(seg_path):
            # Download full video (per-URL cache)
            full_path = str(Path(self.cache_dir) / f"{_safe_stem(url)}.full.mp4")
            if not os.path.exists(full_path):
                _download_youtube(url, full_path, self.cookie_file, self.cookies_from_browser)
            # Trim to segment
            _trim_ffmpeg(full_path, seg_path, start, end)

        return seg_path, caption

    def _load_tensor(self, seg_path: str) -> torch.Tensor:
        frames_pil = load_frames_as_pil(seg_path, self.frames_per_clip, self.sample_strategy)
        pixel_values = self.processor(list(frames_pil), return_tensors="pt").pixel_values.squeeze(0)  # (T,3,H,W)
        return pixel_values

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # If known bad, jump to a random new index
        if self.valid_indices is None:
            effective_len = len(self.ds)
        else:
            effective_len = len(self.valid_indices)

        if idx in self._bad_indices and effective_len > 1:
            idx = random.randrange(effective_len)

        tries = 0
        last_err: Optional[Exception] = None

        while tries < self.max_resample_tries:
            try:
                item = self._get_item_raw(idx)
                seg_path, caption = self._prepare_segment(item)
                pixel_values = self._load_tensor(seg_path)
                return {"pixel_values": pixel_values, "caption": caption, "path": seg_path}
            except Exception as e:
                # mark as bad and resample
                # translate visible idx back to underlying ds index for tracking
                underlying_idx = idx if self.valid_indices is None else self.valid_indices[idx]
                self._bad_indices.add(underlying_idx)
                last_err = e
                idx = random.randrange(effective_len)
                tries += 1

        raise RuntimeError(
            f"Failed to fetch usable sample after {self.max_resample_tries} tries. "
            f"Last error: {type(last_err).__name__}: {last_err}"
        )


# ---------------------------
# Collate & DataModule
# ---------------------------

def collate_fn(batch: list[Dict[str, Any]]) -> Dict[str, Any]:
    pixels = torch.stack([b["pixel_values"] for b in batch], dim=0)  # (B,T,3,H,W)
    captions = [b["caption"] for b in batch]
    paths = [b["path"] for b in batch]
    return {"pixel_values": pixels, "captions": captions, "paths": paths}


class MSRVTTDataModule(L.LightningDataModule):
    """
    Lightning DataModule with:
      - optional manifest filtering (paths.msrvtt_manifest_train / val)
      - cookie file (env YTDLP_COOKIE_FILE) or cookies-from-browser (env YTDLP_COOKIES_FROM_BROWSER)
      - MPS-friendly DataLoader flags
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: Optional[str] = None):
        data_cfg = self.cfg.data
        cache_dir = self.cfg.paths.get("video_cache", "./data/msrvtt_cache")

        # Auth options for yt-dlp
        cookie_file = os.environ.get("YTDLP_COOKIE_FILE", "").strip() or None  # path to youtube_cookies.txt
        cookies_from_browser = os.environ.get("YTDLP_COOKIES_FROM_BROWSER", "").strip() or None  # e.g., "chrome" | "safari"

        # Optional manifests (pre-probed reachable URLs)
        manifest_train = self.cfg.paths.get("msrvtt_manifest_train", "data/msrvtt_manifest_train.csv")
        manifest_val = self.cfg.paths.get("msrvtt_manifest_val", "data/msrvtt_manifest_val.csv")
        keep_train = _load_manifest(manifest_train)
        keep_val = _load_manifest(manifest_val)

        # Build datasets
        self.train_ds = MSRVTTDataset(
            hf_split=data_cfg.get("split_train", "train"),
            dataset_name=data_cfg.dataset_name,
            frames_per_clip=data_cfg.frames_per_clip,
            frame_size=data_cfg.frame_size,
            sample_strategy=data_cfg.sample_strategy,
            cache_dir=cache_dir,
            cookies_from_browser=cookies_from_browser,
            cookie_file=cookie_file,
        )
        self.val_ds = MSRVTTDataset(
            hf_split=data_cfg.get("split_val", "val"),
            dataset_name=data_cfg.dataset_name,
            frames_per_clip=data_cfg.frames_per_clip,
            frame_size=data_cfg.frame_size,
            sample_strategy="uniform",
            cache_dir=cache_dir,
            cookies_from_browser=cookies_from_browser,
            cookie_file=cookie_file,
        )

        # Apply manifest filtering if present; otherwise keep all
        def _indices_filtered(ds: MSRVTTDataset, keep_urls: Set[str]) -> list[int]:
            if not keep_urls:
                return list(range(len(ds.ds)))
            idxs = [i for i in range(len(ds.ds)) if ds.ds[i].get("url") in keep_urls]
            return idxs or list(range(len(ds.ds)))

        self.train_ds.valid_indices = _indices_filtered(self.train_ds, keep_train)
        self.val_ds.valid_indices = _indices_filtered(self.val_ds, keep_val)

    # Dataloaders (MPS-friendly: pin_memory=False; persistent workers for speed)
    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.data.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=False,  # MPS: pinned memory not used; silence warnings
            collate_fn=collate_fn,
            persistent_workers=True if self.cfg.num_workers and self.cfg.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=False,
            collate_fn=collate_fn,
            persistent_workers=True if self.cfg.num_workers and self.cfg.num_workers > 0 else False,
        )
