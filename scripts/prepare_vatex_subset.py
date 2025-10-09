#!/usr/bin/env python3
"""
Script to prepare a subset of VATEX dataset (500 training + 50 validation videos)
with English captions only.

Usage:
    python scripts/prepare_vatex_subset.py --vatex_root /path/to/vatex/full
"""

import os
import json
import shutil
import random
import argparse
from pathlib import Path
from typing import Dict, List


def load_vatex_annotations(ann_path: str) -> Dict[str, List[str]]:
    """Load VATEX annotations and extract English captions."""
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    video_captions = {}
    for item in data:
        video_id = item.get("videoID") or item.get("video_id")
        # Extract English captions only
        caps = item.get("enCap") or item.get("en_captions") or item.get("captions") or []
        
        if isinstance(caps, dict):
            caps = caps.get("captions", [])
        
        if video_id and caps:
            english_caps = [c for c in caps if isinstance(c, str) and c.strip()]
            if english_caps:
                video_captions[video_id] = english_caps
    
    return video_captions


def select_videos_subset(video_captions: Dict[str, List[str]], 
                         source_video_dir: str, 
                         num_videos: int = 500) -> List[str]:
    """Select a random subset of videos that exist in the source directory."""
    # Find which videos actually exist
    available_videos = []
    for video_id in video_captions.keys():
        video_path = os.path.join(source_video_dir, f"{video_id}.mp4")
        if os.path.exists(video_path):
            available_videos.append(video_id)
    
    print(f"Found {len(available_videos)} available videos with English captions")
    
    # Select subset
    if len(available_videos) <= num_videos:
        print(f"Using all {len(available_videos)} available videos")
        return available_videos
    
    random.seed(42)  # For reproducibility
    selected = random.sample(available_videos, num_videos)
    print(f"Selected {len(selected)} videos randomly")
    return selected


def copy_videos_and_annotations(selected_videos: List[str],
                                video_captions: Dict[str, List[str]],
                                source_video_dir: str,
                                dest_video_dir: str,
                                dest_ann_path: str):
    """Copy selected videos and save their annotations."""
    os.makedirs(dest_video_dir, exist_ok=True)
    os.makedirs(os.path.dirname(dest_ann_path), exist_ok=True)
    
    # Copy videos
    print(f"\nCopying videos to {dest_video_dir}...")
    for i, video_id in enumerate(selected_videos, 1):
        src = os.path.join(source_video_dir, f"{video_id}.mp4")
        dst = os.path.join(dest_video_dir, f"{video_id}.mp4")
        
        if os.path.exists(src):
            shutil.copy2(src, dst)
            if i % 50 == 0:
                print(f"  Copied {i}/{len(selected_videos)} videos...")
    
    print(f"✓ Copied {len(selected_videos)} videos")
    
    # Save annotations (English only)
    annotations = []
    for video_id in selected_videos:
        if video_id in video_captions:
            annotations.append({
                "videoID": video_id,
                "enCap": video_captions[video_id]  # English captions only
            })
    
    with open(dest_ann_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved {len(annotations)} annotations to {dest_ann_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare VATEX subset with English captions")
    parser.add_argument("--vatex_root", type=str, required=True,
                       help="Path to full VATEX dataset root directory")
    parser.add_argument("--output_dir", type=str, default="./data/vatex",
                       help="Output directory for subset (default: ./data/vatex)")
    parser.add_argument("--num_train", type=int, default=500,
                       help="Number of training videos (default: 500)")
    parser.add_argument("--num_val", type=int, default=50,
                       help="Number of validation videos (default: 50)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("VATEX Subset Preparation (English Captions Only)")
    print("=" * 60)
    
    # Paths in full VATEX dataset
    full_train_videos = os.path.join(args.vatex_root, "videos_train")
    full_val_videos = os.path.join(args.vatex_root, "videos_val")
    full_train_ann = os.path.join(args.vatex_root, "ann", "train_en.json")
    full_val_ann = os.path.join(args.vatex_root, "ann", "val_en.json")
    
    # Output paths
    output_train_videos = os.path.join(args.output_dir, "videos_train")
    output_val_videos = os.path.join(args.output_dir, "videos_val")
    output_train_ann = os.path.join(args.output_dir, "ann", "train_en.json")
    output_val_ann = os.path.join(args.output_dir, "ann", "val_en.json")
    
    # Process training set
    print(f"\n📚 Processing Training Set ({args.num_train} videos)...")
    print(f"  Source: {full_train_videos}")
    print(f"  Annotations: {full_train_ann}")
    
    if not os.path.exists(full_train_ann):
        print(f"❌ Error: Training annotations not found: {full_train_ann}")
        return
    
    train_captions = load_vatex_annotations(full_train_ann)
    train_selected = select_videos_subset(train_captions, full_train_videos, args.num_train)
    copy_videos_and_annotations(
        train_selected, train_captions, full_train_videos,
        output_train_videos, output_train_ann
    )
    
    # Process validation set
    print(f"\n📚 Processing Validation Set ({args.num_val} videos)...")
    print(f"  Source: {full_val_videos}")
    print(f"  Annotations: {full_val_ann}")
    
    if not os.path.exists(full_val_ann):
        print(f"❌ Error: Validation annotations not found: {full_val_ann}")
        return
    
    val_captions = load_vatex_annotations(full_val_ann)
    val_selected = select_videos_subset(val_captions, full_val_videos, args.num_val)
    copy_videos_and_annotations(
        val_selected, val_captions, full_val_videos,
        output_val_videos, output_val_ann
    )
    
    print("\n" + "=" * 60)
    print("✅ VATEX Subset Prepared Successfully!")
    print("=" * 60)
    print(f"\n📁 Output Directory: {args.output_dir}")
    print(f"  Training: {len(train_selected)} videos (English)")
    print(f"  Validation: {len(val_selected)} videos (English)")
    print(f"\n🚀 You can now train with: python src/train.py")


if __name__ == "__main__":
    main()

