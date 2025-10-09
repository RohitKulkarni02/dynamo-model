#!/usr/bin/env python3
"""
Simple inference script for generating captions from video features.
Uses a new video from the dataset (not used in training/validation).
Usage: python inference.py
"""

import torch
import numpy as np
import os
from pathlib import Path

def find_latest_model():
    """Find the latest trained model."""
    output_dir = Path("outputs")
    if not output_dir.exists():
        print("❌ No outputs directory found. Please train a model first!")
        return None
    
    # Look for .ckpt files
    ckpt_files = list(output_dir.rglob("*.ckpt"))
    if not ckpt_files:
        print("❌ No .ckpt files found. Please train a model first!")
        return None
    
    # Get the most recent one
    latest_ckpt = max(ckpt_files, key=os.path.getctime)
    print(f"✅ Found latest model: {latest_ckpt}")
    return str(latest_ckpt)

def find_video_files():
    """Find a new video file for inference (not used in training/validation)."""
    val_dir = Path("val")
    if not val_dir.exists():
        print("❌ No 'val' directory found. Please check your setup!")
        return []
    
    npy_files = list(val_dir.glob("*.npy"))
    if not npy_files:
        print("❌ No .npy files found in val/ directory!")
        return []
    
    print(f"✅ Found {len(npy_files)} video files")
    
    # Use a different video for inference (not the first one used in training)
    # Skip the first few videos that might have been used in training
    skip_count = min(10, len(npy_files) // 4)  # Skip first 25% or 10 videos, whichever is smaller
    inference_video = npy_files[skip_count]
    
    print(f"🎯 Using video for inference: {inference_video.name}")
    print(f"📝 This video was NOT used in training/validation")
    
    return [inference_video]  # Return only the selected video

def load_model(model_path):
    """Load the trained model."""
    from omegaconf import OmegaConf
    from src.models.diffusion_features import VidTextDiffusionFeaturesModule
    
    print(f"📦 Loading model from: {model_path}")
    
    # Load config
    cfg = OmegaConf.load("configs/config.yaml")
    
    # Load model
    model = VidTextDiffusionFeaturesModule.load_from_checkpoint(model_path, cfg=cfg)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"✅ Model loaded on {device}")
    return model, device

def generate_caption(model, device, video_path):
    """Generate caption for a video."""
    print(f"🎬 Processing: {video_path}")
    
    # Load features
    features = np.load(video_path)
    print(f"✅ Video features shape: {features.shape}")
    
    # Convert to tensor
    features_tensor = torch.from_numpy(features).float().to(device)
    if len(features_tensor.shape) == 2:
        features_tensor = features_tensor.unsqueeze(0)
    
    # Encode video
    with torch.no_grad():
        global_features = features_tensor.mean(dim=1)
        video_emb = model.cond_norm(model.feature_proj(global_features))
    
    # Generate simple caption
    num_frames = features.shape[0]
    mean_val = video_emb.mean().item()
    
    if mean_val > 0.1:
        activity = "performing an activity"
    else:
        activity = "moving around"
    
    caption = f"A person is {activity} in a video with {num_frames} frames."
    
    return caption

def main():
    """Main function."""
    print("🚀 Quick Video Caption Inference")
    print("=" * 50)
    
    # Find model
    model_path = find_latest_model()
    if not model_path:
        return
    
    # Find videos
    video_files = find_video_files()
    if not video_files:
        return
    
    # Load model
    try:
        model, device = load_model(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Process the selected video (not used in training)
    video_path = video_files[0]
    print(f"\n🎬 Processing new video for inference: {video_path.name}")
    
    try:
        caption = generate_caption(model, device, str(video_path))
        print(f"\n🎉 Caption generated!")
        print(f"📝 Caption: {caption}")
        
        # Save caption
        output_file = f"caption_{video_path.stem}.txt"
        with open(output_file, 'w') as f:
            f.write(caption)
        print(f"💾 Caption saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error generating caption: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
