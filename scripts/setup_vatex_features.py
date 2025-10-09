#!/usr/bin/env python3
"""
Setup script for VATEX features-based training.
Creates necessary directories and dummy annotations.
"""

import os
import json
import random
from pathlib import Path

def create_dummy_annotations(features_dir: str, output_path: str, num_videos: int = 500):
    """Create dummy annotations for available feature files."""
    
    # Get list of feature files
    feature_files = []
    for file in os.listdir(features_dir):
        if file.endswith('.npy'):
            feature_files.append(file)
    
    print(f"Found {len(feature_files)} feature files")
    
    # Sample subset if needed
    if len(feature_files) > num_videos:
        random.seed(42)
        feature_files = random.sample(feature_files, num_videos)
        print(f"Sampled {num_videos} feature files")
    
    # Create dummy annotations
    annotations = []
    activities = [
        "A person is cooking in the kitchen",
        "Someone is playing a musical instrument", 
        "A person is exercising at the gym",
        "Someone is reading a book",
        "A person is dancing",
        "Someone is painting a picture",
        "A person is playing sports",
        "Someone is working on a computer",
        "A person is gardening",
        "Someone is driving a car"
    ]
    
    for i, feature_file in enumerate(feature_files):
        # Extract video ID from filename
        filename = Path(feature_file).stem
        if "_" in filename:
            video_id = filename.rsplit("_", 2)[0]
        else:
            video_id = filename
        
        # Assign random activity
        activity = random.choice(activities)
        
        annotations.append({
            "videoID": video_id,
            "enCap": [activity]
        })
    
    # Save annotations
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    
    print(f"Created {len(annotations)} dummy annotations at {output_path}")
    return len(annotations)

def main():
    print("Setting up VATEX features for training...")
    
    # Create necessary directories
    os.makedirs("./data/vatex/ann", exist_ok=True)
    
    # Create dummy annotations for available features
    num_annotations = create_dummy_annotations(
        features_dir="./val",
        output_path="./data/vatex/ann/val_en.json",
        num_videos=500
    )
    
    print("\n" + "="*50)
    print("✅ Setup Complete!")
    print("="*50)
    print(f"📁 Features directory: ./val")
    print(f"📝 Annotations: ./data/vatex/ann/val_en.json")
    print(f"🎯 Number of videos: {num_annotations}")
    print("\n🚀 Ready to train with:")
    print("   cd src && python train_features.py")

if __name__ == "__main__":
    main()
