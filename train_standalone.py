#!/usr/bin/env python3
"""
Standalone training script for VATEX features without Lightning dependencies.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import json
import random
from pathlib import Path
import yaml

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def load_config():
    """Load configuration from YAML file."""
    with open('configs/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def sample_temporal_indices(num_frames: int, total_frames: int, strategy: str = "uniform"):
    """Sample temporal indices from preprocessed features."""
    if total_frames <= num_frames:
        return np.linspace(0, max(total_frames - 1, 0), num_frames).astype(int).tolist()
    if strategy == "uniform":
        return np.linspace(0, total_frames - 1, num_frames).astype(int).tolist()
    return np.sort(np.random.choice(total_frames, num_frames, replace=False)).astype(int).tolist()

class SimpleVATEXDataset:
    """Simple VATEX dataset for features."""
    
    def __init__(self, features_dir: str, ann_json: str, max_videos: int = None):
        self.features_dir = features_dir
        self.ann_json = ann_json
        self.max_videos = max_videos
        
        # Load annotations
        if os.path.exists(self.ann_json):
            with open(self.ann_json, 'r') as f:
                self.annotations = json.load(f)
        else:
            print(f"Warning: Annotations not found at {self.ann_json}")
            self.annotations = []
        
        # Get feature files
        self.feature_files = []
        for file in os.listdir(features_dir):
            if file.endswith('.npy'):
                self.feature_files.append(os.path.join(features_dir, file))
        
        # Limit to max_videos
        if self.max_videos and len(self.feature_files) > self.max_videos:
            random.seed(42)
            self.feature_files = random.sample(self.feature_files, self.max_videos)
        
        print(f"✅ Loaded {len(self.feature_files)} feature files")
    
    def __len__(self):
        return len(self.feature_files)
    
    def __getitem__(self, idx):
        feature_path = self.feature_files[idx]
        
        # Load features
        features = np.load(feature_path)  # Shape: (1, T, 1024)
        features = features.squeeze(0)  # Shape: (T, 1024)
        
        # Sample temporal indices
        temporal_length = features.shape[0]
        temporal_indices = sample_temporal_indices(16, temporal_length, "uniform")
        sampled_features = features[temporal_indices]  # Shape: (16, 1024)
        
        # Convert to tensor
        pixel_values = torch.from_numpy(sampled_features).float()
        
        # Get caption (use dummy if no annotations)
        if self.annotations and idx < len(self.annotations):
            caption = self.annotations[idx].get('enCap', ['A person is performing an activity'])[0]
        else:
            caption = f"A person is performing an activity in video {idx}"
        
        return {
            'pixel_values': pixel_values,
            'caption': caption,
            'path': feature_path
        }

class SimpleDiffusionModel(nn.Module):
    """Simple diffusion model for features."""
    
    def __init__(self, feature_dim=1024, text_dim=512, video_proj_dim=512):
        super().__init__()
        self.feature_dim = feature_dim
        self.text_dim = text_dim
        self.video_proj_dim = video_proj_dim
        
        # Project features to text embedding space
        self.feature_proj = nn.Linear(feature_dim, video_proj_dim)
        self.cond_norm = nn.LayerNorm(video_proj_dim)
        
        # Simple denoiser
        self.denoiser = nn.Sequential(
            nn.Linear(text_dim + video_proj_dim + 128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, text_dim)
        )
        
        # Time embedding
        self.time_embed = nn.Linear(128, 128)
        
    def encode_video(self, features):
        """Encode video features."""
        # Average over temporal dimension
        global_features = features.mean(dim=1)  # (B, 1024)
        video_emb = self.cond_norm(self.feature_proj(global_features))
        return video_emb
    
    def encode_text(self, captions):
        """Simple text encoding (dummy implementation)."""
        # For now, return random embeddings
        batch_size = len(captions)
        return torch.randn(batch_size, self.text_dim)
    
    def forward(self, features, captions):
        """Forward pass."""
        video_emb = self.encode_video(features)
        text_emb = self.encode_text(captions)
        
        # Simple concatenation
        combined = torch.cat([text_emb, video_emb], dim=-1)
        time_emb = torch.randn(combined.shape[0], 128)
        x = torch.cat([combined, time_emb], dim=-1)
        
        output = self.denoiser(x)
        return output

def collate_fn(batch):
    """Collate function for batch."""
    pixel_values = torch.stack([b['pixel_values'] for b in batch])
    captions = [b['caption'] for b in batch]
    return {'pixel_values': pixel_values, 'captions': captions}

def train():
    """Simple training loop."""
    print("🚀 Starting VATEX Features Training...")
    
    # Load config
    cfg = load_config()
    print(f"✅ Config loaded: {cfg['data']['max_train_videos']} train videos")
    
    # Set random seed
    torch.manual_seed(cfg['seed'])
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    
    # Create datasets
    print("📦 Setting up data...")
    train_dataset = SimpleVATEXDataset(
        features_dir='val',
        ann_json='data/vatex/ann/val_en.json',
        max_videos=cfg['data']['max_train_videos']
    )
    
    val_dataset = SimpleVATEXDataset(
        features_dir='val',
        ann_json='data/vatex/ann/val_en.json',
        max_videos=cfg['data']['max_val_videos']
    )
    
    print(f"✅ Train dataset: {len(train_dataset)} samples")
    print(f"✅ Val dataset: {len(val_dataset)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg['data']['batch_size'], 
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg['data']['batch_size'], 
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Create model
    print("🧠 Creating model...")
    model = SimpleDiffusionModel()
    
    # Test a batch
    print("🧪 Testing data loading...")
    try:
        batch = next(iter(train_loader))
        print(f"✅ Batch shape: {batch['pixel_values'].shape}")
        print(f"✅ Captions: {len(batch['captions'])} samples")
        print(f"✅ Sample caption: {batch['captions'][0]}")
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=float(cfg['optim']['lr']), 
        weight_decay=float(cfg['optim']['weight_decay'])
    )
    
    # Training loop
    print("🎯 Starting training...")
    model.train()
    
    for epoch in range(cfg['trainer']['max_epochs']):
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Get batch data
            features = batch['pixel_values']  # (B, T, 1024)
            captions = batch['captions']
            
            # Forward pass
            output = model(features, captions)
            
            # Simple loss (MSE with random target for now)
            target = torch.randn_like(output)
            loss = nn.MSELoss()(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{cfg['trainer']['max_epochs']}, "
                      f"Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"outputs/checkpoint_epoch_{epoch+1}.pth"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    print("🎉 Training completed!")

if __name__ == "__main__":
    train()
