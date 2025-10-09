#!/usr/bin/env python3
"""
Simple training script for VATEX features without Lightning dependencies.
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

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from data.vatex_features_datamodule import VATEXFeaturesDataModule
from models.diffusion_features import VidTextDiffusionFeaturesModule

def load_config():
    """Load configuration from YAML file."""
    import yaml
    with open('../configs/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def simple_train():
    """Simple training loop without Lightning."""
    print("🚀 Starting VATEX Features Training...")
    
    # Load config
    cfg = load_config()
    print(f"✅ Config loaded: {cfg['data']['max_train_videos']} train videos")
    
    # Set random seed
    torch.manual_seed(cfg['seed'])
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    
    # Create datamodule
    print("📦 Setting up data...")
    dm = VATEXFeaturesDataModule(cfg)
    dm.setup()
    
    print(f"✅ Train dataset: {len(dm.train_ds)} samples")
    print(f"✅ Val dataset: {len(dm.val_ds)} samples")
    
    # Create model
    print("🧠 Creating model...")
    model = VidTextDiffusionFeaturesModule(cfg)
    
    # Create dataloaders
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    
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
        lr=cfg['optim']['lr'], 
        weight_decay=cfg['optim']['weight_decay']
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
            with torch.no_grad():
                v = model.encode_video(features)  # (B, video_proj_dim)
                y = model.encode_text(captions)   # (B, 512)
            
            # Sample timesteps
            bsz = y.size(0)
            t = torch.randint(0, model.timesteps, (bsz,), device=y.device, dtype=torch.long)
            
            # Add noise
            noisy, noise = model.q_sample(y, t)
            
            # Predict noise
            pred_noise = model.denoiser(noisy, v, t.float())
            
            # Compute loss
            loss = model.loss_fn(pred_noise, noise)
            
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
            checkpoint_path = f"../outputs/checkpoint_epoch_{epoch+1}.pth"
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
    simple_train()
