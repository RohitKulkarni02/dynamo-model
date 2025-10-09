#!/usr/bin/env python3
"""
Simple training script for VATEX features using the simple datamodule.
"""

import torch
import lightning as L
from hydra import main
from lightning.pytorch.loggers import TensorBoardLogger

# Use the simple datamodule
from data.vatex_simple_datamodule import SimpleVATEXDataModule
from models.diffusion_features import VidTextDiffusionFeaturesModule

@main(version_base=None, config_path="../configs", config_name="config")
def run(cfg):
    L.seed_everything(cfg.seed, workers=True)
    
    # Use simple datamodule
    dm = SimpleVATEXDataModule(cfg)
    dm.setup()
    
    print(f"✅ Train dataset: {len(dm.train_ds)} samples")
    print(f"✅ Val dataset: {len(dm.val_ds)} samples")
    
    # Test a batch
    print("🧪 Testing data loading...")
    try:
        batch = next(iter(dm.train_dataloader()))
        print(f"✅ Batch shape: {batch['pixel_values'].shape}")
        print(f"✅ Captions: {len(batch['captions'])} samples")
        print(f"✅ Sample caption: {batch['captions'][0]}")
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return
    
    model = VidTextDiffusionFeaturesModule(cfg)
    
    logger = TensorBoardLogger(save_dir=cfg.paths.output_dir, name="tb")
    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        val_check_interval=cfg.trainer.val_check_interval,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        precision=cfg.precision,
        logger=logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.gpus if torch.cuda.is_available() else 1,
    )
    
    print("🚀 Starting training...")
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    run()
