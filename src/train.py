# src/train.py
import torch
import lightning as L
from hydra import main
from lightning.pytorch.loggers import TensorBoardLogger

# swap import:
# from src.data.msrvtt_datamodule import MSRVTTDataModule
from src.data.vatex_datamodule import VATEXDataModule
from src.models.diffusion_text_embed import VidTextDiffusionModule

@main(version_base=None, config_path="../configs", config_name="config")
def run(cfg):
    L.seed_everything(cfg.seed, workers=True)

    # dm = MSRVTTDataModule(cfg)
    dm = VATEXDataModule(cfg)
    dm.setup()

    model = VidTextDiffusionModule(cfg)

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
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    run()
