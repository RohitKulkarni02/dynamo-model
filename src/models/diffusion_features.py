import math
import torch
import torch.nn as nn
import lightning as L
from einops import rearrange
from utils.text import FrozenClipTextEncoder

def make_beta_schedule(timesteps: int, mode: str = "linear"):
    if mode == "linear":
        beta_start, beta_end = 1e-4, 0.02
        return torch.linspace(beta_start, beta_end, timesteps)
    raise ValueError("Only 'linear' beta schedule implemented in this base.")

class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.lin2 = nn.Linear(dim, dim)

    def forward(self, t):
        freqs = torch.arange(0, 64, device=t.device).float()
        freqs = 1.0 / (10000 ** (2 * (freqs // 2) / 128))
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        emb = self.lin2(self.act(self.lin1(emb)))
        return emb

class Denoiser(nn.Module):
    def __init__(self, text_dim: int, video_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        layers = []
        in_dim = text_dim + video_dim + 128
        d = in_dim
        for _ in range(num_layers):
            layers += [nn.Linear(d, hidden_dim), nn.SiLU(), nn.Dropout(dropout)]
            d = hidden_dim
        layers += [nn.Linear(hidden_dim, text_dim)]
        self.net = nn.Sequential(*layers)
        self.tproj = TimeEmbedding(128)
        self.vproj = nn.Linear(video_dim, video_dim)
        self.norm_v = nn.LayerNorm(video_dim)

    def forward(self, noisy_text, video_emb, t):
        t_emb = self.tproj(t)
        v = self.norm_v(self.vproj(video_emb))
        x = torch.cat([noisy_text, v, t_emb], dim=-1)
        return self.net(x)

class VidTextDiffusionFeaturesModule(L.LightningModule):
    """
    Modified diffusion model for preprocessed video features.
    Instead of using VideoMAE on raw frames, we work directly with preprocessed features.
    """
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg

        # No video encoder needed - we work with preprocessed features directly
        # Just need a projection layer to match text embedding dimension
        self.feature_dim = 1024  # From .npy files shape: (T, 1024)
        text_dim = 512  # CLIP text embedding dimension
        
        # Project features to text embedding space
        self.feature_proj = nn.Linear(self.feature_dim, cfg.model.video_proj_dim)
        self.cond_norm = nn.LayerNorm(cfg.model.video_proj_dim)

        # Text encoder (same as before)
        self.text_encoder = FrozenClipTextEncoder(cfg.model.clip_text_encoder_name)
        if cfg.model.freeze_text_encoder:
            for p in self.text_encoder.parameters(): 
                p.requires_grad = False

        # Diffusion setup
        self.timesteps = cfg.model.diffusion.timesteps
        self.register_buffer("betas", make_beta_schedule(self.timesteps, cfg.model.diffusion.beta_schedule))
        alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))

        # Denoiser
        self.denoiser = Denoiser(
            text_dim=text_dim,
            video_dim=cfg.model.video_proj_dim,
            hidden_dim=cfg.model.diffusion.hidden_dim,
            num_layers=cfg.model.diffusion.num_layers,
            dropout=cfg.model.diffusion.dropout
        )
        self.loss_fn = nn.MSELoss()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.optim.lr, weight_decay=self.cfg.optim.weight_decay)

    @torch.no_grad()
    def encode_video(self, features):  # (B, T, 1024)
        """
        Encode preprocessed video features.
        Input: features of shape (B, T, 1024) where T is temporal dimension
        Output: video embeddings of shape (B, video_proj_dim)
        """
        # Average over temporal dimension to get global video representation
        # features: (B, T, 1024) -> (B, 1024)
        global_features = features.mean(dim=1)
        
        # Project to desired dimension
        video_emb = self.cond_norm(self.feature_proj(global_features))
        return video_emb

    @torch.no_grad()
    def encode_text(self, captions):
        return self.text_encoder(captions)  # (B, 512)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_cum = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_1m = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        return sqrt_cum * x_start + sqrt_1m * noise, noise

    def training_step(self, batch, batch_idx):
        features = batch["pixel_values"]  # (B, T, 1024) - preprocessed features
        captions = batch["captions"]

        with torch.no_grad():
            v = self.encode_video(features)  # (B, video_proj_dim)
            y = self.encode_text(captions)   # (B, 512)

        bsz = y.size(0)
        t = torch.randint(0, self.timesteps, (bsz,), device=y.device, dtype=torch.long)
        noisy, noise = self.q_sample(y, t)
        pred_noise = self.denoiser(noisy, v, t.float())
        loss = self.loss_fn(pred_noise, noise)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        features = batch["pixel_values"]  # (B, T, 1024)
        captions = batch["captions"]

        with torch.no_grad():
            v = self.encode_video(features)
            y = self.encode_text(captions)

        bsz = y.size(0)
        t = torch.randint(0, self.timesteps, (bsz,), device=y.device, dtype=torch.long)
        noisy, noise = self.q_sample(y, t)
        pred_noise = self.denoiser(noisy, v, t.float())
        loss = self.loss_fn(pred_noise, noise)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
