import math
import torch
import torch.nn as nn
import lightning as L
from einops import rearrange
from transformers import VideoMAEModel
from ..utils.text import FrozenClipTextEncoder

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

class VidTextDiffusionModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg

        self.video_model = VideoMAEModel.from_pretrained(cfg.model.video_encoder_name)
        if cfg.model.freeze_video_encoder:
            for p in self.video_model.parameters(): p.requires_grad = False

        self.text_encoder = FrozenClipTextEncoder(cfg.model.clip_text_encoder_name)
        if cfg.model.freeze_text_encoder:
            for p in self.text_encoder.parameters(): p.requires_grad = False

        video_hidden = self.video_model.config.hidden_size
        text_dim = 512  # CLIP-base text width

        self.video_proj = nn.Linear(video_hidden, cfg.model.video_proj_dim)
        self.cond_norm = nn.LayerNorm(cfg.model.video_proj_dim)

        self.timesteps = cfg.model.diffusion.timesteps
        self.register_buffer("betas", make_beta_schedule(self.timesteps, cfg.model.diffusion.beta_schedule))
        alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))

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
    def encode_video(self, pixel_values):  # (B, T, 3, H, W)
        x = rearrange(pixel_values, "b t c h w -> b c t h w")
        out = self.video_model(pixel_values=x)
        vid_feat = out.last_hidden_state.mean(dim=1)
        vid_feat = self.cond_norm(self.video_proj(vid_feat))
        return vid_feat

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
        pixels = batch["pixel_values"]
        captions = batch["captions"]

        with torch.no_grad():
            v = self.encode_video(pixels)
            y = self.encode_text(captions)

        bsz = y.size(0)
        t = torch.randint(0, self.timesteps, (bsz,), device=y.device, dtype=torch.long)
        noisy, noise = self.q_sample(y, t)
        pred_noise = self.denoiser(noisy, v, t.float())
        loss = self.loss_fn(pred_noise, noise)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pixels = batch["pixel_values"]
        captions = batch["captions"]

        with torch.no_grad():
            v = self.encode_video(pixels)
            y = self.encode_text(captions)

        bsz = y.size(0)
        t = torch.randint(0, self.timesteps, (bsz,), device=y.device, dtype=torch.long)
        noisy, noise = self.q_sample(y, t)
        pred_noise = self.denoiser(noisy, v, t.float())
        loss = self.loss_fn(pred_noise, noise)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
