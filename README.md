# Video → Text-Embedding Diffusion (MSR-VTT + VideoMAE)

This baseline trains a diffusion model to reconstruct **caption embeddings** (CLIP text) from noise **conditioned on VideoMAE** video features.

## Setup
```bash
conda create -n vid-diff python=3.10 -y
conda activate vid-diff
pip install -r requirements.txt
