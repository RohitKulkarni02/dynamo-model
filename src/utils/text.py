import torch
from transformers import CLIPTextModel, CLIPTokenizer

class FrozenClipTextEncoder(torch.nn.Module):
    def __init__(self, name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(name)
        self.text_model = CLIPTextModel.from_pretrained(name)
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, texts):
        toks = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        out = self.text_model(**toks.to(self.text_model.device))
        # Use pooled text embedding (last_hidden_state CLS equivalent)
        # CLIPTextModel doesn't expose pooled_output; mean-pool is common:
        emb = out.last_hidden_state.mean(dim=1)  # (B, D)
        return emb
