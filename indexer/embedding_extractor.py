from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class FashionClipExtractor:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", device: Optional[str] = None):
        import open_clip

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.pretrained = pretrained
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        self.embed_dim = self.model.visual.output_dim

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> np.ndarray:
        x = self.preprocess(image).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(x)
        return F.normalize(feat, dim=-1).squeeze(0).cpu().numpy()

    @torch.no_grad()
    def encode_images(self, images: List[Image.Image]) -> np.ndarray:
        if not images:
            return np.array([])
        batch = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        feat = self.model.encode_image(batch)
        return F.normalize(feat, dim=-1).cpu().numpy()

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        tokens = self.tokenizer(texts).to(self.device)
        feat = self.model.encode_text(tokens)
        return F.normalize(feat, dim=-1).cpu().numpy()

    def encode_single_text(self, text: str) -> np.ndarray:
        return self.encode_text([text])[0]
