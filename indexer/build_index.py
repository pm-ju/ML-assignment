import argparse
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from data.prepare_dataset import get_image, load_dataset, load_images
from indexer.color_extractor import extract_color_profile
from indexer.embedding_extractor import FashionClipExtractor
from indexer.region_extractor import extract_regions, region_to_metadata
from indexer.vector_store import FactorIndex


SCENE_PROMPTS = {
    "office": "a person inside a modern office",
    "street": "a person walking on a city street",
    "park": "a person in a green park",
    "home": "a person in a home interior",
    "indoor": "a person indoors",
    "outdoor": "a person outdoors",
    "city": "a casual city walk outfit",
    "formal": "professional formal business attire",
    "runway": "a fashion runway show",
}


def _score_scenes(full_embedding: np.ndarray, scene_text_embeddings: Dict[str, np.ndarray]) -> Dict[str, float]:
    scores = {}
    for name, emb in scene_text_embeddings.items():
        scores[name] = round(float(full_embedding @ emb), 4)
    return scores


def extract_and_store(
    annotations_path: str = None,
    images_dir: str = "data/test",
    output_path: str = "output/index.pkl",
    limit: int = None,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
):
    if annotations_path and Path(annotations_path).exists():
        items = load_dataset(annotations_path, images_dir, limit)
    else:
        items = load_images(images_dir, limit)

    extractor = FashionClipExtractor(model_name=model_name, pretrained=pretrained)
    scene_text = extractor.encode_text(list(SCENE_PROMPTS.values()))
    scene_text_embeddings = dict(zip(SCENE_PROMPTS.keys(), scene_text))

    factors = {
        "global": {"embeddings": [], "image_ids": []},
        "region": {"embeddings": [], "image_ids": []},
        "scene": {"embeddings": [], "image_ids": []},
    }
    metadata: List[Dict] = []

    for item in tqdm(items, desc="Extracting features"):
        image = get_image(item)
        full_embedding = extractor.encode_image(image)
        regions = extract_regions(image, item)
        region_images = [region.crop for region in regions]
        region_embeddings = extractor.encode_images(region_images)
        region_meta = []
        image_colors = {}

        for region, emb in zip(regions, region_embeddings):
            colors = extract_color_profile(region.crop, region.mask)
            region_meta.append(region_to_metadata(region, colors))
            factors["region"]["embeddings"].append(emb)
            factors["region"]["image_ids"].append(item.image_id)
            for color, score in colors.items():
                image_colors[color] = max(image_colors.get(color, 0.0), score)

        factors["global"]["embeddings"].append(full_embedding)
        factors["global"]["image_ids"].append(item.image_id)
        factors["scene"]["embeddings"].append(full_embedding)
        factors["scene"]["image_ids"].append(item.image_id)

        metadata.append(
            {
                "image_id": item.image_id,
                "path": item.path,
                "file_name": item.file_name,
                "category": item.category,
                "supercategory": item.supercategory,
                "colors": image_colors,
                "scene_scores": _score_scenes(full_embedding, scene_text_embeddings),
                "regions": region_meta,
            }
        )

    index = FactorIndex(embed_dim=extractor.embed_dim)

    for factor, values in factors.items():
        if values["embeddings"]:
            index.add(factor, np.stack(values["embeddings"]), values["image_ids"])

    index.add_metadata(metadata)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    index.save(output_path)
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="data/val_annotations.json")
    parser.add_argument("--images", default="data/test")
    parser.add_argument("--output", default="output/index.pkl")
    parser.add_argument("--limit", type=int, default=3000)
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    args = parser.parse_args()

    index = extract_and_store(args.annotations, args.images, args.output, args.limit, args.model, args.pretrained)
    print(f"Index built: {index.count()}")
