from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from data.prepare_dataset import FashionItem, FashionRegion


@dataclass
class ImageRegion:
    name: str
    role: str
    crop: Image.Image
    mask: Optional[Image.Image]
    bbox: Tuple[int, int, int, int]


GARMENT_NAMES = {
    "shirt": ["shirt", "t-shirt", "top", "blouse", "sweatshirt", "polo"],
    "tie": ["tie", "neck tie"],
    "coat": ["coat", "jacket", "raincoat", "overcoat", "blazer", "cardigan"],
    "pants": ["pants", "trousers", "jeans", "leggings"],
    "skirt": ["skirt"],
    "dress": ["dress", "gown"],
    "shorts": ["shorts"],
    "shoes": ["shoe", "shoes", "boots", "sneakers", "heels"],
    "bag": ["bag", "handbag", "backpack", "purse"],
    "hat": ["hat", "cap"],
    "scarf": ["scarf"],
}

ROLE_MAP = {
    "shirt": "upper",
    "tie": "accessory",
    "coat": "upper",
    "pants": "lower",
    "skirt": "lower",
    "dress": "full",
    "shorts": "lower",
    "shoes": "feet",
    "bag": "accessory",
    "hat": "head",
    "scarf": "accessory",
}


def normalize_garment(category: str, supercategory: str = "") -> Tuple[str, str]:
    text = f"{category} {supercategory}".lower()
    for garment, words in GARMENT_NAMES.items():
        if any(word in text for word in words):
            return garment, ROLE_MAP.get(garment, "detail")
    if "upper" in text:
        return "upper clothing", "upper"
    if "lower" in text:
        return "lower clothing", "lower"
    if "whole" in text:
        return "whole outfit", "full"
    return category.lower() if category else "full image", "detail"


def _polygon_mask(region: FashionRegion, image_size: Tuple[int, int]) -> Optional[Image.Image]:
    if not region.segmentation:
        return None

    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    segments = region.segmentation if isinstance(region.segmentation, list) else []

    for segment in segments:
        if not isinstance(segment, list) or len(segment) < 6:
            continue
        points = [(segment[i], segment[i + 1]) for i in range(0, len(segment) - 1, 2)]
        draw.polygon(points, fill=255)

    return mask


def _bbox_from_region(region: FashionRegion, image_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    w, h = image_size
    if region.bbox and len(region.bbox) >= 4:
        x, y, bw, bh = region.bbox[:4]
        left = max(0, int(x))
        top = max(0, int(y))
        right = min(w, int(x + bw))
        bottom = min(h, int(y + bh))
        if right > left and bottom > top:
            return left, top, right, bottom
    return 0, 0, w, h


def _fallback_regions(image: Image.Image) -> List[ImageRegion]:
    w, h = image.size
    return [
        ImageRegion("full image", "full", image, None, (0, 0, w, h)),
        ImageRegion("upper clothing", "upper", image.crop((0, 0, w, h // 2)), None, (0, 0, w, h // 2)),
        ImageRegion("lower clothing", "lower", image.crop((0, h // 2, w, h)), None, (0, h // 2, w, h)),
    ]


def extract_regions(image: Image.Image, item: FashionItem, max_regions: int = 12) -> List[ImageRegion]:
    if not item.regions:
        return _fallback_regions(image)

    regions: List[ImageRegion] = []
    image_size = image.size

    for raw in item.regions[:max_regions]:
        name, role = normalize_garment(raw.category, raw.supercategory)
        bbox = _bbox_from_region(raw, image_size)
        crop = image.crop(bbox)
        full_mask = _polygon_mask(raw, image_size)
        local_mask = full_mask.crop(bbox) if full_mask else None
        regions.append(ImageRegion(name, role, crop, local_mask, bbox))

    if not regions:
        return _fallback_regions(image)

    return regions


def region_to_metadata(region: ImageRegion, colors: Dict[str, float]) -> Dict:
    return {
        "name": region.name,
        "role": region.role,
        "bbox": list(region.bbox),
        "colors": colors,
    }
