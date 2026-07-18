import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image


@dataclass
class FashionRegion:
    category: str
    supercategory: str
    bbox: Optional[List[float]] = None
    segmentation: Optional[List] = None


@dataclass
class FashionItem:
    image_id: str
    path: str
    file_name: str
    width: int = 0
    height: int = 0
    category: str = "unknown"
    supercategory: str = "unknown"
    regions: List[FashionRegion] = field(default_factory=list)


def load_dataset(annotations_path: str, images_dir: str, limit: Optional[int] = None) -> List[FashionItem]:
    with open(annotations_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    id_to_image = {img["id"]: img for img in data.get("images", [])}
    id_to_category = {cat["id"]: cat for cat in data.get("categories", [])}
    grouped: Dict[str, FashionItem] = {}

    for ann in data.get("annotations", []):
        image_info = id_to_image.get(ann.get("image_id"))
        if not image_info:
            continue

        image_id = str(image_info["id"])
        file_name = image_info.get("file_name", "")
        img_path = os.path.join(images_dir, file_name)

        if not os.path.exists(img_path):
            continue

        cat = id_to_category.get(ann.get("category_id"), {})
        region = FashionRegion(
            category=cat.get("name", "unknown"),
            supercategory=cat.get("supercategory", "unknown"),
            bbox=ann.get("bbox"),
            segmentation=ann.get("segmentation"),
        )

        if image_id not in grouped:
            grouped[image_id] = FashionItem(
                image_id=image_id,
                path=img_path,
                file_name=file_name,
                width=int(image_info.get("width", 0)),
                height=int(image_info.get("height", 0)),
                category=region.category,
                supercategory=region.supercategory,
                regions=[region],
            )
        else:
            grouped[image_id].regions.append(region)

        if limit and len(grouped) >= limit:
            break

    return list(grouped.values())


def load_images(image_dir: str, limit: Optional[int] = None) -> List[FashionItem]:
    image_dir = Path(image_dir)
    valid_ext = {".jpg", ".jpeg", ".png", ".webp"}
    items = []

    for p in sorted(image_dir.iterdir()):
        if p.suffix.lower() not in valid_ext:
            continue
        with Image.open(p) as img:
            width, height = img.size
        items.append(
            FashionItem(
                image_id=p.stem,
                path=str(p),
                file_name=p.name,
                width=width,
                height=height,
            )
        )
        if limit and len(items) >= limit:
            break

    return items


def get_image(item: FashionItem) -> Image.Image:
    return Image.open(item.path).convert("RGB")


def to_metadata(items: List[FashionItem]) -> List[Dict]:
    result = []
    for item in items:
        result.append(
            {
                "image_id": item.image_id,
                "path": item.path,
                "file_name": item.file_name,
                "width": item.width,
                "height": item.height,
                "category": item.category,
                "supercategory": item.supercategory,
                "regions": [region.__dict__ for region in item.regions],
            }
        )
    return result
