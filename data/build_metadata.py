import argparse
import json

from data.prepare_dataset import load_dataset, load_images, to_metadata


def build_metadata(images_dir: str, output_path: str, annotations_path: str = None, limit: int = None):
    if annotations_path:
        items = load_dataset(annotations_path, images_dir, limit)
    else:
        items = load_images(images_dir, limit)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"images": to_metadata(items)}, f, indent=2)

    return items


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="data/test")
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--output", default="data/metadata.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    items = build_metadata(args.images, args.output, args.annotations, args.limit)
    print(f"Saved metadata for {len(items)} images")
