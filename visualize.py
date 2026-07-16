import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_results(path: str = "output/result.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("queries", [])


def make_grid(results, output_path: str = "output/visualization.png", top_n: int = 5):
    thumb_w, thumb_h = 220, 260
    pad = 26
    label_h = 38
    rows = len(results)
    cols = top_n
    canvas = Image.new("RGB", (cols * (thumb_w + pad) + pad, rows * (thumb_h + label_h + pad) + pad), "white")
    draw = ImageDraw.Draw(canvas)

    for row_idx, item in enumerate(results):
        for col_idx, result in enumerate(item.get("retrieved", [])[:top_n]):
            path = result.get("path")
            if not path or not Path(path).exists():
                continue

            img = Image.open(path).convert("RGB")
            img.thumbnail((thumb_w, thumb_h))
            x = pad + col_idx * (thumb_w + pad)
            y = pad + row_idx * (thumb_h + label_h + pad)
            canvas.paste(img, (x, y + label_h))
            label = f"Rank {col_idx + 1}: {result.get('id')} {result.get('score')}"
            draw.text((x, y), label[:34], fill=(60, 60, 60))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    canvas.save(output_path)
    return output_path


if __name__ == "__main__":
    path = make_grid(load_results())
    print(f"Saved to: {path}")
