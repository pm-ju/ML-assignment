from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from retriever.query_parser import ParsedQuery


@dataclass
class RankedResult:
    image_id: str
    score: float
    parts: Dict[str, float] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)


def _region_match(region: Dict, garment: str) -> bool:
    name = region.get("name", "").lower()
    role = region.get("role", "").lower()
    if garment in name:
        return True
    if garment in {"shirt", "coat"} and role == "upper":
        return True
    if garment in {"pants", "skirt", "shorts"} and role == "lower":
        return True
    if garment == "dress" and role == "full":
        return True
    return False


def garment_score(parsed: ParsedQuery, record) -> float:
    if not parsed.garments:
        return 1.0
    hits = 0
    for garment in parsed.garments:
        if any(_region_match(region, garment) for region in record.regions):
            hits += 1
    return hits / max(1, len(parsed.garments))


def color_score(parsed: ParsedQuery, record) -> float:
    if not parsed.colors and not parsed.bindings:
        return 1.0

    if parsed.bindings:
        scores = []
        for binding in parsed.bindings:
            best = 0.0
            for region in record.regions:
                if _region_match(region, binding.garment):
                    best = max(best, float(region.get("colors", {}).get(binding.color, 0.0)))
            if best == 0.0:
                best = float(record.colors.get(binding.color, 0.0))
            scores.append(min(1.0, best * 2.0))
        return float(np.mean(scores)) if scores else 0.0

    scores = [min(1.0, float(record.colors.get(color, 0.0)) * 2.0) for color in parsed.colors]
    return float(np.mean(scores)) if scores else 0.0


def scene_score(parsed: ParsedQuery, record) -> float:
    target = parsed.environment or parsed.style
    if not target:
        return 1.0
    value = float(record.scene_scores.get(target, 0.0))
    return max(0.0, min(1.0, (value + 0.35) / 0.7))


def rerank_constraints(parsed: ParsedQuery, base_scores: Dict[str, float], factor_index, top_k: int = 10) -> List[RankedResult]:
    results = []

    for image_id, base in base_scores.items():
        record = factor_index.get_image(image_id)
        if record is None:
            continue

        g = garment_score(parsed, record)
        c = color_score(parsed, record)
        s = scene_score(parsed, record)
        final = 0.52 * base + 0.18 * g + 0.20 * c + 0.10 * s

        results.append(
            RankedResult(
                image_id=image_id,
                score=float(final),
                parts={
                    "base": round(float(base), 4),
                    "garment": round(float(g), 4),
                    "color": round(float(c), 4),
                    "scene": round(float(s), 4),
                },
                metadata=record.__dict__,
            )
        )

    results.sort(key=lambda x: x.score, reverse=True)
    return results[:top_k]
