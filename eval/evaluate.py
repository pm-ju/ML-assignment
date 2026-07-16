import argparse
import json
import time
from typing import Dict, List

import numpy as np

from indexer.vector_store import load_index
from retriever.constraint_reranker import color_score, garment_score, scene_score
from retriever.query_parser import QueryParser
from retriever.retriever import FashionRetriever


def _score_ids(query: str, image_ids: List[str], index):
    parsed = QueryParser.parse(query)
    scores = []

    for image_id in image_ids:
        record = index.get_image(str(image_id))
        if record is None:
            continue
        scores.append(
            0.35 * garment_score(parsed, record)
            + 0.40 * color_score(parsed, record)
            + 0.25 * scene_score(parsed, record)
        )

    return float(np.mean(scores)) if scores else None


def _load_baseline(path: str) -> Dict[str, List[str]]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = {}
    for item in data.get("queries", []):
        result[item["query"]] = [str(row["id"]) for row in item.get("retrieved", [])]
    return result


def evaluate(index_path: str, queries_path: str, baseline_path: str = None, k: int = 10):
    index = load_index(index_path)
    retriever = FashionRetriever(index)
    baseline = _load_baseline(baseline_path)

    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    rows = []

    for item in queries:
        query = item["query"]
        start = time.time()
        results = retriever.retrieve(query, k=k)
        latency = (time.time() - start) * 1000
        ours_ids = [r.image_id for r in results]
        ours_score = _score_ids(query, ours_ids, index)
        baseline_ids = baseline.get(query, [])
        baseline_score = _score_ids(query, baseline_ids[:k], index) if baseline_ids else None

        rows.append(
            {
                "query": query,
                "ours_score": None if ours_score is None else round(ours_score, 4),
                "baseline_score": None if baseline_score is None else round(baseline_score, 4),
                "latency_ms": round(latency, 1),
                "top_ids": ours_ids,
            }
        )

    ours_values = [row["ours_score"] for row in rows if row["ours_score"] is not None]
    ours_avg = float(np.mean(ours_values)) if ours_values else None
    baseline_values = [row["baseline_score"] for row in rows if row["baseline_score"] is not None]
    baseline_avg = float(np.mean(baseline_values)) if baseline_values else None

    return {
        "rows": rows,
        "ours_avg": None if ours_avg is None else round(ours_avg, 4),
        "baseline_avg": None if baseline_avg is None else round(baseline_avg, 4),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="output/index.pkl")
    parser.add_argument("--queries", default="eval/queries.json")
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    report = evaluate(args.index, args.queries, args.baseline, args.k)
    print(json.dumps(report, indent=2))
