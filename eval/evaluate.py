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
    scores = []

    for image_id in image_ids:
        score = _score_one(query, image_id, index)
        if score is not None:
            scores.append(score)

    return float(np.mean(scores)) if scores else None


def _score_one(query: str, image_id: str, index):
    parsed = QueryParser.parse(query)
    record = index.get_image(str(image_id))
    if record is None:
        return None
    return (
        0.35 * garment_score(parsed, record)
        + 0.40 * color_score(parsed, record)
        + 0.25 * scene_score(parsed, record)
    )


def _pseudo_relevant_ids(query: str, index, threshold: float):
    relevant = set()
    for image_id in index.get_all_image_ids():
        score = _score_one(query, image_id, index)
        if score is not None and score >= threshold:
            relevant.add(str(image_id))
    return relevant


def _recall_at_k(image_ids: List[str], relevant_ids, k: int):
    if not relevant_ids:
        return None
    retrieved = set(str(x) for x in image_ids[:k])
    return len(retrieved & relevant_ids) / len(relevant_ids)


def _hit_at_k(image_ids: List[str], relevant_ids, k: int):
    if not relevant_ids:
        return None
    retrieved = set(str(x) for x in image_ids[:k])
    return 1.0 if retrieved & relevant_ids else 0.0


def _load_baseline(path: str) -> Dict[str, List[str]]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = {}
    for item in data.get("queries", []):
        result[item["query"]] = [str(row["id"]) for row in item.get("retrieved", [])]
    return result


def evaluate(index_path: str, queries_path: str, baseline_path: str = None, k: int = 10, ks: List[int] = None, threshold: float = 0.75):
    index = load_index(index_path)
    retriever = FashionRetriever(index)
    baseline = _load_baseline(baseline_path)
    ks = ks or [1, 3, 5, 10]

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
        relevant_ids = _pseudo_relevant_ids(query, index, threshold)
        ours_recall = {}
        baseline_recall = {}
        ours_hit = {}
        baseline_hit = {}

        for current_k in ks:
            ours_value = _recall_at_k(ours_ids, relevant_ids, current_k)
            ours_recall[f"R@{current_k}"] = None if ours_value is None else round(ours_value, 4)
            ours_hit_value = _hit_at_k(ours_ids, relevant_ids, current_k)
            ours_hit[f"Hit@{current_k}"] = None if ours_hit_value is None else round(ours_hit_value, 4)

            if baseline_ids:
                baseline_value = _recall_at_k(baseline_ids, relevant_ids, current_k)
                baseline_recall[f"R@{current_k}"] = None if baseline_value is None else round(baseline_value, 4)
                baseline_hit_value = _hit_at_k(baseline_ids, relevant_ids, current_k)
                baseline_hit[f"Hit@{current_k}"] = None if baseline_hit_value is None else round(baseline_hit_value, 4)

        rows.append(
            {
                "query": query,
                "ours_score": None if ours_score is None else round(ours_score, 4),
                "baseline_score": None if baseline_score is None else round(baseline_score, 4),
                "latency_ms": round(latency, 1),
                "pseudo_relevant_count": len(relevant_ids),
                "ours_recall_at_k": ours_recall,
                "baseline_recall_at_k": baseline_recall if baseline_ids else None,
                "ours_hit_at_k": ours_hit,
                "baseline_hit_at_k": baseline_hit if baseline_ids else None,
                "top_ids": ours_ids,
            }
        )

    ours_values = [row["ours_score"] for row in rows if row["ours_score"] is not None]
    ours_avg = float(np.mean(ours_values)) if ours_values else None
    baseline_values = [row["baseline_score"] for row in rows if row["baseline_score"] is not None]
    baseline_avg = float(np.mean(baseline_values)) if baseline_values else None
    avg_recall = {}
    avg_baseline_recall = {}
    avg_hit = {}
    avg_baseline_hit = {}

    for current_k in ks:
        r_key = f"R@{current_k}"
        h_key = f"Hit@{current_k}"
        values = [row["ours_recall_at_k"][r_key] for row in rows if row["ours_recall_at_k"].get(r_key) is not None]
        base_values = [
            row["baseline_recall_at_k"][r_key]
            for row in rows
            if row["baseline_recall_at_k"] and row["baseline_recall_at_k"].get(r_key) is not None
        ]
        hit_values = [row["ours_hit_at_k"][h_key] for row in rows if row["ours_hit_at_k"].get(h_key) is not None]
        base_hit_values = [
            row["baseline_hit_at_k"][h_key]
            for row in rows
            if row["baseline_hit_at_k"] and row["baseline_hit_at_k"].get(h_key) is not None
        ]
        avg_recall[r_key] = None if not values else round(float(np.mean(values)), 4)
        avg_baseline_recall[r_key] = None if not base_values else round(float(np.mean(base_values)), 4)
        avg_hit[h_key] = None if not hit_values else round(float(np.mean(hit_values)), 4)
        avg_baseline_hit[h_key] = None if not base_hit_values else round(float(np.mean(base_hit_values)), 4)

    return {
        "rows": rows,
        "ours_avg": None if ours_avg is None else round(ours_avg, 4),
        "baseline_avg": None if baseline_avg is None else round(baseline_avg, 4),
        "threshold": threshold,
        "ours_recall_at_k_avg": avg_recall,
        "baseline_recall_at_k_avg": avg_baseline_recall if baseline else None,
        "ours_hit_at_k_avg": avg_hit,
        "baseline_hit_at_k_avg": avg_baseline_hit if baseline else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="output/index.pkl")
    parser.add_argument("--queries", default="eval/queries.json")
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ks", default="1,3,5,10")
    parser.add_argument("--threshold", type=float, default=0.75)
    args = parser.parse_args()
    ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]

    report = evaluate(args.index, args.queries, args.baseline, args.k, ks, args.threshold)
    print(json.dumps(report, indent=2))
