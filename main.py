import json
import os
import time

import torch

from indexer.vector_store import load_index
from retriever.retriever import FashionRetriever

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)

QUERIES = [
    "A person in a bright yellow raincoat.",
    "Professional business attire inside a modern office.",
    "Someone wearing a blue shirt sitting on a park bench.",
    "Casual weekend outfit for a city walk.",
    "A red tie and a white shirt in a formal setting.",
]


def run_query(query: str, retriever: FashionRetriever, k: int = 10):
    start = time.time()
    results = retriever.retrieve(query, k=k)
    latency = (time.time() - start) * 1000

    return {
        "query": query,
        "retrieved": [
            {
                "id": row.image_id,
                "score": round(row.score, 4),
                "parts": row.parts,
                "path": row.metadata.get("path"),
            }
            for row in results
        ],
        "latency_ms": round(latency, 1),
    }


def main():
    index_path = "output/index.pkl"
    print(f"Loading index: {index_path}")
    index = load_index(index_path)
    retriever = FashionRetriever(index)
    print(f"Loaded {len(index.get_all_image_ids())} images")

    rows = []
    total_latency = 0.0

    print("=" * 60)
    print("Retrieval Results")
    print("=" * 60)

    for i, query in enumerate(QUERIES, 1):
        row = run_query(query, retriever, k=10)
        rows.append(row)
        total_latency += row["latency_ms"]
        print(f"\n[{i}] {query}")
        print(f"    Latency: {row['latency_ms']:.0f}ms")
        print(f"    Top-5: {[r['id'] for r in row['retrieved'][:5]]}")

    print("\n" + "=" * 60)
    print(f"Avg Latency: {total_latency / len(rows):.0f}ms")
    print("=" * 60)

    os.makedirs("output", exist_ok=True)
    with open("output/result.json", "w", encoding="utf-8") as f:
        json.dump({"queries": rows}, f, indent=2)

    print("\nSaved to: output/result.json")


if __name__ == "__main__":
    main()
