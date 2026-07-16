from typing import Dict, List

import numpy as np

from indexer.embedding_extractor import FashionClipExtractor
from indexer.vector_store import load_index
from retriever.constraint_reranker import RankedResult, rerank_constraints
from retriever.query_parser import QueryParser


class FashionRetriever:
    def __init__(self, index, clip_extractor=None):
        self.index = index
        self.extractor = clip_extractor or FashionClipExtractor()

    def _query_vectors(self, query: str):
        parsed = QueryParser.parse(query)
        vectors = {
            "global": self.extractor.encode_single_text(query),
        }

        region_texts = []
        for binding in parsed.bindings:
            region_texts.append(f"{binding.color} {binding.garment}")
        for garment in parsed.garments:
            if not any(binding.garment == garment for binding in parsed.bindings):
                region_texts.append(garment)

        if region_texts:
            vectors["region"] = self.extractor.encode_text(region_texts).mean(axis=0)
            vectors["region"] = vectors["region"] / (np.linalg.norm(vectors["region"]) + 1e-8)

        if parsed.environment or parsed.style:
            target = parsed.environment or parsed.style
            vectors["scene"] = self.extractor.encode_single_text(f"a person in a {target} setting")

        return parsed, vectors

    def retrieve(self, query: str, k: int = 10, search_k: int = 120) -> List[RankedResult]:
        parsed, vectors = self._query_vectors(query)
        candidate_scores: Dict[str, float] = {}
        candidate_weights: Dict[str, float] = {}
        weights = {"global": 0.45, "region": 0.35, "scene": 0.20}

        for factor, vector in vectors.items():
            hits = self.index.search(factor, vector, search_k)
            weight = weights.get(factor, 0.1)

            for image_id, score in hits:
                candidate_scores[image_id] = candidate_scores.get(image_id, 0.0) + weight * score
                candidate_weights[image_id] = candidate_weights.get(image_id, 0.0) + weight

        normalized = {}
        for image_id, score in candidate_scores.items():
            weight = candidate_weights.get(image_id, 0.0)
            if weight > 0:
                normalized[image_id] = score / weight

        return rerank_constraints(parsed, normalized, self.index, top_k=k)


def load_retriever(index_path: str):
    return FashionRetriever(load_index(index_path))
