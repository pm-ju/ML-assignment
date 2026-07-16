import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss
except Exception:
    faiss = None


@dataclass
class ImageRecord:
    image_id: str
    path: str
    file_name: str = ""
    category: str = "unknown"
    supercategory: str = "unknown"
    colors: Dict[str, float] = field(default_factory=dict)
    scene_scores: Dict[str, float] = field(default_factory=dict)
    regions: List[Dict] = field(default_factory=list)


class FactorIndex:
    def __init__(self, embed_dim: int = 512):
        self.embed_dim = embed_dim
        self.indices: Dict[str, object] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.id_maps: Dict[str, List[str]] = {}
        self.images: Dict[str, ImageRecord] = {}

    def add(self, factor: str, embeddings: np.ndarray, image_ids: List[str]):
        embeddings = embeddings.astype(np.float32)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        self.embeddings[factor] = embeddings
        self.id_maps[factor] = list(image_ids)

        if faiss is not None:
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            self.indices[factor] = index

    def add_metadata(self, records: List[Dict]):
        for rec in records:
            self.images[rec["image_id"]] = ImageRecord(
                image_id=rec["image_id"],
                path=rec.get("path", ""),
                file_name=rec.get("file_name", ""),
                category=rec.get("category", "unknown"),
                supercategory=rec.get("supercategory", "unknown"),
                colors=rec.get("colors", {}),
                scene_scores=rec.get("scene_scores", {}),
                regions=rec.get("regions", []),
            )

    def get_image(self, image_id: str) -> Optional[ImageRecord]:
        return self.images.get(image_id)

    def get_all_image_ids(self) -> List[str]:
        return list(self.images.keys())

    def get_embeddings(self, factor: str) -> Optional[np.ndarray]:
        return self.embeddings.get(factor)

    def get_id_map(self, factor: str) -> List[str]:
        return self.id_maps.get(factor, [])

    def search(self, factor: str, query_embedding: np.ndarray, k: int = 50) -> List[Tuple[str, float]]:
        embeddings = self.embeddings.get(factor)
        id_map = self.id_maps.get(factor, [])

        if embeddings is None or len(id_map) == 0:
            return []

        query = query_embedding.astype(np.float32).reshape(1, -1)
        query = query / (np.linalg.norm(query) + 1e-8)

        if faiss is not None and factor in self.indices:
            search_k = min(k, self.indices[factor].ntotal)
            scores, indices = self.indices[factor].search(query, search_k)
            return [(id_map[i], float(s)) for s, i in zip(scores[0], indices[0]) if i >= 0 and i < len(id_map)]

        scores = (query @ embeddings.T).flatten()
        ranked = np.argsort(scores)[::-1][:k]
        return [(id_map[i], float(scores[i])) for i in ranked]

    def count(self) -> Dict[str, int]:
        return {factor: len(ids) for factor, ids in self.id_maps.items()}

    def save(self, path: str):
        data = {
            "embed_dim": self.embed_dim,
            "embeddings": self.embeddings,
            "id_maps": self.id_maps,
            "images": {k: v.__dict__ for k, v in self.images.items()},
        }

        if faiss is not None:
            data["indices"] = {k: faiss.serialize_index(v) for k, v in self.indices.items()}

        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.embed_dim = data.get("embed_dim", 512)
        self.embeddings = data.get("embeddings", {})
        self.id_maps = data.get("id_maps", {})
        self.images = {k: ImageRecord(**v) for k, v in data.get("images", {}).items()}
        self.indices = {}

        if faiss is not None and "indices" in data:
            self.indices = {k: faiss.deserialize_index(v) for k, v in data["indices"].items()}
        elif faiss is not None:
            for factor, embeddings in self.embeddings.items():
                index = faiss.IndexFlatIP(embeddings.shape[1])
                index.add(embeddings.astype(np.float32))
                self.indices[factor] = index


def load_index(index_path: str) -> FactorIndex:
    index = FactorIndex()
    index.load(index_path)
    return index
