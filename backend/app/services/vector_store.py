import faiss
import numpy as np
import os
import pickle

INDEX_PATH = "faiss.index"
META_PATH = "chunks.pkl"


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks = []

        if os.path.exists(INDEX_PATH):
            self._load()

    def add(self, embeddings, texts):
        embeddings = np.asarray(embeddings, dtype="float32")

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        self.index.add(embeddings)
        self.chunks.extend(texts)
        self._save()

    def search(self, query_embedding, top_k=5):
        query_embedding = np.array([query_embedding]).astype("float32")
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx_val, i in enumerate(indices[0]):
            # Skip invalid index
            if i < 0 or i >= len(self.chunks):
                continue

            results.append({
                "content": self.chunks[i],
                "score": float(scores[0][idx_val])
            })

        return results


    def _save(self):
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "wb") as f:
            pickle.dump(self.chunks, f)

    def _load(self):
        self.index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "rb") as f:
            self.chunks = pickle.load(f)
