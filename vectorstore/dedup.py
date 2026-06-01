from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from config.settings import VECTOR_STORE_PATH
from config.rubric import CHALK_DEDUP_THRESHOLD

_MODEL_NAME = "all-MiniLM-L6-v2"
_DIM = 384


class DedupIndex:
    def __init__(self, path: str = VECTOR_STORE_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.model = SentenceTransformer(_MODEL_NAME)
        self.index = (
            faiss.read_index(str(self.path))
            if self.path.exists()
            else faiss.IndexFlatIP(_DIM)
        )

    def _embed(self, text: str) -> np.ndarray:
        return self.model.encode([text], normalize_embeddings=True).astype("float32")

    def is_duplicate(self, text: str, threshold: float = CHALK_DEDUP_THRESHOLD) -> tuple[bool, float]:
        if self.index.ntotal == 0:
            return False, 0.0
        vec = self._embed(text)
        D, _ = self.index.search(vec, 1)
        similarity = float(D[0][0])
        return similarity >= threshold, round(similarity, 4)

    def add(self, text: str) -> None:
        vec = self._embed(text)
        self.index.add(vec)
        faiss.write_index(self.index, str(self.path))

    @property
    def size(self) -> int:
        return self.index.ntotal