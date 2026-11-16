"""Embedding local bằng Sentence Transformers.

GUIDE.md Day 7 cho phép dùng model chạy local. Bài đổi embedding model so với
Day 6 vì key OpenAI đã bị thu hồi và không tạo được key mới; index được xây
lại tương ứng, dùng chung một model cho document và query nên query embedding
tương thích với index. Lý do và số chiều được ghi rõ trong báo cáo.
"""
from __future__ import annotations

from typing import Any

from sentence_transformers import SentenceTransformer


class LocalEmbedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.last_usage: dict[str, int] = {"tokens": 0}  # chạy local, không đếm token API

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(list(texts), batch_size=32,
                                    normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vectors]

    @staticmethod
    def embed_cost(tokens: int) -> float:
        return 0.0
