"""Ba phương pháp retrieval của Day 7, dùng chung một tập chunk và metadata.

1. Dense: query embedding local bằng Sentence Transformers (paraphrase-multilingual-MiniLM-L12-v2, 384 chiều) tìm trong
   Chroma cosine (index day7_dense, chunk giống Day 6). Là phương pháp
   bắt buộc chạy trên vector store Day 6.
2. Lexical TF-IDF: dùng TfidfIndex của starter (tokenizer \\w+, NFC, casefold).
3. Hybrid: hợp nhất hai bảng xếp hạng bằng RRF (Reciprocal Rank Fusion, k=60);
   RRF gộp theo rank nên không cần chuẩn hóa score giữa hai retriever.

Filter metadata (period) chỉ áp dụng khi câu hỏi cung cấp giá trị trước khi
truy xuất (trường filter trong benchmark); không suy diễn từ gold.
"""
from __future__ import annotations

from typing import Any

import chromadb
from chromadb.config import Settings

from .dataio import TfidfIndex, pack_context  # re-export để test và run scripts dùng chung

RRF_K = 60
CANDIDATES_PER_LEG = 10


def rrf_fuse(rankings: list[list[str]], k: int = RRF_K, top_k: int = 5) -> list[tuple[str, float]]:
    """Hợp nhất nhiều bảng xếp hạng bằng RRF; trả (chunk_id, rrf_score) giảm dần."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for pos, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + pos + 1)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]


class DenseRetriever:
    """Query embedding rồi tìm cosine trong collection day7_dense."""

    def __init__(self, store_path: str, collection: str, client: Any) -> None:
        self._client = client
        self.last_query_tokens = 0
        self.col = chromadb.PersistentClient(path=store_path, settings=Settings(anonymized_telemetry=False)) \
            .get_or_create_collection(name=collection, metadata={"hnsw:space": "cosine"})

    def search(self, query: str, *, filters: dict[str, Any] | None, top_k: int) -> list[dict[str, Any]]:
        vector = self._client.embed_query(query)
        self.last_query_tokens = getattr(self._client, "last_usage", {}).get("tokens", 0)
        where = None
        if filters and filters.get("field") == "period" and filters.get("value"):
            where = {"period": {"$eq": filters["value"]}}
        result = self.col.query(query_embeddings=[vector], n_results=top_k, where=where,
                                include=["documents", "metadatas", "distances"])
        items = []
        for i in range(len(result["ids"][0])):
            meta = result["metadatas"][0][i]
            items.append({
                "id": result["ids"][0][i],
                "text": result["documents"][0][i],
                "source_id": meta.get("document_id"),
                "source": meta.get("source"),
                "page": meta.get("page"),
                "metadata": {"title": meta.get("title", ""), "period": meta.get("period", "")},
                "score": round(1.0 - result["distances"][0][i], 6),
            })
        return items


class TfidfRetriever:
    """Lexical retrieval; khi có filter, chỉ chấm điểm trên chunk đúng period.

    Index TF-IDF được dùng lại cho mỗi giá trị period (corpus chỉ có 4 giá trị
    nên chi phí xây lại không đáng kể; index None dùng chung).
    """

    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        self._indexes: dict[str, TfidfIndex] = {"": TfidfIndex(chunks)}
        self.last_query_tokens = 0

    def _index(self, period: str) -> TfidfIndex:
        if period not in self._indexes:
            subset = [c for c in self.chunks if c["metadata"].get("period") == period]
            self._indexes[period] = TfidfIndex(subset)
        return self._indexes[period]

    def search(self, query: str, *, filters: dict[str, Any] | None, top_k: int) -> list[dict[str, Any]]:
        period = filters.get("value", "") if filters and filters.get("field") == "period" else ""
        results = self._index(period).search(query, top_k=top_k)
        items = []
        for r in results:
            items.append({
                "id": r["id"], "text": r["text"], "source_id": r["source_id"],
                "source": r["source"], "page": r["page"], "metadata": r["metadata"],
                "score": round(r["score"], 6),
            })
        return items


class HybridRetriever:
    """Dense + TF-IDF, hợp nhất bằng RRF; filter áp dụng cho cả hai nhánh."""

    def __init__(self, dense: DenseRetriever, lexical: TfidfRetriever) -> None:
        self.dense = dense
        self.lexical = lexical

    def search(self, query: str, *, filters: dict[str, Any] | None, top_k: int) -> list[dict[str, Any]]:
        dense = self.dense.search(query, filters=filters, top_k=CANDIDATES_PER_LEG)
        self.last_query_tokens = getattr(self.dense, "last_query_tokens", 0)
        lexical = self.lexical.search(query, filters=filters, top_k=CANDIDATES_PER_LEG)
        fused = rrf_fuse([[x["id"] for x in dense], [x["id"] for x in lexical]], top_k=top_k)
        pool = {x["id"]: x for x in dense + lexical}
        dense_rank = {x["id"]: p for p, x in enumerate(dense, start=1)}
        lexical_rank = {x["id"]: p for p, x in enumerate(lexical, start=1)}
        items = []
        for cid, score in fused:
            item = pool[cid]
            item["score"] = round(score, 6)
            item["metadata"] = {**item["metadata"], "dense_rank": dense_rank.get(cid),
                                "tfidf_rank": lexical_rank.get(cid)}
            items.append(item)
        return items
