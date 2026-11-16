"""Tạo index Chroma cho Day 7 từ chunks.json (đã xuất từ Day 6).

Dùng lại đúng chunk, ID và metadata Day 6; embedding model local paraphrase-multilingual-MiniLM-L12-v2 (384 chiều) nên query embedding tương thích với index.
Metric cosine, collection day7_dense trong starter/vector_store.

Script idempotent: nếu collection đã đủ số chunk (1909) thì bỏ qua phần
embedding để không tốn chi phí; xóa collection để nạp lại từ đầu.

Cách chạy từ thư mục starter:
    .venv/Scripts/python.exe scripts/build_index.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # starter/

import chromadb
from chromadb.config import Settings

from src.openai_client import make_embedder

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BATCH = 96
EXPECTED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EXPECTED_DIMENSIONS = 384


def main() -> None:
    parser = argparse.ArgumentParser(description="Nạp chunk vào Chroma Day 7")
    parser.add_argument("--chunks", type=Path, default=Path("../data/chunks.json"))
    parser.add_argument("--store", type=Path, default=Path("vector_store"))
    parser.add_argument("--collection", default="day7_dense")
    args = parser.parse_args()

    chunks = json.loads(args.chunks.resolve().read_text(encoding="utf-8"))
    client_chroma = chromadb.PersistentClient(path=str(args.store.resolve()),
                                              settings=Settings(anonymized_telemetry=False))
    col = client_chroma.get_or_create_collection(name=args.collection, metadata={"hnsw:space": "cosine"})
    if col.count() == len(chunks):
        try:
            info = json.loads((Path("../results/index_info.json")).read_text(encoding="utf-8"))
            if info.get("embedding_model") != EXPECTED_MODEL or info.get("dimensions") != EXPECTED_DIMENSIONS or info.get("chunk_count") != len(chunks):
                raise SystemExit("Collection đủ số chunk nhưng index_info không khớp model/dimension/chunk_count; xóa vector_store để nạp lại.")
        except FileNotFoundError:
            print(f"Collection {args.collection} đã có {col.count()} chunk, bỏ qua (không gọi API).")
        else:
            print(f"Collection {args.collection} đã có {col.count()} chunk, bỏ qua (không gọi API).")
        return
    if col.count() > 0:
        raise SystemExit(f"Collection có {col.count()} chunk khác {len(chunks)}; xóa collection để nạp lại.")

    client = make_embedder()
    actual_model = getattr(client, "embedding_model", getattr(client, "model_name", None))
    if actual_model != EXPECTED_MODEL:
        raise SystemExit(f"Embedding model không khớp: {actual_model} != {EXPECTED_MODEL}")
    started = time.perf_counter()
    tokens = 0
    vectors = None
    for start in range(0, len(chunks), BATCH):
        batch = chunks[start:start + BATCH]
        vectors = client.embed_documents([c["text"] for c in batch])
        tokens += client.last_usage["tokens"]
        if vectors and len(vectors[0]) != EXPECTED_DIMENSIONS:
            raise SystemExit(f"Vector dimension không khớp: {len(vectors[0])} != {EXPECTED_DIMENSIONS}")
        col.upsert(
            ids=[c["id"] for c in batch],
            embeddings=vectors,
            documents=[c["text"] for c in batch],
            metadatas=[{
                "document_id": c["source_id"],
                "source": c["source"],
                "page": c["page"],
                "title": c["metadata"].get("title", ""),
                "period": c["metadata"].get("period", ""),
            } for c in batch],
        )
        done = min(start + BATCH, len(chunks))
        print(f"  đã nạp {done}/{len(chunks)} chunk ({tokens} token embedding)")
    elapsed = time.perf_counter() - started

    assert col.count() == len(chunks), "nạp lại không đủ số chunk"
    info = {
        "collection": args.collection,
        "store_path": str(args.store.resolve()),
        "chunk_count": col.count(),
        "embedding_model": actual_model,
        "dimensions": len(vectors[0]) if vectors else EXPECTED_DIMENSIONS,
        "metric": "cosine",
        "index_seconds": round(elapsed, 1),
        "doc_embed_tokens": tokens,
        "doc_embed_cost_usd": round(client.embed_cost(tokens), 6),
    }
    out = Path("../results/index_info.json").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
