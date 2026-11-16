"""Xuất chunk từ vector store Day 6 (Chroma, collection day6_C) ra JSON Day 7.

Day 6 đã chia chunk và lưu cùng embedding; Day 7 dùng lại đúng chunk, ID,
nguồn và metadata đó. Script chỉ đọc và đổi tên trường, không gọi API,
không suy diễn thêm thông tin.

Cách chạy từ thư mục starter:
    .venv/Scripts/python.exe scripts/export_chunks.py \
        --store <đường dẫn thư mục vector_store của Day 6> \
        --collection day6_C \
        --output ../data/chunks.json
"""
import argparse
import json
import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description="Xuất chunk Day 6 sang schema Day 7")
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--collection", default="day6_C")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=str(args.store), settings=Settings(anonymized_telemetry=False))
    col = client.get_collection(args.collection)
    raw = col.get(include=["documents", "metadatas"])
    if len(raw["ids"]) != len(set(raw["ids"])):
        raise SystemExit("Chunk ID bị trùng trong collection")

    chunks = []
    for cid, text, meta in zip(raw["ids"], raw["documents"], raw["metadatas"]):
        chunks.append({
            "id": cid,
            "text": text,
            "source_id": meta.get("document_id"),
            "source": meta.get("source"),
            "page": meta.get("page"),
            "metadata": {"title": meta.get("title", ""), "period": meta.get("period", "")},
        })
    chunks.sort(key=lambda c: (c["source_id"], c["page"], c["id"]))

    total_chars = sum(len(c["text"]) for c in chunks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(chunks, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Đã xuất {len(chunks)} chunk ({total_chars} ký tự) từ collection {args.collection} -> {args.output}")
    periods = sorted({c["metadata"]["period"] for c in chunks})
    print("Giá trị period:", ", ".join(periods))


if __name__ == "__main__":
    main()
