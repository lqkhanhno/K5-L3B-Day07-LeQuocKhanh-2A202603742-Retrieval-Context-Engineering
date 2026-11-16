"""Đọc dữ liệu chung và chuẩn hóa văn bản.

Nguồn chunk: data/chunks.json xuất từ vector store Day 6 (scripts/export_chunks.py).
Benchmark: benchmark/questions.json theo schema trong data/README.md.
"""
from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

STARTER = Path(__file__).resolve().parents[1]
ROOT = STARTER.parent
sys.path.insert(0, str(STARTER))  # để dùng TfidfIndex và pack_context từ starter/baseline.py

from baseline import TfidfIndex, pack_context  # noqa: E402


def norm(text: str) -> str:
    """Chuẩn hóa giống Day 6: NFC, gập khoảng trắng, casefold."""
    return " ".join(unicodedata.normalize("NFC", text).split()).casefold()


def load_chunks(path: Path | None = None) -> list[dict]:
    return json.loads((path or ROOT / "data" / "chunks.json").read_text(encoding="utf-8"))


def load_questions(path: Path | None = None) -> list[dict]:
    return json.loads((path or ROOT / "benchmark" / "questions.json").read_text(encoding="utf-8"))


def evidence_hit(item: dict, gold: list[dict]) -> bool:
    """Hit khi chunk đúng nguồn/trang và chứa đủ evidence (chuẩn hóa như Day 6)."""
    for g in gold:
        if item["source_id"] == g["source_id"] and item["page"] == g["page"] \
                and norm(g["evidence"]) in norm(item["text"]):
            return True
    return False
