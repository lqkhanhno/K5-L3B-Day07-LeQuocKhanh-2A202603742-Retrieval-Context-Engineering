"""Chạy retrieval-only cho các cấu hình Day 7 trên một split.

Những câu thiếu bằng chứng không đưa vào mẫu số retrieval metrics
(theo BENCHMARK.md) nên chỉ chạy trên câu có đáp án.

Mỗi cấu hình xuất results/retrieval/<cấu hình>-<split>/{traces.json, summary.json}
báo Hit@1, Evidence Hit@3, Context Evidence Rate, thời gian và chi phí token.

Cách chạy từ thư mục starter:
    .venv/Scripts/python.exe -m src.run_retrieval --config dense-top3 tfidf-top3 hybrid-top3 --split dev
    .venv/Scripts/python.exe -m src.run_retrieval --config hybrid-top5-filter --split test
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from .dataio import ROOT, evidence_hit, load_chunks, load_questions
from .openai_client import make_embedder
from .retrievers import DenseRetriever, HybridRetriever, TfidfRetriever, pack_context

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Các cấu hình: 3 phương pháp (baseline) + 2 cấu hình cải thiện.
# Budget tính bằng ký tự: chunk Day 6 dài trung bình khoảng 971 ký tự (tối đa
# 1322), nên budget 800 của starter mẫu không vừa chunk nào; chọn 4500 cho
# top-3 và 7500 cho top-5 để vừa hết các chunk chọn (cộng cả header nguồn/trang).
CONFIGS: dict[str, dict] = {
    "dense-top3":         dict(method="dense",  top_k=3, budget=4500, use_filter=False),
    "tfidf-top3":         dict(method="tfidf",  top_k=3, budget=4500, use_filter=False),
    "hybrid-top3":        dict(method="hybrid", top_k=3, budget=4500, use_filter=False),
    "hybrid-top5":        dict(method="hybrid", top_k=5, budget=7500, use_filter=False),
    "hybrid-top5-filter": dict(method="hybrid", top_k=5, budget=7500, use_filter=True),
}


def build_retriever(config: dict, chunks: list[dict], embedder):
    if config["method"] == "dense":
        return DenseRetriever(str((ROOT / "starter" / "vector_store").resolve()), "day7_dense", embedder)
    lexical = TfidfRetriever(chunks)
    if config["method"] == "tfidf":
        return lexical
    return HybridRetriever(DenseRetriever(str((ROOT / "starter" / "vector_store").resolve()),
                                          "day7_dense", embedder), lexical)


def run_config(name: str, config: dict, questions: list[dict], chunks: list[dict],
               embedder, output_dir: Path, args_split: str) -> dict:
    retriever = build_retriever(config, chunks, embedder)
    traces = []
    tokens = 0
    for q in questions:
        if q.get("answerable") is False:
            continue  # câu thiếu bằng chứng không vào mẫu số retrieval metrics
        filters = q.get("filter") if config["use_filter"] else None
        started = time.perf_counter()
        results = retriever.search(q["question"], filters=filters, top_k=config["top_k"])
        tokens += 0 if config["method"] == "tfidf" else getattr(embedder, "last_usage", {}).get("tokens", 0)
        context, context_ids = pack_context(results, budget=config["budget"])
        elapsed_ms = (time.perf_counter() - started) * 1000
        marked = [{**c, "hit": evidence_hit(c, q["gold"])} for c in results]
        marked = [{k: v for k, v in m.items() if k != "text_norm"} for m in marked]
        top3 = marked[:3]
        traces.append({
            "query_id": q["id"], "split": q["split"], "question": q["question"],
            "config": name, "method": config["method"], "top_k": config["top_k"],
            "budget_chars": config["budget"], "filters": filters,
            "gold": q["gold"], "results": marked,
            "context": context, "context_ids": context_ids,
            "elapsed_ms": round(elapsed_ms, 1),
            "hit1": bool(top3 and top3[0]["hit"]),
            "hit3": any(c["hit"] for c in top3),
            "context_hit": any(c["hit"] and c["id"] in context_ids for c in marked),
        })
    n = len(traces)
    summary = {
        "config": name, "method": config["method"], "split": args_split,
        "top_k": config["top_k"], "budget_chars": config["budget"],
        "use_filter": config["use_filter"], "n": n,
        "hit1": f"{sum(t['hit1'] for t in traces)}/{n}",
        "hit3": f"{sum(t['hit3'] for t in traces)}/{n}",
        "context_evidence_rate": f"{sum(t['context_hit'] for t in traces)}/{n}",
        "median_ms": round(statistics.median(t["elapsed_ms"] for t in traces), 1),
        "query_embed_tokens": tokens,
        "cost_usd": round(embedder.embed_cost(tokens), 6),
    }
    out = output_dir / f"{name}-{summary['split']}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy retrieval Day 7 trên một split")
    parser.add_argument("--config", required=True, nargs="+", choices=sorted(CONFIGS))
    parser.add_argument("--split", required=True, choices=("dev", "test"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "retrieval")
    args = parser.parse_args()

    chunks = load_chunks()
    questions = [q for q in load_questions() if q["split"] == args.split]
    embedder = make_embedder()
    total_tokens = 0
    for name in args.config:
        summary = run_config(name, CONFIGS[name], questions, chunks, embedder, args.output_dir, args.split)
        total_tokens += summary["query_embed_tokens"]
        print(json.dumps(summary, ensure_ascii=False))
    print(f"Tổng query embedding tokens: {total_tokens} (khoảng {embedder.embed_cost(total_tokens):.6f} USD)")


if __name__ == "__main__":
    main()
