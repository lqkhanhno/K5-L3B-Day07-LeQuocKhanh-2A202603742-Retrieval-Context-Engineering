"""Chạy agent thật cho một cấu hình và split.

Agent chạy trên toàn bộ câu hỏi của split (gồm cả câu thiếu bằng chứng để đo
abstention). Mỗi câu lưu: status, câu trả lời, citations, từng tool call
(query, filter, kết quả, context), thời gian, token và chi phí; lỗi provider
được ghi rõ và không tính vào chất lượng.

Chạy từ thư mục starter:
    .venv/Scripts/python.exe -m src.run_agent --config agent-dense-top3 --split dev
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from .agent import AgentController, AgentRetriever
from .dataio import ROOT, load_chunks, load_questions
from .openai_client import OpenAIClient, load_config, make_embedder
from .retrievers import DenseRetriever, TfidfRetriever

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Cấu hình agent: baseline (dense top-3, phương pháp bắt buộc) và cấu hình chốt
# trên dev theo tiêu chí công bố trước (Hit@3, rồi CER, rồi thời gian/chi phí).
# max_searches=2 (thay vì 4 mặc định của khung starter) vì corpus 38 tài liệu
# ngắn, câu hỏi chủ yếu single-hop, và để khống chế chi phí token của gói miễn phí.
AGENT_CONFIGS: dict[str, dict] = {
    "agent-dense-top3": dict(method="dense", top_k=3, budget=4500, use_filter=False, max_searches=2),
    "agent-tfidf-top3": dict(method="tfidf", top_k=3, budget=4500, use_filter=False, max_searches=2),
}


def build_agent_retriever(method: str, chunks: list[dict], embedder, store: Path) -> AgentRetriever:
    if method == "dense":
        return AgentRetriever(DenseRetriever(str(store), "day7_dense", embedder), embedder)
    if method == "tfidf":
        return AgentRetriever(TfidfRetriever(chunks), embedder)
    raise ValueError(f"Phương pháp agent không hợp lệ: {method}")


def compact_results(results: list[dict]) -> list[dict]:
    """Trace giữ id/score/nguồn/trang/metadata; văn bản đầy đủ nằm trong chunks.json."""
    return [{"id": r["id"], "score": r.get("score"), "source_id": r["source_id"],
             "page": r["page"], "metadata": r["metadata"]} for r in results]


def load_trace_file(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return data if isinstance(data, list) else []


def run_one(q: dict, retriever: AgentRetriever, chat_client: OpenAIClient, config: dict) -> dict:
    from agentic import run_agent  # khung starter, đã thêm starter vào sys.path qua dataio

    controller = AgentController(chat_client)
    retriever.query_embed_tokens = 0
    filters = q.get("filter") if config["use_filter"] else None
    started = time.perf_counter()
    try:
        result = run_agent(q["question"], controller, retriever, filters=filters,
                           max_searches=config["max_searches"], top_k=config["top_k"],
                           budget=config["budget"])
        elapsed_ms = (time.perf_counter() - started) * 1000
        events = []
        for e in result["events"]:
            compact = {"event": e["event"]}
            if e["event"] == "search":
                compact.update({"query": e["query"], "filter": e.get("filter"),
                                "results": compact_results(e["results"]),
                                "context_ids": e["context_ids"],
                                "context_chars": len(e["context"])})
            events.append(compact)
        tokens = controller.usage
        embed_tokens = retriever.query_embed_tokens
        cost = chat_client.chat_cost(tokens["input_tokens"], tokens["output_tokens"]) + chat_client.embed_cost(embed_tokens)
        return {
            "query_id": q["id"], "split": q["split"], "type": q.get("type"),
            "question": q["question"], "status": result["status"],
            "answer": result.get("answer"), "citations": result.get("citations", []),
            "events": events, "elapsed_ms": round(elapsed_ms, 1), "model": chat_client.agent_model,
            "config": config["name"], "top_k": config["top_k"], "budget_chars": config["budget"],
            "filters": filters, "search_calls": sum(e["event"] == "search" for e in result["events"]),
            "max_searches": config["max_searches"],
            "chat_tokens": tokens, "query_embed_tokens": embed_tokens,
            "cost_usd": round(cost, 6),
            "decisions": controller.decisions,
            "model_calls": controller.model_calls,
        }
    except Exception as exc:  # lỗi provider/store: báo riêng, không fallback
        elapsed_ms = (time.perf_counter() - started) * 1000
        tokens = dict(controller.usage)
        embed_tokens = retriever.query_embed_tokens
        cost = chat_client.chat_cost(tokens["input_tokens"], tokens["output_tokens"]) + chat_client.embed_cost(embed_tokens)
        return {
            "query_id": q["id"], "split": q["split"], "type": q.get("type"),
            "question": q["question"], "status": "error", "answer": None,
            "citations": [], "events": [], "elapsed_ms": round(elapsed_ms, 1), "model": chat_client.agent_model,
            "config": config["name"], "top_k": config["top_k"], "budget_chars": config["budget"],
            "filters": filters, "search_calls": sum(e["event"] == "search" for e in controller.decisions),
            "max_searches": config["max_searches"],
            "chat_tokens": tokens, "query_embed_tokens": embed_tokens,
            "cost_usd": round(cost, 6),
            "decisions": controller.decisions,
            "model_calls": controller.model_calls,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy agent Day 7 trên một split")
    parser.add_argument("--config", required=True, choices=sorted(AGENT_CONFIGS))
    parser.add_argument("--split", required=True, choices=("dev", "test"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "agent")
    args = parser.parse_args()

    config = {**AGENT_CONFIGS[args.config], "name": args.config}
    chunks = load_chunks()
    questions = [q for q in load_questions() if q["split"] == args.split]
    full_config = load_config()
    chat_client = OpenAIClient(full_config)
    embedder = make_embedder(full_config)
    retriever = build_agent_retriever(config["method"], chunks, embedder, ROOT / "starter" / "vector_store")

    out = args.output_dir / f"{args.config}-{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    trace_path = out / "traces.json"
    traces = load_trace_file(trace_path)
    completed = {t.get("query_id") for t in traces if isinstance(t, dict)}

    for q in questions:
        if q["id"] in completed:
            continue
        trace = run_one(q, retriever, chat_client, config)
        traces.append(trace)
        trace_path.write_text(json.dumps(traces, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        completed.add(q["id"])
        print(f"  {q['id']}: {trace['status']} (search={trace.get('search_calls', '-')}, {trace.get('elapsed_ms', 0):.0f} ms)")

    ok = [t for t in traces if t["status"] != "error"]
    summary = {
        "config": args.config, "split": args.split, "n": len(traces),
        "answered": sum(t["status"] == "answered" for t in traces),
        "abstain": sum(t["status"] == "abstain" for t in traces),
        "errors": sum(t["status"] == "error" for t in traces),
        "search_calls_total": sum(t.get("search_calls", 0) for t in traces),
        "median_ms": round(statistics.median([t["elapsed_ms"] for t in ok]), 1) if ok else None,
        "chat_input_tokens": sum(t.get("chat_tokens", {}).get("input_tokens", 0) for t in traces),
        "chat_output_tokens": sum(t.get("chat_tokens", {}).get("output_tokens", 0) for t in traces),
        "query_embed_tokens": sum(t.get("query_embed_tokens", 0) for t in traces),
        "cost_usd": round(sum(t.get("cost_usd", 0) for t in traces), 6),
        "model": chat_client.agent_model,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
