"""Gom dữ liệu chấm thủ công: trace + văn bản chunk được trích dẫn + gold.

Script chỉ phục vụ việc ĐỌC để người chấm (tác giả bài) ghi nhãn reviews.json;
không tự chấm, không ghi nhãn giùm. In ra từng câu: câu hỏi, đáp án model,
citations kèm văn bản chunk tương ứng từ data/chunks.json, gold evidence.

Cách chay tu thu muc starter:
    .venv/Scripts/python.exe scripts/label_view.py --trace ../results/agent/agent-tfidf-top3-dev/traces.json --split dev
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description="Xem trace để chấm thủ công")
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("dev", "test"))
    args = parser.parse_args()

    chunks = {c["id"]: c for c in json.loads((ROOT / "data" / "chunks.json").read_text(encoding="utf-8"))}
    questions = {q["id"]: q for q in json.loads((ROOT / "benchmark" / "questions.json").read_text(encoding="utf-8"))
                 if q["split"] == args.split}
    traces = json.loads(args.trace.resolve().read_text(encoding="utf-8"))

    for t in traces:
        q = questions[t["query_id"]]
        print(f"===== {t['query_id']} ({q.get('type')}) | status: {t['status']} =====")
        print(f"CÂU HỎI: {q['question']}")
        if q.get("answerable") is False:
            print(f"KỲ VỌNG: từ chối ({q.get('absence_reason', '')[:120]})")
        else:
            print(f"ĐÁP ÁN ĐÚNG: {q['expected_answer']}")
            for g in q["gold"]:
                print(f"GOLD: {g['source_id']} trang {g['page']} | {g['evidence'][:120]}")
        print(f"MODEL TRẢ LỜI: {(t.get('answer') or '(không)')[:300]}")
        print(f"CITATIONS: {t.get('citations')}")
        for cid in t.get("citations", []):
            c = chunks.get(cid)
            if c is None:
                print(f"  [{cid}] KHÔNG TỒN TẠI trong chunks.json")
            else:
                print(f"  [{cid}] {c['source_id']} trang {c['page']} | {c['text'][:200]}")
        for d in t.get("decisions", []):
            if d["action"] == "search":
                print(f"SEARCH: {d['query']}")
            else:
                print(f"SUBMIT: {(d.get('answer') or '')[:160]} | cites={d.get('citations')}")
        if t.get("error"):
            print(f"LỖI: {t['error'][:160]}")
        print()


if __name__ == "__main__":
    main()
