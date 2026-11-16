"""Kiểm chứng benchmark Day 7 trước khi chạy pipeline.

Kiểm tra:
1. chunks.json đúng schema Day 7 (id, text, source_id, source, page, metadata), ID duy nhất.
2. Câu có đáp án: evidence nằm trọn trong một chunk đúng nguồn/trang (so khớp sau
   chuẩn hóa NFC, gập khoảng trắng, casefold) — cùng quy tắc với hàm hit khi chấm.
3. Câu thiếu bằng chứng: có absence_reason, và các absence_check kiểu
   absent_token/absent_regex phải thực sự không xuất hiện trong corpus; kiểu manual
   được ghi lại để người chấm đối chiếu.

Cách chạy từ thư mục starter:
    .venv/Scripts/python.exe scripts/verify_benchmark.py --chunks ../data/chunks.json --questions ../benchmark/questions.json
"""
import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split()).casefold()


def main() -> None:
    parser = argparse.ArgumentParser(description="Kiểm chứng benchmark Day 7")
    parser.add_argument("--chunks", required=True, type=Path)
    parser.add_argument("--questions", required=True, type=Path)
    args = parser.parse_args()

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    if not isinstance(chunks, list) or not chunks or not isinstance(questions, list) or not questions:
        raise SystemExit("chunks/questions phải là JSON array không rỗng")

    errors: list[str] = []
    ids = set()
    qids = []
    split_counts = Counter()
    for c in chunks:
        missing = [k for k in ("id", "text", "source_id", "source", "page", "metadata") if not c.get(k)]
        if missing:
            errors.append(f"chunk {c.get('id')}: thiếu {', '.join(missing)}")
        cid = c.get("id")
        if not isinstance(cid, str) or not cid.strip():
            errors.append("chunk id phải là chuỗi không rỗng")
        elif cid in ids:
            errors.append(f"chunk ID trùng: {cid}")
        else:
            ids.add(cid)

    corpus = "\n".join(c["text"] for c in chunks)
    n_by_split = {"dev": {"answerable": 0, "abstain": 0}, "test": {"answerable": 0, "abstain": 0}}
    for q in questions:
        qid = q.get("id")
        if not isinstance(qid, str) or not qid.strip():
            errors.append("question id phải là chuỗi không rỗng")
        elif qid in qids:
            errors.append(f"question ID trùng: {qid}")
        else:
            qids.append(qid)
        split = q.get("split")
        if split not in ("dev", "test") or not isinstance(q.get("question"), str) or not q["question"].strip():
            errors.append(f"{qid}: split/question không hợp lệ")
            continue
        split_counts[split] += 1
        if q.get("answerable") is False:
            n_by_split[split]["abstain"] += 1
            if not q.get("absence_reason"):
                errors.append(f"{qid}: thiếu absence_reason")
            for check in q.get("absence_check", []):
                if not isinstance(check, dict):
                    errors.append(f"{qid}: absence_check không hợp lệ")
                    continue
                kind, value = check.get("kind"), check.get("value", "")
                if kind == "absent_token" and value and value in corpus:
                    errors.append(f"{qid}: chuỗi '{value}' vẫn còn xuất hiện trong corpus")
                elif kind == "absent_regex" and value and re.search(value, corpus):
                    errors.append(f"{qid}: regex '{value}' vẫn còn khớp trong corpus")
                elif kind not in ("absent_token", "absent_regex", "manual"):
                    errors.append(f"{qid}: absence_check kind không hợp lệ: {kind}")
            continue
        n_by_split[split]["answerable"] += 1
        if not isinstance(q.get("expected_answer"), str) or not q["expected_answer"].strip():
            errors.append(f"{qid}: thiếu expected_answer")
        gold = q.get("gold")
        if not isinstance(gold, list) or not gold:
            errors.append(f"{qid}: thiếu gold")
            continue
        for g in gold:
            if not isinstance(g, dict) or not isinstance(g.get("evidence"), str) or not g["evidence"].strip():
                errors.append(f"{qid}: gold evidence rỗng hoặc không hợp lệ")
                continue
            evidence = norm(g.get("evidence", ""))
            found = any(c["source_id"] == g.get("source_id") and c["page"] == g.get("page")
                        and evidence in norm(c["text"]) for c in chunks)
            if not found:
                errors.append(f"{qid}: evidence không nằm trọn trong chunk nào đúng nguồn/trang "
                              f"({g.get('source_id')} trang {g.get('page')})")
        f = q.get("filter")
        if f and (f.get("field") != "period" or not f.get("value")):
            errors.append(f"{qid}: filter không hợp lệ (chỉ cho phép period)")

    print(f"Chunks: {len(chunks)} (ID duy nhất: {len(ids)})")
    for split, stat in n_by_split.items():
        print(f"  {split}: {stat['answerable']} câu có đáp án, {stat['abstain']} câu thiếu bằng chứng")
    if split_counts["dev"] < 17 or split_counts["test"] < 7:
        errors.append(f"số câu theo split không đủ: dev={split_counts['dev']} test={split_counts['test']}")
    if errors:
        print("LỖI:")
        for e in errors:
            print(" -", e)
        raise SystemExit(1)
    print("Kiểm chứng đạt: mọi evidence nằm trong chunk đúng nguồn/trang, các kiểm tra absence đều pass.")


if __name__ == "__main__":
    main()
