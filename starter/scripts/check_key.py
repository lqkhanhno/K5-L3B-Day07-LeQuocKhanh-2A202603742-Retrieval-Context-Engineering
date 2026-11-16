"""Kiểm tra nhà cung cấp API trước khi chạy pipeline (không tốn đáng kể).

Kiểm tra theo thứ tự:
1. GET /models: liệt kê model mà gateway phục vụ (nếu endpoint hỗ trợ).
2. POST /embeddings: thử các tên model embedding phổ biến, in cái nào dùng được.
3. POST /chat/completions kèm một tool: kiểm tra model chat có hiểu tool calling.

Cách chạy từ thư mục starter:
    .venv/Scripts/python.exe scripts/check_key.py
"""
import json
import os
import sys

import requests
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

STARTER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(STARTER, ".env"))

EMBED_BASE = (os.environ.get("OPENAI_BASE_URL", "").strip().rstrip("/")
              or "https://api.openai.com/v1")
EMBED_KEY = os.environ.get("OPENAI_API_KEY", "").strip().strip('"')
PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "api").strip() or "api"
CHAT_BASE = (os.environ.get("AGENT_BASE_URL", "").strip().rstrip("/") or EMBED_BASE)
CHAT_KEY = (os.environ.get("AGENT_API_KEY", "").strip().strip('"') or EMBED_KEY)
CHAT_MODEL = os.environ.get("AGENT_MODEL", "gpt-4.1-mini").strip()
CHAT_SESSION = os.environ.get("AGENT_SESSION_ID", "").strip()

EMBED_CANDIDATES = [
    os.environ.get("EMBEDDING_MODEL", "").strip(),
    "text-embedding-3-small",
    "text-embedding-3-large",
    "bge-m3",
    "qwen3-embedding",
    "nomic-embed-text",
]


def main() -> None:
    print(f"Embedding: provider={PROVIDER}" + (f" | endpoint: {EMBED_BASE}" if PROVIDER == "api" else ""))
    print(f"Chat endpoint: {CHAT_BASE} | model: {CHAT_MODEL}")

    if PROVIDER == "local":
        print("Embedding chạy local (Sentence Transformers), bỏ qua kiểm tra API embedding.")
    else:
        for label, base, key in (("embedding", EMBED_BASE, EMBED_KEY), ("chat/agent", CHAT_BASE, CHAT_KEY)):
            r = requests.get(f"{base}/models", headers={"Authorization": f"Bearer {key}"}, timeout=60)
            if r.status_code == 200:
                ids = [m.get("id") for m in r.json().get("data", [])]
                print(f"[{label}] Gateway phục vụ {len(ids)} model, gồm: {', '.join(ids[:25])}")
            else:
                print(f"[{label}] GET /models trả {r.status_code} (nhiều gateway không mở endpoint này, không sao)")
        usable = None
        for cand in EMBED_CANDIDATES:
            if not cand:
                continue
            r = requests.post(f"{EMBED_BASE}/embeddings",
                              headers={"Authorization": f"Bearer {EMBED_KEY}", "Content-Type": "application/json"},
                              json={"model": cand, "input": ["kiểm tra"]}, timeout=120)
            if r.status_code == 200:
                dims = len(r.json()["data"][0]["embedding"])
                print(f"EMBEDDING DÙNG ĐƯỢC: {cand} ({dims} chiều)")
                usable = cand
                break
        if usable is None:
            print("KẾT LUẬN: endpoint embedding không dùng được model nào trong danh sách thử.")

    headers = {"Authorization": f"Bearer {CHAT_KEY}", "Content-Type": "application/json"}
    if CHAT_SESSION and "opencode" in CHAT_BASE:
        headers["x-opencode-session"] = CHAT_SESSION
    r = requests.post(f"{CHAT_BASE}/chat/completions",
                      headers=headers,
                      json={"model": CHAT_MODEL, "temperature": 0.0, "messages": [
                          {"role": "user", "content": "Gọi tool search với query 'Bạch Đằng 938'."}],
                          "tools": [{"type": "function", "function": {
                              "name": "search",
                              "description": "Truy xuất tài liệu.",
                              "parameters": {"type": "object",
                                             "properties": {"query": {"type": "string"}},
                                             "required": ["query"]}}}]}, timeout=180)
    if r.status_code != 200:
        print(f"CHAT LOI: {r.status_code} {r.text[:200]}")
        return
    message = r.json()["choices"][0]["message"]
    calls = message.get("tool_calls") or []
    if calls:
        fn = calls[0].get("function", {})
        print(f"CHAT TOOL CALLING OK: model goi {fn.get('name')} args={fn.get('arguments')}")
    else:
        print(f"CHAT phan hoi nh van ban, khong goi tool: {str(message.get('content'))[:150]}")


if __name__ == "__main__":
    main()
