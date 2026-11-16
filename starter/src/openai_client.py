"""Client API tương thích OpenAI, có ghi usage và giới hạn thời gian thử lại."""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import dotenv_values

DEFAULT_BASE = "https://api.openai.com/v1"
# Chi phí dịch vụ của cấu hình nộp bài: embedding local và gói Groq miễn phí.
PRICE_EMBED_PER_M = 0.0
PRICE_CHAT_IN_PER_M = 0.0
PRICE_CHAT_OUT_PER_M = 0.0


class APIError(RuntimeError):
    def __init__(self, message: str, *, quota_exhausted: bool = False):
        super().__init__(message)
        self.quota_exhausted = quota_exhausted


def load_config() -> dict[str, str]:
    values = {**os.environ, **{k: v for k, v in dotenv_values(
        Path(__file__).resolve().parents[1] / ".env", encoding="utf-8-sig"
    ).items() if v is not None}}
    def get(name, default=""):
        return str(values.get(name, default)).strip()
    provider = get("EMBEDDING_PROVIDER", "local")
    if provider not in ("local", "api"):
        raise ValueError("EMBEDDING_PROVIDER phải là local hoặc api")
    embed_key = get("OPENAI_API_KEY")
    return {
        "embedding_provider": provider,
        "embed_key": embed_key,
        "embed_base": get("OPENAI_BASE_URL") or DEFAULT_BASE,
        "embedding_model": get("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"),
        "chat_key": get("AGENT_API_KEY") or embed_key,
        "chat_base": get("AGENT_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/"),
        "agent_model": get("AGENT_MODEL", "openai/gpt-oss-120b"),
        "agent_session": get("AGENT_SESSION_ID"),
        "agent_extra_headers": get("AGENT_EXTRA_HEADERS"),
    }


def make_embedder(config: dict[str, str] | None = None):
    config = config or load_config()
    if config["embedding_provider"] == "local":
        from .local_embedder import LocalEmbedder
        return LocalEmbedder(config["embedding_model"])
    return OpenAIClient(config)


class OpenAIClient:
    def __init__(self, config: dict[str, str]) -> None:
        self.embedding_model = config["embedding_model"]
        self.agent_model = config["agent_model"]
        self.session = requests.Session()
        self.embed_base = config["embed_base"].rstrip("/")
        self.chat_base = config["chat_base"].rstrip("/")
        self._secrets = [v for v in (config["embed_key"], config["chat_key"]) if v]
        self.embed_headers = {"Authorization": f"Bearer {config['embed_key']}", "Content-Type": "application/json"}
        self.chat_headers = {"Authorization": f"Bearer {config['chat_key']}", "Content-Type": "application/json"}
        if config.get("agent_session") and "opencode.ai" in self.chat_base:
            self.chat_headers["x-opencode-session"] = config["agent_session"]
        for line in config.get("agent_extra_headers", "").splitlines():
            if ":" in line:
                name, value = line.split(":", 1)
                if name.strip().lower() not in ("authorization", "host"):
                    self.chat_headers[name.strip()] = value.strip()
        self.last_usage = {"tokens": 0}
        self.total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        self.request_log: list[dict] = []
        self._next_chat_at = 0.0

    def _safe_error(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, "[REDACTED]")
        return re.sub(r"(?:gsk_|sk-|oc_sk_)[A-Za-z0-9_-]+", "[REDACTED]", text)[:900]

    @staticmethod
    def _parse_retry_after(text: str) -> float | None:
        match = re.search(r"(?:try again|retry) in\s*(?:(\d+)h)?\s*(?:(\d+)m)?\s*([\d.]+)\s*s", text, re.I)
        if not match:
            return None
        hours, minutes, seconds = match.groups()
        return int(hours or 0) * 3600 + int(minutes or 0) * 60 + float(seconds)

    def _post(self, base: str, headers: dict[str, str], path: str,
              payload: dict[str, Any], *, max_retries: int = 3) -> dict[str, Any]:
        started = time.monotonic()
        for attempt in range(max_retries):
            try:
                response = self.session.post(base + path, json=payload, headers=headers, timeout=(10, 45))
            except requests.RequestException as exc:
                self.request_log.append({"path": path, "attempt": attempt + 1, "error": type(exc).__name__})
                raise APIError(f"Không nhận được phản hồi API: {type(exc).__name__}") from exc
            record = {"path": path, "attempt": attempt + 1, "status": response.status_code}
            self.request_log.append(record)
            if response.status_code == 200:
                return response.json()
            text = response.text
            daily = response.status_code == 429 and any(x in text.lower() for x in ("tokens per day", "requests per day", "perday", "per day limit"))
            retry = self._parse_retry_after(text)
            try:
                retry = float(response.headers.get("Retry-After", retry)) if response.headers.get("Retry-After") else retry
            except (TypeError, ValueError):
                pass
            record["retry_after_seconds"] = retry
            message = self._safe_error(text)
            if daily:
                raise APIError(f"Hết quota ngày: {message}", quota_exhausted=True)
            wait = (retry + 1) if retry is not None else (5 * (attempt + 1))
            if response.status_code not in (429, 500, 502, 503, 529) or attempt == max_retries - 1 or wait > 60 or time.monotonic() - started + wait > 120:
                raise APIError(f"API trả HTTP {response.status_code}: {message}")
            print(f"  API {response.status_code}; thử lại sau {wait:.1f} giây.", flush=True)
            time.sleep(wait)
        raise APIError("Không thể gọi API")

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        data = self._post(self.embed_base, self.embed_headers, "/embeddings", {"model": self.embedding_model, "input": texts})
        self.last_usage = {"tokens": data.get("usage", {}).get("total_tokens", 0)}
        vectors = [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
        if len(vectors) != len(texts):
            raise RuntimeError("Số vector trả về không khớp số văn bản")
        return vectors

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        key = self.chat_headers["Authorization"].removeprefix("Bearer ")
        if not key or not key.isascii():
            raise APIError("Chưa có AGENT_API_KEY hợp lệ trong starter/.env")
        delay = self._next_chat_at - time.monotonic()
        if delay > 0:
            time.sleep(min(delay, 60))
        started = time.monotonic()
        payload: dict[str, Any] = {"model": self.agent_model, "messages": messages,
                                   "temperature": 0.0, "max_completion_tokens": 768}
        if tools:
            payload["tools"] = tools
        data = self._post(self.chat_base, self.chat_headers, "/chat/completions", payload)
        usage = data.get("usage", {})
        self.last_usage = {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)}
        for field in ("input_tokens", "output_tokens"):
            self.total_usage[field] += self.last_usage[field]
        self.total_usage["total_tokens"] += usage.get("total_tokens", sum(self.last_usage.values()))
        # Giữ tốc độ dưới 8.000 token/phút của tài khoản Groq đã kiểm tra.
        self._next_chat_at = started + max(2, usage.get("total_tokens", sum(self.last_usage.values())) * 60 / 6500)
        choice = data["choices"][0]
        if choice.get("finish_reason") == "length":
            raise APIError("Model hết ngân sách output trước khi hoàn tất tool call")
        return choice["message"]

    @staticmethod
    def embed_cost(tokens: int) -> float:
        return tokens / 1e6 * PRICE_EMBED_PER_M

    @staticmethod
    def chat_cost(input_tokens: int, output_tokens: int) -> float:
        return input_tokens / 1e6 * PRICE_CHAT_IN_PER_M + output_tokens / 1e6 * PRICE_CHAT_OUT_PER_M
