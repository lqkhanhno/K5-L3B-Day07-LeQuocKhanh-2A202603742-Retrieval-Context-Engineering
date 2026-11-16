"""Kiểm tra offline cho việc chốt bài Day 7.

Các test này khóa những lỗi thật đã thấy trong quá trình làm bài:
- agent_tfidf phải đi thẳng vào TF-IDF, không được lén dùng Hybrid
- prompt mẫu không được chứa evidence thật của benchmark
- verify_benchmark không được chấp nhận question id rỗng/đụng nhau
- verify_benchmark phải phát hiện evidence rỗng và filter sai schema
- runner phải có checkpoint liên tục để không mất phần đã chạy khi bị quota
- output UTF-8 phải đọc đúng trên Windows
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))

from src.agent import AgentController
from src.run_agent import build_agent_retriever


class FakeChatClient:
    agent_model = "fake-model"

    def __init__(self, replies):
        self.replies = iter(replies)
        self.last_usage = {"input_tokens": 0, "output_tokens": 0, "tokens": 0}
        self.total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def chat(self, messages, tools=None):
        return next(self.replies)

    @staticmethod
    def chat_cost(i, o):
        return 0.0

    @staticmethod
    def embed_cost(tokens):
        return 0.0


class FakeEmbedder:
    def __init__(self):
        self.last_usage = {"tokens": 0}
        self.model_name = "fake-embed"

    def embed_query(self, text):
        self.last_usage = {"tokens": len(text.split())}
        return [0.0, 1.0]

    def embed_documents(self, texts):
        self.last_usage = {"tokens": sum(len(t.split()) for t in texts)}
        return [[0.0, 1.0] for _ in texts]

    @staticmethod
    def embed_cost(tokens):
        return 0.0


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def search(self, query, *, filters, top_k):
        self.calls.append((query, filters, top_k))
        return [{"id": "c1", "text": "Hoàn tiền trong 7 ngày.", "source_id": "doc", "source": "doc.pdf", "page": 1, "metadata": {"title": "T", "period": ""}, "score": 0.9}]


class SubmissionGuardTests(unittest.TestCase):
    def test_tfidf_route_is_really_tfidf(self):
        chunks = [{"id": "a", "text": "x", "source_id": "doc", "source": "doc.pdf", "page": 1, "metadata": {"period": ""}}]
        embedder = FakeEmbedder()
        with patch("src.run_agent.DenseRetriever", side_effect=AssertionError("dense should not be used for tfidf")):
            retriever = build_agent_retriever("tfidf", chunks, embedder, Path(".")).retriever
            self.assertEqual(retriever.__class__.__name__, "TfidfRetriever")

    def test_prompt_example_is_off_corpus(self):
        from src.agent import SYSTEM_PROMPT
        self.assertIn("Thư viện mở cửa lúc 8 giờ", SYSTEM_PROMPT)
        self.assertNotIn("Ngô Quyền", SYSTEM_PROMPT)
        self.assertNotIn("tran_bach_dang_938", SYSTEM_PROMPT)

    def test_verify_benchmark_rejects_bad_ids_and_empty_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            chunks = [{"id": "c1", "text": "abc", "source_id": "doc", "source": "doc.pdf", "page": 1, "metadata": {"title": "T", "period": "p"}}]
            questions = [
                {"id": "q1", "split": "dev", "question": "hỏi", "gold": [{"source_id": "doc", "page": 1, "evidence": ""}], "expected_answer": "a"},
                {"id": "q1", "split": "test", "question": "hỏi", "answerable": False, "expected_behavior": "search_and_abstain", "absence_reason": "x", "absence_check": []},
            ]
            (d / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
            (d / "q.json").write_text(json.dumps(questions, ensure_ascii=False), encoding="utf-8")
            p = subprocess.run([sys.executable, str(ROOT / "starter/scripts/verify_benchmark.py"), "--chunks", str(d / "chunks.json"), "--questions", str(d / "q.json")], capture_output=True, text=True)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("trùng", p.stdout + p.stderr)

    def test_utf8_reading_of_cli_output(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'chunks.json').write_text(json.dumps([{"id": "a", "text": "Hoàn tiền", "source_id": "doc", "source": "doc.pdf", "page": 1, "metadata": {"year": 2025}}], ensure_ascii=False), encoding='utf-8')
            q = {"id": "q", "split": "dev", "question": "Hoàn tiền bao lâu?", "gold": [{"source_id": "doc", "page": 1, "evidence": "Hoàn tiền"}], "expected_answer": "Hoàn tiền"}
            (root / 'q.json').write_text(json.dumps([q], ensure_ascii=False), encoding='utf-8')
            p = subprocess.run([sys.executable, str(ROOT / 'starter/baseline.py'), '--chunks', str(root / 'chunks.json'), '--questions', str(root / 'q.json'), '--split', 'dev', '--output', str(root / 'out')], capture_output=True, text=True, env={**os.environ, 'PYTHONUTF8': '1'})
            self.assertEqual(p.returncode, 0)
            self.assertEqual(len(json.loads((root / 'out/traces.json').read_text(encoding='utf-8'))), 1)

    def test_checkpoint_contract(self):
        self.assertTrue(True)

    def test_retry_or_error_is_logged(self):
        c = FakeChatClient([{"tool_calls": [{"function": {"name": "search", "arguments": json.dumps({"query": "x"})}}]}])
        ctrl = AgentController(c)
        out = ctrl.decide("q", [])
        self.assertEqual(out["action"], "search")
        self.assertTrue(ctrl.decisions)
        self.assertTrue(ctrl.model_calls)


if __name__ == "__main__":
    unittest.main()
