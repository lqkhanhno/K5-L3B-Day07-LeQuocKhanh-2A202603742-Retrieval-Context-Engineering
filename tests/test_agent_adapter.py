"""Kiểm tra offline cho adapter agent (parser tool call và tích hợp khung starter run_agent).

FakeChatClient chỉ giả lập phần hồi tool call; đo chất lượng model thật phải
chạy agent thật và chấm thủ công, không dùng fake để báo benchmark.
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))

from agentic import run_agent
from src.agent import AgentController, AgentRetriever

CHUNK = {"id": "c1", "text": "Hoàn tiền trong 7 ngày.", "source_id": "doc",
         "source": "doc.pdf", "page": 1, "metadata": {"title": "T", "period": ""}, "score": 0.9}


class FakeChatClient:
    agent_model = "fake-model"

    def __init__(self, replies):
        self.replies = iter(replies)
        self.last_usage = {"tokens": 0}

    def chat(self, messages, tools=None):
        message = next(self.replies)
        self.last_usage = {"input_tokens": 10, "output_tokens": 5}
        return message

    @staticmethod
    def chat_cost(i, o):
        return 0.0

    @staticmethod
    def embed_cost(tokens):
        return 0.0


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def search(self, query, *, filters, top_k):
        self.calls.append((query, filters, top_k))
        return [CHUNK]


def tool_call(name, args):
    return {"tool_calls": [{"function": {"name": name, "arguments": json.dumps(args)}}]}


class AgentAdapterTests(unittest.TestCase):
    def test_search_then_cited_answer(self):
        client = FakeChatClient([tool_call("search", {"query": "hoàn tiền"}),
                                 tool_call("submit_answer", {"answer": "7 ngày [c1]", "citations": ["c1"]})])
        retriever = FakeRetriever()
        result = run_agent("Hoàn tiền mất bao lâu?", AgentController(client),
                           AgentRetriever(retriever, client), filters={"field": "period", "value": "co_dai"})
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["citations"], ["c1"])
        self.assertEqual(retriever.calls[0], ("hoàn tiền", {"field": "period", "value": "co_dai"}, 3))

    def test_controller_usage_accumulates(self):
        client = FakeChatClient([tool_call("search", {"query": "q1"}),
                                 tool_call("submit_answer", {"answer": "a [c1]", "citations": ["c1"]})])
        controller = AgentController(client)
        run_agent("q", controller, AgentRetriever(FakeRetriever(), client))
        self.assertEqual(controller.usage, {"input_tokens": 20, "output_tokens": 10})

    def test_no_tool_call_raises_error(self):
        client = FakeChatClient([{"content": "Tôi sẽ trả lời thẳng.", "tool_calls": None}])
        with self.assertRaises(RuntimeError):
            run_agent("q", AgentController(client), AgentRetriever(FakeRetriever(), client))

    def test_answer_without_citation_abstains(self):
        client = FakeChatClient([tool_call("search", {"query": "q1"}),
                                 tool_call("submit_answer", {"answer": "7 ngày.", "citations": []})])
        result = run_agent("q", AgentController(client), AgentRetriever(FakeRetriever(), client))
        self.assertEqual(result["status"], "abstain")


if __name__ == "__main__":
    unittest.main()
