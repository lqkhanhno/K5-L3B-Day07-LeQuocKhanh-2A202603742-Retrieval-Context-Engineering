"""Kiểm tra offline cho fusion RRF, filter, evidence matching."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))

from src.retrievers import HybridRetriever, TfidfRetriever, rrf_fuse
from src.dataio import evidence_hit, norm


def chunk(cid, text, page=1, source_id="doc", period=""):
    return {"id": cid, "text": text, "source_id": source_id, "source": f"{source_id}.pdf",
            "page": page, "metadata": {"title": "T", "period": period}}


class RrfFuseTests(unittest.TestCase):
    def test_higher_fused_rank_first(self):
        fused = rrf_fuse([["a", "b", "c"], ["b", "d"]], top_k=3)
        self.assertEqual([cid for cid, _ in fused], ["b", "a", "d"])
        self.assertTrue(all(score > 0 for _, score in fused))

    def test_top_k_limit(self):
        fused = rrf_fuse([["a", "b"], ["c", "d"], ["e"]], top_k=2)
        self.assertEqual(len(fused), 2)


class TfidfRetrieverTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [chunk("a", "Hoàn tiền trong bảy ngày.", 1, period="co_dai"),
                       chunk("b", "Giao hàng ba ngày.", 2, period="1858_ve_sau")]

    def test_rank_and_no_filter(self):
        r = TfidfRetriever(self.chunks)
        items = r.search("hoàn tiền", filters=None, top_k=2)
        self.assertEqual(items[0]["id"], "a")
        self.assertGreater(items[0]["score"], 0)

    def test_filter_restricts_pool(self):
        r = TfidfRetriever(self.chunks)
        items = r.search("giao hàng bao nhiêu ngày",
                         filters={"field": "period", "value": "1858_ve_sau"}, top_k=2)
        self.assertEqual([i["id"] for i in items], ["b"])

    def test_unknown_query_empty(self):
        r = TfidfRetriever(self.chunks)
        self.assertEqual(r.search("zzzzqqqq", filters=None, top_k=2), [])


class EvidenceHitTests(unittest.TestCase):
    def test_normalization_and_source_page_match(self):
        item = chunk("a", "Xử lý trong 7 ngày làm  việc.", 3)
        gold = [{"source_id": "doc", "page": 3, "evidence": "trong 7 ngày"}]
        self.assertTrue(evidence_hit(item, gold))
        wrong_page = [{"source_id": "doc", "page": 4, "evidence": "trong 7 ngày"}]
        self.assertFalse(evidence_hit(item, wrong_page))
        self.assertFalse(evidence_hit(chunk("a", "không liên quan", 3), gold))


class HybridRetrieverTests(unittest.TestCase):
    def test_fusion_uses_both_legs_and_keeps_metadata(self):
        class FakeDense:
            def search(self, query, *, filters, top_k):
                self.filters = filters
                if filters and filters.get("value") == "co_dai":
                    return [chunk("a", "dense text", period="co_dai")]
                return [chunk("a", "dense text"), chunk("b", "dense text 2")]

        lexical = TfidfRetriever([chunk("b", "Hoàn tiền bảy ngày.", period="co_dai"),
                                  chunk("c", "Giao hàng ba ngày.", 2)])
        hybrid = HybridRetriever(FakeDense(), lexical)
        items = hybrid.search("hoàn tiền", filters={"field": "period", "value": "co_dai"}, top_k=3)
        ids = [i["id"] for i in items]
        self.assertIn("b", ids)  # xuất hiện ở cả hai nhánh phải được ghép đúng
        meta = next(i for i in items if i["id"] == "b")["metadata"]
        self.assertIn("dense_rank", meta)
        self.assertTrue(all(i["metadata"]["period"] == "co_dai" for i in items))


if __name__ == "__main__":
    unittest.main()
