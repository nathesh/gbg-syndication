"""
Three tests, one per design decision. Stdlib only:

    python3 -m unittest discover -s tests -v

1. Dedupe happens in Korean, before translation, and the tier-3 scan does not
   stop at the first "not the same" verdict (the bug fixed while building).
2. Nothing auto-merges below the 0.92 bar: an ambiguous clinic goes to the
   human queue with clinic_id still None.
3. Idempotency lives at the stage boundary: a redelivered item is a no-op,
   and one poisoned item cannot fail its neighbours.
"""

import os
import sys
import unittest
from datetime import datetime, timezone
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gangnam_syndication_workflow as w  # noqa: E402

NOW = datetime.now(timezone.utc)
BODY = "쌍꺼풀 수술 2024년 3월에 받았어요. 붓기가 오래갔지만 결과는 만족합니다."
CATALOG = [w.Clinic("c_mirae", "강남미래성형외과", "Gangnam Mirae Plastic Surgery",
                    "123-45-67890", "02-555-1234", 37.4979, 127.0276)]


def raw(url: str, body: str = BODY, source: str = "naver_cafe",
        clinic: str = "강남미래성형외과") -> w.RawReview:
    return w.RawReview(source, url, body, NOW, "u", clinic, "쌍꺼풀")


class TestDedupeBeforeTranslate(unittest.TestCase):
    def test_verbatim_repost_never_reaches_translation(self):
        reviews = [w.Review(raw=raw("https://a/1")),
                   w.Review(raw=raw("https://b/2", body=BODY + " ")),   # whitespace noise
                   w.Review(raw=raw("https://c/3", body=BODY + "!!"))]  # punct noise
        with mock.patch.object(w, "translate", wraps=w.translate) as tr:
            out = w.run([r.raw for r in reviews], CATALOG)
            self.assertEqual(len(out), 1)
            self.assertEqual(tr.call_count, 1, "translation ran on a duplicate")

    def test_tier3_keeps_scanning_after_a_not_same_verdict(self):
        # Three candidates: A, then B in the ambiguous band vs A but judged
        # different, then C which is a near-dup of A. Before the fix, the
        # verdict on B ended the scan and C was published as new.
        a = w.Review(raw=raw("https://a/1", body=BODY))
        c = w.Review(raw=raw("https://c/3", body=BODY.replace("만족", "만족만족")))
        b = w.Review(raw=raw("https://b/2", body="완전 다른 후기입니다 " + BODY[:20]))
        dists = {}
        def fake_simhash(text, bits=64):
            # Force distances: A<->B ambiguous (5), A<->C near-dup (2), B<->C far.
            return {a.raw.body_ko: 0, b.raw.body_ko: 0b11111, c.raw.body_ko: 0b11}[text]
        with mock.patch.object(w, "_simhash", fake_simhash), \
             mock.patch.object(w, "llm_json", return_value={"same": False, "why": "x"}):
            kept = w.dedupe([a, b, c])
        self.assertEqual([r.raw.id for r in kept], [a.raw.id, b.raw.id])
        self.assertEqual(c.dup_of, a.raw.id, "near-dup slipped past the scan")


class TestNothingAutoMergesBelowThreshold(unittest.TestCase):
    def test_registration_number_is_a_hard_anchor(self):
        r = w.Review(raw=raw("https://x/1", clinic="어디성형외과 (사업자 123-45-67890)"))
        w.resolve_clinic(r, CATALOG)
        self.assertEqual((r.clinic_id, r.clinic_confidence), ("c_mirae", 1.0))

    def test_ambiguous_match_is_queued_not_merged(self):
        r = w.Review(raw=raw("https://x/2", clinic="미래성형외과"))
        with mock.patch.object(w, "llm_json",
                               return_value={"clinic_id": "c_mirae", "confidence": 0.71}):
            w.resolve_clinic(r, CATALOG)
        self.assertIsNone(r.clinic_id, "merged below the auto bar")
        self.assertEqual(r.clinic_confidence, 0.71)
        self.assertTrue(any("queued" in f for f in r.needs_human))

    def test_model_above_bar_merges_and_model_error_escalates(self):
        r = w.Review(raw=raw("https://x/3", clinic="미래성형외과"))
        with mock.patch.object(w, "llm_json",
                               return_value={"clinic_id": "c_mirae", "confidence": 0.95}):
            w.resolve_clinic(r, CATALOG)
        self.assertEqual(r.clinic_id, "c_mirae")

        r2 = w.Review(raw=raw("https://x/4", clinic="미래성형외과"))
        with mock.patch.object(w, "llm_json", side_effect=ValueError("schema")):
            w.resolve_clinic(r2, CATALOG)
        self.assertIsNone(r2.clinic_id)
        self.assertTrue(r2.needs_human)


class TestIdempotentStageBoundary(unittest.TestCase):
    def test_redelivered_item_is_a_noop(self):
        calls = []
        stage = w.IdempotentStage("t", lambda r: (calls.append(r.raw.id), r)[1])
        r = w.Review(raw=raw("https://x/1"))
        stage([r])
        stage([r])          # redelivery
        stage([r, r])       # duplicate inside one batch
        self.assertEqual(calls, [r.raw.id])

    def test_poisoned_item_does_not_fail_neighbours(self):
        def fn(r):
            if r.raw.source_url.endswith("/poison"):
                raise RuntimeError("boom")
            r.body_en = "ok"
            return r
        stage = w.IdempotentStage("t", fn)
        good1, bad, good2 = (w.Review(raw=raw("https://x/1")),
                             w.Review(raw=raw("https://x/poison")),
                             w.Review(raw=raw("https://x/2")))
        out = stage([good1, bad, good2])
        self.assertEqual(len(out), 3)
        self.assertEqual([r.body_en for r in (good1, good2)], ["ok", "ok"])
        self.assertEqual(bad.needs_human, ["t errored"])
        # The failed item is NOT marked done, so a retry re-attempts it alone.
        stage([bad])
        self.assertEqual(bad.needs_human, ["t errored", "t errored"])


if __name__ == "__main__":
    unittest.main()
