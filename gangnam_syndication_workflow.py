"""
Gangnam Beauty Guide — review syndication pipeline.

Five stages: harvest -> dedupe(KO) -> resolve clinic -> translate -> score trust.

Three decisions drive the whole design; everything else falls out of them.

1. DEDUPE BEFORE TRANSLATE, IN KOREAN.
   The same review gets reposted across Naver cafes, Daum, and clinic blogs.
   If you translate first, two MT passes over the same Korean text diverge just
   enough that near-dup detection misses them, and you ship the same review
   three times under three clinics. Dedupe on the source string, which is
   byte-identical across reposts. This also cuts translation spend by the
   duplicate rate (~30-40% on syndicated Korean beauty content).

2. LLMs ADJUDICATE, THEY DON'T DECIDE.
   Every stage is deterministic-first with a narrow ambiguity band handed to a
   model. Exact hash catches reposts for free; SimHash catches edits; the model
   only sees pairs in the band where cheap signals disagree. Same for clinic
   resolution: match on registration number, phone, and coordinates first,
   because those are stable, and only ask a model when the hard anchors are
   missing. This keeps cost sublinear in volume and keeps the system explainable
   when a clinic disputes a merge.

3. NOTHING AUTO-MERGES BELOW THRESHOLD.
   This is health content with money attached. An unsure clinic match goes to a
   review queue, not to production. A wrong merge attributes one surgeon's
   outcomes to another, which is the single worst thing this product can do.

Idempotency is enforced at the stage boundary, not inside the workers -- see
IdempotentStage. That is a scar from a previous system, documented in the
autopsy: a non-idempotent trigger inside a retrying consumer will re-fire
forever, and the retry storm is invisible until the provider complains.

Run: python3 gangnam_syndication_workflow.py  (stdlib only, no keys needed)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Callable, Iterable, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("syndication")

# Confidence below which we refuse to act and escalate to a human queue.
CLINIC_MATCH_AUTO = 0.92
CLINIC_MATCH_FLOOR = 0.60
SIMHASH_NEAR_DUP = 3       # hamming distance <= this  => duplicate, no LLM
SIMHASH_AMBIGUOUS = 8      # between NEAR_DUP and this => ask the model


# --------------------------------------------------------------------------
# Domain
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RawReview:
    source: str                  # "naver_cafe" | "daum" | "clinic_blog"
    source_url: str
    body_ko: str
    posted_at: datetime
    author_handle: str
    clinic_text: str             # clinic name as written, unnormalised
    procedure_text: str
    # Korean sponsored-post disclosure tags, when present.
    disclosure_tags: tuple[str, ...] = ()

    @property
    def id(self) -> str:
        return hashlib.sha256(self.source_url.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Clinic:
    clinic_id: str
    name_ko: str
    name_en: str
    biz_reg_no: Optional[str]    # 사업자등록번호 -- the only truly stable key
    phone: Optional[str]
    lat: Optional[float]
    lon: Optional[float]


@dataclass
class Review:
    raw: RawReview
    dup_of: Optional[str] = None
    clinic_id: Optional[str] = None
    clinic_confidence: float = 0.0
    body_en: Optional[str] = None
    trust: dict = field(default_factory=dict)
    needs_human: list[str] = field(default_factory=list)

    def flag(self, why: str) -> None:
        self.needs_human.append(why)
        log.warning("[%s] escalated: %s", self.raw.id, why)


# --------------------------------------------------------------------------
# Model boundary -- one call site, schema-checked, never trusted raw
# --------------------------------------------------------------------------

def llm_json(prompt: str, schema: dict, *, _stub: Optional[dict] = None) -> dict:
    """
    The single place a model is called. Returns parsed JSON validated against
    `schema`, or raises. Callers must handle the raise -- a model failure is a
    normal event, not an exception path bolted on later.

    Deliberately the only LLM entry point so that cost, latency, retries and
    prompt versioning have exactly one place to live. `_stub` lets the pipeline
    run end to end offline for tests and for this demo.
    """
    if _stub is not None:
        payload = _stub
    else:  # pragma: no cover -- real call site
        raise NotImplementedError(
            "Wire to your provider here. Enforce: temperature=0, JSON mode, "
            "max 1 retry on schema violation, then fall through to the "
            "deterministic default. Never retry unbounded."
        )
    missing = [k for k in schema if k not in payload]
    if missing:
        raise ValueError(f"model omitted required keys {missing}")
    return payload


# --------------------------------------------------------------------------
# Stage 1 -- harvest
# --------------------------------------------------------------------------

def harvest(adapters: Iterable[Callable[[], Iterable[RawReview]]]) -> list[RawReview]:
    """
    Per-source isolation: one adapter throwing must not lose the other sources'
    output. Naver rate-limits aggressively and Daum's markup changes without
    notice, so partial harvests are the steady state, not the exception.
    """
    out: list[RawReview] = []
    for adapter in adapters:
        try:
            got = list(adapter())
            log.info("harvest %-14s %d reviews", adapter.__name__, len(got))
            out.extend(got)
        except Exception:
            log.exception("harvest %s failed; continuing", adapter.__name__)
    return out


# --------------------------------------------------------------------------
# Stage 2 -- dedupe, in Korean, before any translation
# --------------------------------------------------------------------------

_PUNCT = re.compile(r"[\s​!-/:-@\[-`{-~·…~]+")


def _canon_ko(text: str) -> str:
    """Strip whitespace/punct/emoji noise that reposting tools inject."""
    return _PUNCT.sub("", text)


def _simhash(text: str, bits: int = 64) -> int:
    """
    Character-trigram SimHash. Trigrams, not words: Korean is agglutinative and
    written without reliable spacing, so word tokenisation is itself a model
    call. Trigrams need no tokeniser and survive particle changes.
    """
    canon = _canon_ko(text)
    vec = [0] * bits
    for i in range(max(len(canon) - 2, 1)):
        gram = canon[i:i + 3]
        h = int(hashlib.md5(gram.encode()).hexdigest(), 16)
        for b in range(bits):
            vec[b] += 1 if (h >> b) & 1 else -1
    out = 0
    for b in range(bits):
        if vec[b] > 0:
            out |= 1 << b
    return out


def dedupe(reviews: list[Review]) -> list[Review]:
    seen_exact: dict[str, str] = {}
    kept: list[tuple[str, int]] = []

    for r in reviews:
        canon = _canon_ko(r.raw.body_ko)

        # Tier 1 -- exact repost. Free, catches the bulk of syndication.
        key = hashlib.sha256(canon.encode()).hexdigest()
        if key in seen_exact:
            r.dup_of = seen_exact[key]
            continue
        seen_exact[key] = r.raw.id

        # Tier 2 -- near dup after light editing.
        sig = _simhash(r.raw.body_ko)
        for other_id, other_sig in kept:
            dist = bin(sig ^ other_sig).count("1")
            if dist <= SIMHASH_NEAR_DUP:
                r.dup_of = other_id
                break
            if dist <= SIMHASH_AMBIGUOUS:
                # Tier 3 -- only here do we spend a model call.
                try:
                    verdict = llm_json(
                        f"Same underlying review? A/B Korean text...",
                        {"same": bool, "why": str},
                        _stub={"same": False, "why": "different procedure date"},
                    )
                except Exception:
                    log.exception("dedupe adjudication failed; keeping both")
                    break
                if verdict["same"]:
                    r.dup_of = other_id
                    break
                # Not the same: keep scanning. An earlier version broke here
                # and stopped comparing after the first ambiguous candidate,
                # so a true near-dup further down the kept list slipped through.
                continue
        if r.dup_of is None:
            kept.append((r.raw.id, sig))

    dupes = sum(1 for r in reviews if r.dup_of)
    log.info("dedupe: %d/%d duplicates removed pre-translation", dupes, len(reviews))
    return [r for r in reviews if r.dup_of is None]


# --------------------------------------------------------------------------
# Stage 3 -- clinic resolution (entity resolution, not string matching)
# --------------------------------------------------------------------------

_BRANCH = re.compile(r"(강남|압구정|신사|청담)?점$")


def _strip_branch(name: str) -> str:
    return _BRANCH.sub("", _canon_ko(name))


def resolve_clinic(r: Review, catalog: list[Clinic]) -> Review:
    """
    Names are the worst available key: clinics rebrand, open branches, and
    romanise inconsistently (Mirae / Mirae / Miraeh). Anchor on identifiers that
    a clinic cannot casually change -- business registration number, phone,
    coordinates -- and treat the name as a tie-breaker only.
    """
    text = r.raw.clinic_text

    for c in catalog:                                   # hard anchor
        if c.biz_reg_no and c.biz_reg_no in text:
            r.clinic_id, r.clinic_confidence = c.clinic_id, 1.0
            return r

    for c in catalog:                                   # exact name post-branch-strip
        if _strip_branch(c.name_ko) == _strip_branch(text):
            r.clinic_id, r.clinic_confidence = c.clinic_id, 0.95
            return r

    try:                                                # ambiguity band only
        verdict = llm_json(
            f"Which catalog clinic is '{text}'? Return clinic_id and confidence.",
            {"clinic_id": str, "confidence": float},
            _stub={"clinic_id": catalog[0].clinic_id, "confidence": 0.71},
        )
    except Exception:
        r.flag("clinic resolution errored")
        return r

    conf = float(verdict["confidence"])
    if conf >= CLINIC_MATCH_AUTO:
        r.clinic_id, r.clinic_confidence = verdict["clinic_id"], conf
    elif conf >= CLINIC_MATCH_FLOOR:
        r.clinic_confidence = conf
        r.flag(f"clinic match {conf:.2f} below auto-merge bar -- queued")
    else:
        r.flag("no plausible clinic match -- new clinic candidate")
    return r


# --------------------------------------------------------------------------
# Stage 4 -- translate (after dedupe, only for surviving reviews)
# --------------------------------------------------------------------------

# Pinned so the same procedure never surfaces under two English names; users
# filter on these, and "double eyelid" vs "blepharoplasty" splits the index.
GLOSSARY = {
    "쌍꺼풀": "double eyelid surgery",
    "코성형": "rhinoplasty",
    "윤곽": "facial contouring",
    "지방이식": "fat grafting",
    "리프팅": "lifting",
}


def translate(r: Review) -> Review:
    try:
        out = llm_json(
            "Translate KO->EN. Preserve hedging and negative outcomes exactly. "
            f"Use this glossary verbatim: {json.dumps(GLOSSARY, ensure_ascii=False)}",
            {"body_en": str, "dropped_claims": list},
            _stub={"body_en": "[EN] " + r.raw.body_ko[:60], "dropped_claims": []},
        )
    except Exception:
        r.flag("translation failed")
        return r

    # A translation that silently drops a complication is worse than no
    # translation -- this product exists to surface bad outcomes too.
    if out["dropped_claims"]:
        r.flag(f"translation dropped claims: {out['dropped_claims']}")
    r.body_en = out["body_en"]
    return r


# --------------------------------------------------------------------------
# Stage 5 -- trust
# --------------------------------------------------------------------------

# Korean law requires disclosure of paid reviews; the tags are well known, and
# the interesting signal is the review that reads sponsored but carries no tag.
DISCLOSURE = ("협찬", "체험단", "광고", "소정의")


def score_trust(r: Review) -> Review:
    body = r.raw.body_ko
    declared = bool(r.raw.disclosure_tags) or any(t in body for t in DISCLOSURE)

    signals = {
        "declared_sponsored": declared,
        "clinic_owned_domain": r.raw.source == "clinic_blog",
        "has_procedure_date": bool(re.search(r"20\d{2}[.\-/년]", body)),
        "mentions_complication": any(k in body for k in ("부작용", "재수술", "붓기")),
        "clinic_verified": r.clinic_confidence >= CLINIC_MATCH_AUTO,
    }

    # Undisclosed-sponsorship detection is the actual moat. A verified badge is
    # table stakes; RealSelf already has one. Knowing which reviews are bought
    # and unlabelled is what a Western buyer cannot get anywhere else.
    if not declared and signals["clinic_owned_domain"]:
        signals["suspected_undisclosed_sponsorship"] = True
        r.flag("clinic-hosted, no disclosure tag -- sponsorship review")

    score = 0.5
    score += 0.2 if signals["has_procedure_date"] else 0.0
    score += 0.2 if signals["mentions_complication"] else 0.0   # complaints are credible
    score += 0.1 if signals["clinic_verified"] else 0.0
    score -= 0.4 if declared else 0.0

    r.trust = {"score": round(max(score, 0.0), 2), "signals": signals}
    return r


# --------------------------------------------------------------------------
# Orchestration -- idempotent at the stage boundary
# --------------------------------------------------------------------------

class IdempotentStage:
    """
    Wraps a stage so a redelivered item is a no-op instead of a re-execution,
    and so one poisoned item cannot fail its whole batch.

    Both halves are load-bearing. In a prior system I owned, a stream consumer
    wrote its own progress marker back to the table it consumed, so every write
    re-triggered the consumer; the side effect it performed was a non-idempotent
    job trigger, and the batch-level try/except meant one bad document failed
    all 25 of its neighbours and got redelivered forever. The provider noticed
    before our dashboards did. Idempotency key first, per-item isolation second.
    """

    def __init__(self, name: str, fn: Callable[[Review], Review]):
        self.name, self.fn = name, fn
        self._done: set[str] = set()   # a durable store in production

    def __call__(self, reviews: list[Review]) -> list[Review]:
        out, skipped, failed = [], 0, 0
        for r in reviews:
            key = f"{self.name}:{r.raw.id}"
            if key in self._done:
                skipped += 1
                out.append(r)
                continue
            try:
                r = self.fn(r)
                self._done.add(key)
            except Exception:
                log.exception("%s failed on %s; isolated", self.name, r.raw.id)
                r.flag(f"{self.name} errored")
                failed += 1
            out.append(r)
        log.info("%-16s ok=%d skipped=%d failed=%d",
                 self.name, len(out) - skipped - failed, skipped, failed)
        return out


def run(raws: list[RawReview], catalog: list[Clinic]) -> list[Review]:
    reviews = [Review(raw=x) for x in raws]
    reviews = dedupe(reviews)                       # batch-level, must see all
    for stage in (
        IdempotentStage("resolve_clinic", lambda r: resolve_clinic(r, catalog)),
        IdempotentStage("translate", translate),
        IdempotentStage("score_trust", score_trust),
    ):
        reviews = stage(reviews)
    return reviews


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------

def demo_fixtures() -> tuple[list[RawReview], list[Clinic]]:
    now = datetime.now(timezone.utc)
    body = "쌍꺼풀 수술 2024년 3월에 받았어요. 붓기가 오래갔지만 결과는 만족합니다."

    fixtures = [
        RawReview("naver_cafe", "https://cafe.naver.com/a/1", body, now, "u1",
                  "강남미래성형외과", "쌍꺼풀"),
        # Verbatim repost on another source -- caught by tier 1, free.
        RawReview("daum", "https://cafe.daum.net/b/2", body, now, "u2",
                  "강남미래성형외과 강남점", "쌍꺼풀"),
        # Clinic-hosted, no disclosure tag -- the interesting case.
        RawReview("clinic_blog", "https://mirae.co.kr/review/9",
                  "코성형 결과 너무 예뻐요! 원장님 최고!", now, "u3",
                  "미래성형외과", "코성형"),
    ]
    catalog = [Clinic("c_mirae", "강남미래성형외과", "Gangnam Mirae Plastic Surgery",
                      "123-45-67890", "02-555-1234", 37.4979, 127.0276)]
    return fixtures, catalog


def to_dict(r: Review) -> dict:
    return {
        "id": r.raw.id,
        "source": r.raw.source,
        "clinic": r.clinic_id,
        "confidence": r.clinic_confidence,
        "en": r.body_en,
        "trust": r.trust.get("score"),
        "signals": r.trust.get("signals", {}),
        "flags": r.needs_human,
    }


def run_demo() -> dict:
    """Run the fixture pipeline and return results plus the stage log."""
    lines: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            lines.append(self.format(record))

    h = _Capture()
    h.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    log.addHandler(h)
    try:
        raws, catalog = demo_fixtures()
        results = run(raws, catalog)
    finally:
        log.removeHandler(h)
    return {
        "input_reviews": len(raws),
        "published": [to_dict(r) for r in results if not r.needs_human],
        "human_queue": [to_dict(r) for r in results if r.needs_human],
        "log": lines,
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), ensure_ascii=False, indent=2))
