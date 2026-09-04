# Gangnam Beauty Guide — review syndication workflow

Five-stage agentic pipeline for syndicating Korean clinic reviews to an English-language audience.

```
harvest -> dedupe (in Korean) -> resolve clinic -> translate -> score trust
```

Run locally, no dependencies, no API keys:

```bash
python3 gangnam_syndication_workflow.py
```

Tests, one per design decision below (seven assertions, stdlib `unittest`):

```bash
python3 -m unittest discover -s tests -v
```

## Agent steps

Three model calls, each behind the single `llm_json` boundary, each fed by the deterministic stage before it:

| Step | Trigger | Model's job | Hands off to |
|---|---|---|---|
| Dedupe adjudicator | SimHash distance in the 4-8 band | "Same underlying review?" on two Korean texts | Clinic resolver (survivors only) |
| Clinic resolver | No registration number, phone, or exact-name hit | Pick a catalog clinic + confidence | Translator (or human queue if < 0.92) |
| Translator + claims check | Every surviving review | Translate with pinned glossary, then list any dropped complications | Trust scorer (flags if anything dropped) |

With `ANTHROPIC_API_KEY` set the calls go to `claude-opus-5` (JSON-only system prompt, one retry on schema violation, then the stage escalates). Without a key the same control flow runs against fixed stubs, which is how the tests run. The response reports `"mode": "live" | "stub"`.

Deployed: `GET /` runs the pipeline over three fixture reviews and returns what would publish, what went to the human queue, and the per-stage log. Set `ANTHROPIC_API_KEY` in the Vercel project for live mode.

## The three decisions that drive the design

1. **Dedupe before translate, in Korean.** Reposts are byte-identical in the source language. Two MT passes over the same text diverge just enough to slip past near-dup detection. Deduping first also cuts translation spend by the duplicate rate.
2. **Models adjudicate, they don't decide.** Exact hash catches reposts for free, character-trigram SimHash catches light edits, and a model only sees pairs where the cheap signals disagree. Same shape for clinic resolution: registration number, phone and coordinates first, model only when hard anchors are missing.
3. **Nothing auto-merges below threshold.** A clinic match under 0.92 goes to a review queue, not production. A wrong merge attributes one surgeon's outcomes to another.

Idempotency is enforced at the stage boundary (`IdempotentStage`), with per-item isolation so one poisoned review cannot fail its batch.

## Where it broke while building

- Tier-3 dedupe had a `break` after a "not the same" model verdict, so the first ambiguous candidate ended the scan and a genuine near-dup further down the kept list slipped through. Changed to `continue`.
- The clinic-hosted fixture names the clinic `미래성형외과`; the catalog has `강남미래성형외과`. The branch stripper only removes suffixes like `강남점`, so the prefix form fell to the model at 0.71 and queued. Deliberately left as-is: stripping prefixes would merge `강남미래` with a genuinely different `신사미래`.
