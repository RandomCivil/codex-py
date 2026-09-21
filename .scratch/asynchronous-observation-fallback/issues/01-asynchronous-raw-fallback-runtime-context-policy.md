# 01: Asynchronous Raw fallback in Runtime context policy

**What to build:** A settled Tool-call batch becomes immediately usable transient Raw evidence while its single no-tool Observation request runs independently. Every Runtime context snapshot keeps the newest three Tool rounds as Raw and represents each older round with its validated Observation when available, otherwise its complete Raw tool result; required Raw evidence remains lossless when applying the context budget.

**Blocked by:** None (can start immediately).

**Status:** resolved

- [x] Recording a settled Tool-call batch returns without waiting for its Observation request, and immediately makes its ordered calls, arguments, results, and errors available as a Raw tool result.
- [x] A Runtime context snapshot retains exactly the newest three Tool rounds as Raw and substitutes a validated Observation only for eligible older rounds.
- [x] A pending, failed, or invalid Observation produces complete Raw fallback for its eligible older round; a later successful Observation appears in a subsequent snapshot with its required provenance.
- [x] Required Raw fallback evidence is neither truncated nor re-summarized; budget maintenance fails explicitly when Durable State and required Raw evidence cannot fit.
- [x] Existing synchronous, provenance-preserving merging applies only to validated Observations and retains its strict failure behavior.

## Answer

Implemented the asynchronous Raw fallback policy in `RuntimeContextPolicy`. Settled Tool rounds are recorded immediately, Observation requests run as event-loop-local best-effort tasks, and context snapshots retain complete Raw fallback until a validated older Observation is available. Added controlled delayed/invalid Observation coverage in `tests/test_runtime_context.py`.

Verification: `poetry run pytest -q` — 253 passed, 7 skipped.
