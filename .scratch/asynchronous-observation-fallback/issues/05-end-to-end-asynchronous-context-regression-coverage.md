# 05: End-to-end asynchronous context regression coverage

**What to build:** The asynchronous Observation fallback contract is regression-protected across both tool-capable execution modes, including success, delay, failure, validation, terminal lifecycle, budget, provenance, and transient-state behavior.

**Blocked by:** 03: ReAct asynchronous Runtime context integration; 04: Plan-step Executor asynchronous Runtime context integration.

**Status:** resolved

- [x] Controlled asynchronous model doubles demonstrate the round-five window: Raw rounds 2–4 plus a ready Observation for round 1, or Raw rounds 1–4 when that Observation is pending, failed, or invalid.
- [x] Regression coverage proves that Observation work does not delay either mode, does not increment Tool rounds, and cannot alter an already terminal answer.
- [x] Regression coverage proves that no runtime evidence crosses checkpoint, Durable State, Step-context, recovery, or execution boundaries.
- [x] Regression coverage preserves failure behavior for required-Raw budget overflow and synchronous validated-Observation merge failure, including source-round and tool-call-ID provenance.

## Answer

Added `tests/test_ticket05_asynchronous_context.py` with public-seam end-to-end coverage for ReAct and Plan-step Executor round-five Raw fallback behavior. The tests use controlled asynchronous model doubles, verify complete call-ID-attributed Raw evidence while Observation work is pending, verify terminal completion remains successful, and verify checkpoint snapshots exclude transient runtime evidence. Existing runtime-context tests retain coverage for ready Observation replacement, provider/validation failure fallback, budget overflow, synchronous merge failure, provenance, model-use accounting, and transient lifecycle boundaries.

Verification: `poetry run pytest -q` — 265 passed, 7 skipped.
