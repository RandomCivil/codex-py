# 03: Per-call write Observations

**What to build:** Compact write effects without stalling the tool loop: every `apply_patch`, `write_file`, and write-class `exec` call starts one asynchronous Observation after its Tool-call batch settles, while retaining lossless Raw tool-result fallback whenever that compact representation is unavailable.

**Blocked by:** 01: Per-call native read evidence; 02: Conservative exec evidence classification.

**Status:** ready-for-human

- [x] Each observation-class call starts exactly one no-tool, strict-schema Observation request after its containing batch settles, without awaiting it before later Runtime-context assembly.
- [x] A pending Observation leaves its originating call's Raw tool result visible; a validated Observation replaces only that call in a later context snapshot.
- [x] Provider failure, cancellation, invalid structure, and invalid provenance are diagnosable, do not retry or fail the active tool loop, and leave permanent Raw fallback.
- [x] A mixed batch can simultaneously retain short-lived raw reads, permanent raw reads, pending write raw results, and completed write Observations with stable request-order provenance.
- [x] Budget maintenance merges only validated Observations and never truncates, evicts, or re-summarizes required Raw fallback evidence.

## Comments

- Implemented per-call Observation tasks and diagnostics in `RuntimeContextPolicy`, including tool-call provenance validation and per-call Raw fallback.
- Added ticket 3 seam tests covering mixed batches, pending/validated replacement, provider failure, cancellation, invalid provenance, and no-retry behavior.
- Verification: `poetry run pytest -q` — 241 passed, 7 skipped.
