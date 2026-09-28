# 09: Runtime-context messages rendering and budget

**What to build:** Make the Runtime context capable of producing the native
`messages` layout while preserving the existing `grouped` rendering. A visible
Raw Tool call becomes a matching AI tool-call and Tool result pair, followed by
the ordered Durable State, Observations, feedback, Goal, and acceptance-criteria
messages. The full selected request has bounded Runtime-context behavior.

**Blocked by:** 08: Simplify Runtime-context evidence presentation.

**Status:** resolved

## Acceptance criteria

- [ ] A reconstructed Tool pair preserves call ID, name, arguments, original
  request order, complete stable result content, and failure status/error. A
  partial batch contains only its visible Raw calls and has no dangling pair.
- [ ] An Observation-replaced call has neither reconstructed AI nor Tool
  message; its model-facing evidence occurs only in the Observations message.
- [ ] Native `grep` and `read_file` results are Tool messages under `messages`;
  no file grouping, Markdown result wrapper, or synthetic Round heading occurs.
- [ ] The complete `messages` request is budgeted. Existing Observation merging
  occurs before explicit failure when required Durable State or Raw evidence
  cannot fit.
- [ ] Existing `grouped` rendering and all evidence lifecycle, diagnostics,
  Checkpoint, and recovery boundaries remain unchanged.

See [the feature specification](../spec.md) for the full contract.

## Comments

- Implemented native `messages` Runtime-context rendering with visible Raw AI/Tool pairs, stable result/error encoding, Observation replacement pruning, ordered trailing context layers, and full-request budget accounting. Verified with focused Runtime-context tests; unrelated existing full-suite failures remain in LLM and ReAct tests.
