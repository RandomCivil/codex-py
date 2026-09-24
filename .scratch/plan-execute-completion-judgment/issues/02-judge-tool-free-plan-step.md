# 02: Judge Tool-free Plan steps and premature exits

**What to build:** A Plan step that needs no Tool call can complete from Agent state and Step context only after a Completion judge confirms its criterion. If the Tool-calling model attempts to finish while the criterion is pending, the Executor returns the pending state and continues within its round budget. Once the criterion is complete, a separate handoff request ends the Step.

**Blocked by:** 01: Judge a Tool-backed Plan step.

**Status:** completed

- [x] A no-tool response while the criterion is pending invokes the judge with available Agent state, Step context, and accumulated Step evidence, even if no Tool has been called.
- [x] A pending verdict rejects the premature answer, provides the selected criterion's pending state and continuation feedback to the Tool-calling model, and consumes an operational Tool round; repeated premature exits eventually fail at budget exhaustion.
- [x] A completed verdict discards the pre-judgment answer and obtains a new, tool-free, nonempty handoff. A Tool-call attempt or empty handoff receives one tool-free repair opportunity and then fails the Step if still invalid.
- [x] Controlled public Executor behavior covers already-satisfied no-tool Steps, pending continuation, completed handoff, repair, and budget exhaustion.

## Comments

- Implemented with TDD. Public Executor tests cover no-tool judging, pending continuation, completed handoff, repair, and round-budget exhaustion. Full suite: 337 passed, 7 skipped.
