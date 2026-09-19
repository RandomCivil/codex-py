# 01: Shared Runtime context policy

**What to build:** ReAct mode and the Plan-step Executor can share one Runtime context policy that keeps the newest three Tool rounds as exact Raw tool results, turns earlier rounds into attributable Observations, uses a default 128,000-token component-configurable budget, and fails explicitly when context maintenance cannot preserve its contract.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] A Tool round's final arguments and request-ordered settled results, including failures, are retained as Raw tool results; only its Observation enters context after it leaves the three-round raw window.
- [ ] Every settled Tool-call batch produces a no-tool, strict-schema Observation with a round identifier and source-ID-attributed confirmed facts, reported errors, and inferences; generation or validation failure fails the active execution without consuming a Tool round.
- [ ] The policy assembles Durable State, older Observations, and at most three newest Raw tool results for all model requests; it supports configured budgets, oldest-first strict-schema Observation merging, and explicit failure when required context cannot be maintained.
- [ ] Controlled tests cover raw-window rollover, observation provenance and validation, merge behavior, model-use versus Tool-round accounting, default and overridden budget configuration, and failures without live providers or MCP tools.
