# 04: Apply Executor host authority controls

**What to build:** A host can adapt the Executor's Atom MCP integration without editing its logic: it can override stdio command details, supplement inherited environment values, limit which MCP tools are exposed to the model, and reuse one properly closed MCP session for sequential Step executions.

**Blocked by:** 01 — Deliver a usable Atom MCP Plan-step Executor.

**Status:** ready-for-agent

- [x] The host can override Atom stdio command, arguments, and working directory, while inherited environment remains the default and supplied environment values supplement it.
- [x] An optional tool allowlist limits the MCP tool set made available to the model without changing Plan semantics.
- [x] Multiple sequential Step executions reuse the Executor's session during its context lifetime, and its closure releases MCP resources predictably.
