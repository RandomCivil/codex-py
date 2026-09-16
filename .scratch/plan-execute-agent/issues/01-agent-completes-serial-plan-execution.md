# 01: Agent completes serial Plan execution

**What to build:** A host application can submit Agent state to one asynchronous Agent entry point and receive a successful terminal result with final Agent state after an initial Plan is derived and each Plan step completes in order. The Agent owns the Executor context for this goal run, so MCP resources are entered once and released when it ends.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] An Agent result distinguishes successful completion and retains the final Agent state.
- [ ] A new goal derives its initial Plan, records each completed Step execution, and executes its Plan steps in declared order.
- [ ] The Agent enters and exits the supplied Executor context once per run, and a fully completed Plan returns successfully without unnecessary work.
- [ ] High-level Agent tests use controlled Planner and Executor collaborators rather than live model or MCP dependencies.

