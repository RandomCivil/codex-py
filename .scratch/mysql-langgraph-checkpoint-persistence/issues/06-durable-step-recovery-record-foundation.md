# 06: Durable Step recovery-record foundation

**Parent:** 05: Fresh interrupted-step recovery records

**What to build:** An Agent run durably records each Execution attempt's Step execution and successful Context update in a project-owned MySQL recovery record, so that the latest mutually confirmed state can be read back to reconstruct trustworthy Agent state and Step context without depending on LangGraph saver-internal tables.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] `agent migrate` idempotently creates the project-owned, immutable Step recovery-record storage alongside the existing run foundation, without modifying third-party Checkpoint schema ownership.
- [ ] A normal Plan execution records its running, completed, failed, and interrupted attempts with a monotonic attempt number; successful attempts include their validated Context update, while Agent state continues to expose the latest Step execution for each Plan step.
- [ ] Checkpoints and recovery records carry a common monotonic version. Reading recovery state reconstructs every latest Step execution and Context updates only from completed Steps; an absent or unconfirmable common version returns the existing persistence outcome before model or MCP Tool work.
- [ ] Controlled durable-Agent tests and optional local-MySQL integration tests prove attempt-history round trips, Context-update restoration, idempotent migration, and fail-closed version mismatch behavior without asserting saver-internal SQL or Checkpoint IDs.
