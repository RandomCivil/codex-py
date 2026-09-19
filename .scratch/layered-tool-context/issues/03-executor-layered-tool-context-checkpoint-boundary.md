# 03: Executor layered Tool context and Checkpoint boundary

**What to build:** Each Plan-step Executor invocation uses the shared Runtime context policy while receiving only its existing Durable State from prior successful work; Raw tool results and runtime Observations remain available for diagnostics during the process but never become checkpoint, recovery, or next-step context.

**Blocked by:** 01: Shared Runtime context policy.

**Status:** ready-for-agent

- [ ] The Executor uses the shared context for every operational and structured completion request, creates an Observation after each Tool-call batch, and preserves existing completion, Tool-round, concurrency, tool-error, and host-authority semantics.
- [ ] A new Step execution begins with no prior step's Raw results or runtime Observations but receives complete Agent state and accumulated successful Step context as Durable State.
- [ ] Checkpoints exclude Raw tool results and runtime Observations while retaining existing `running` persistence, successful Context updates, fresh recovery behavior, and full diagnostic trace records.
- [ ] Public Executor and DurableAgent/recovery tests verify context rollover, final-completion evidence, checkpoint exclusion, trace availability, and recovery compatibility without live provider, MCP, or MySQL requirements.
