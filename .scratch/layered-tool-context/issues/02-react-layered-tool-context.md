# 02: ReAct layered Tool context

**What to build:** A ReAct invocation follows the shared Runtime context policy throughout its ephemeral Tool loop: it maintains exact recent evidence and compact older observations, applies that context to operational and final completion requests, and preserves the existing concurrent tool-batch, tool-error, and goal-satisfaction behavior.

**Blocked by:** 01: Shared Runtime context policy.

**Status:** ready-for-agent

- [ ] Each ReAct Tool-call batch creates its Observation and all later ReAct model requests receive the correctly assembled Runtime context, including the final structured goal-satisfaction request.
- [ ] ReAct retains its current Tool-round limit, concurrent request-order batch behavior, constrained Tool runtime, and ephemeral lifecycle while Observation work counts only as model use.
- [ ] ReAct fails cleanly for invalid or failed Observation/merge work and provider oversized-input rejection, without automatic fallback or persistence.
- [ ] Public-seam tests through `ReactMode.run` verify rollover, exact recent tool-error context, final-completion context, budget maintenance failures, and unchanged successful completion behavior.
