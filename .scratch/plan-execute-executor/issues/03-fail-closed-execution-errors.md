# 03: Fail closed on orchestration errors and return tool errors to the model

**What to build:** Orchestration failures reliably become failed Step executions before unsafe work continues. Invalid Plan-step references and model failures are terminal for the current Step execution. MCP tool errors and Atom command results with nonzero exit codes are returned to the model as error context so it can choose a corrective next action, without hidden Executor retry, replanning, or mutation of Agent state or a Plan.

**Blocked by:** 01 — Deliver a usable Atom MCP Plan-step Executor.

**Status:** ready-for-agent

- [x] Unknown revision or Plan-step IDs fail before an MCP session or tool invocation is created.
- [x] Model failures produce diagnostic failed Step executions. MCP errors and Atom command nonzero exit codes are returned to the model and cannot count as successful tool use.
- [x] Failure handling leaves the input Agent state and all Plan revisions unchanged and does not make another model or tool attempt for the Step execution.
