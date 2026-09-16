# 03: Fail closed on execution and tool errors

**What to build:** Orchestration and tool failures reliably become failed Step executions before unsafe work continues. Invalid Plan-step references, model failures, MCP tool errors, and Atom command results with nonzero exit codes are terminal for the current Step execution, with no hidden retry, recovery, replanning, or mutation of Agent state or a Plan.

**Blocked by:** 01 — Deliver a usable Atom MCP Plan-step Executor.

**Status:** ready-for-agent

- [x] Unknown revision or Plan-step IDs fail before an MCP session or tool invocation is created.
- [x] Model failures, MCP errors, and Atom command nonzero exit codes each produce diagnostic failed Step executions immediately and cannot count as successful tool use.
- [x] Failure handling leaves the input Agent state and all Plan revisions unchanged and does not make another model or tool attempt for the Step execution.
