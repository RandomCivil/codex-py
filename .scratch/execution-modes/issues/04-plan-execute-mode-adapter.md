# 04: Plan–execute mode adapter

**What to build:** An application integrator can select `plan_execute` through the same factory while retaining the existing durable Planner/Plan-step Executor workflow, recovery behavior, and a common Execution answer result.

**Blocked by:** 01: Execution mode factory and Direct mode.

**Status:** ready-for-agent

- [x] The adapter delegates to the established Plan–Execute path rather than reimplementing planning, Plan-step execution, persistence, leases, Checkpoints, or recovery.
- [x] A completed durable run produces an Execution answer from its last completed Plan-step handoff; blocked or failed runs produce failed Execution answers without a new model synthesis request.
- [x] Existing Python injection seams and observable durable behavior remain compatible.
- [x] Public-seam tests prove completed and failed mapping while existing Agent, DurableAgent, Executor, and recovery tests remain green unchanged in intent.

## Comments

- Implemented `PlanExecuteMode` and factory wiring in `agent/execution.py`.
- Added public-seam tests for completed handoff mapping and blocked-run failure mapping.
