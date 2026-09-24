# 01: Judge a Tool-backed Plan step

**What to build:** After a successful Tool-call batch, the Plan-step Executor obtains an independent Completion judge verdict for the selected Plan step's criterion. A valid pending verdict keeps the Step active; a valid completed verdict stops Tool work and leads to a separate final handoff from the Tool-calling model. The host, rather than the Tool-calling model's prose, owns criterion progress.

**Blocked by:** None (can start immediately).

**Status:** completed

- [x] Each settled batch with a successful Tool call creates one tool-free judge request using the Executor's provider/model, the selected criterion, accumulated available Step evidence, and that batch's successful Raw results in request order.
- [x] A locally validated `STEP_COMPLETION_PROGRESS` judgment can report no progress or complete only criterion `1` with concise evidence; it contains no final `ANSWER`, and Tool-calling-model prose cannot change progress.
- [x] The next operational model request shows the selected criterion's `pending` or `completed` state. A completed judgment leads to a tool-free, nonempty Step handoff from the Tool-calling model; no later Tool call executes for that Step.
- [x] Judge requests are traced and metered as `completion_judge` without consuming the operational Tool-round budget. Controlled Executor behavior demonstrates pending continuation and completed handoff through its public execution result.
