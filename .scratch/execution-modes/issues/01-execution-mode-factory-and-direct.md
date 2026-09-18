# 01: Execution mode factory and Direct mode

**What to build:** An application integrator can explicitly select `direct` through a common Execution mode factory, run a whole-task goal, and receive a completed or failed Execution answer from one tool-free language-model call. The existing Plan-step Executor remains unchanged in meaning.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] The factory exposes explicit `direct`, `tool_agent`, `react`, and `plan_execute` selections and a common whole-task runner contract without task analysis or routing logic.
- [ ] Direct mode makes one call with no tools, completes only for nonempty text, and returns a safe failed Execution answer for empty output or invocation errors.
- [ ] Tests exercise the factory-and-runner public seam and prove existing Planner/Plan-step Executor terminology and behavior are not generalized or changed.
