# 02: Deterministic task routing

**What to build:** A validated Task Analysis deterministically selects the appropriate whole-task Execution mode: direct for tool-free goals, tool-agent for short certain tool work, Plan–execute for long or replan-heavy work, and ReAct otherwise.

**Blocked by:** 01: Task Analyzer contract.

**Status:** resolved

- [x] The ordered routing policy implements the agreed `needs_tools`, expected-step, path-uncertainty, horizon, subgoal, and replanning thresholds exactly.
- [x] Boundary and precedence tests demonstrate that no-tools and short-certain-tool rules take precedence over later Plan–execute conditions.
- [x] The Plan-step Executor remains exclusive to Plan–execute mode; routing selects existing Execution modes rather than introducing generic executors.

## Answer

Implemented the ordered `route(TaskAnalysis)` policy and exported it through
the `agent` package. Tests cover every branch, threshold boundary, and the
no-tools/short-certain-tool precedence over Plan–execute conditions. Existing
Execution modes and the Plan-step Executor boundary remain unchanged.

## Comments

- Verified with `poetry run pytest -q`: 175 passed, 5 skipped.
