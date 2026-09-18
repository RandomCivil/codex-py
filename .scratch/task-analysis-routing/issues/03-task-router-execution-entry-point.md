# 03: Task Router execution entry point

**What to build:** An application can submit one goal to a Task Router, which analyzes it before work begins, creates and runs the selected Execution mode, then returns the existing Execution answer together with the selected mode and analysis evidence.

**Blocked by:** 01: Task Analyzer contract; 02: Deterministic task routing.

**Status:** resolved

- [x] The router performs analysis, deterministic selection, and exactly one selected Execution-mode invocation through the existing factory seam.
- [x] Analysis invocation or validation failures conservatively select Plan–execute, include a safe analysis-error value, and do not retry analysis or switch modes after execution starts.
- [x] The routed outcome preserves the selected mode's completed/failed Execution answer unchanged while exposing mode and optional analysis metadata.
- [x] Tests use controlled analyzer and mode-factory doubles, require no live model, MCP, or MySQL services, and prove no post-selection fallback or escalation occurs.

## Answer

Implemented and exported the `TaskRouter` seam with `RoutedExecutionAnswer`.
It analyzes once, selects one mode through the injected factory, runs that mode
once, and preserves its completed/failed `ExecutionAnswer` unchanged. Analyzer
invocation and validation failures conservatively select `plan_execute` with a
safe analysis error; execution failures do not trigger retry, fallback, or
escalation. Tests use controlled analyzer and mode doubles only.

## Comments

- Verified with `poetry run pytest -q`: 175 passed, 5 skipped.
