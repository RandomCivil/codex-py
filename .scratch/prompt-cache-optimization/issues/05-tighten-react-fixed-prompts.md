# 05: Tighten ReAct fixed prompts by Structured-output mode

**What to build:** ReAct mode minimizes fixed tool-loop and terminal goal-completion prompt text whenever its strict Structured-output contract already enforces it, while retaining evidence-based Tool execution and the JSON-object compatibility contract.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] Strict-schema ReAct requests omit only response-shape instructions guaranteed by the configured terminal contract; the JSON-object and no-structured-output paths retain the text needed to produce and locally validate a goal-satisfied response.
- [ ] Tool selection, Tool-result handling, Runtime context, round limits, final goal-completion behavior, and completed or failed Execution answers remain unchanged.

