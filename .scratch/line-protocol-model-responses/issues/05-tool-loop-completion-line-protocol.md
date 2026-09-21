# 05: Tool-loop completion Line Protocol slice

**What to build:** ReAct and Plan-step execution use Line Protocol for every non-tool reply while retaining native tool calls, tool-loop boundaries, and their existing separate final-completion request.

**Blocked by:** 01: Line Protocol core and Planner slice; 04: Runtime-context Observation Line Protocol slice.

**Status:** ready-for-agent

- [ ] A tool-selection reply is either native function-tool calls or an otherwise empty `NO_TOOL` block; any other no-tool text or mixed response fails.
- [ ] A valid `NO_TOOL` response triggers the existing separate completion request rather than being treated as completed work.
- [ ] ReAct accepts completion only from a valid `GOAL_COMPLETION` with `GOAL_SATISFIED=true`; Plan-step execution accepts only a valid `STEP_COMPLETION` satisfying its existing receipt contract.
- [ ] Tool definitions and their parameter schemas remain native/provider-facing, while model replies use prompts and local codecs instead of provider output-format settings.
- [ ] Tests cover native tool-call continuity, `NO_TOOL`, separate completion calls, malformed/mixed replies, completion validation, and provider-format omission.
