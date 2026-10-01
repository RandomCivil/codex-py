# 03: Run ReAct mode through the Responses API

**What to build:** ReAct mode completes its existing iterative Tool loop through the shared streamed Responses API path, including native Tool-result continuation and Completion judgment, while producing independently attributable logs for every model request.

**Blocked by:** 01: Complete the Responses stream observability contract.

**Status:** ready-for-agent

- [ ] ReAct can continue from native function-call output with host-executed Tool results until it reaches its existing terminal decision behavior.
- [ ] Completion judge and judge-repair requests remain tool-free and are traced independently as `completion_judge`; Tool rounds remain traced as `react`.
- [ ] Existing ReAct Tool execution, round-limit, validation, and Completion-judgment semantics remain externally unchanged.
