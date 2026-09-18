# 02: Shared Tool runtime and tool-agent mode

**What to build:** An application integrator can select `tool_agent` and receive a bounded single-request/single-tool whole-task answer using the host's constrained MCP authority. Tool output is rendered without another language-model call, and failure remains explicit.

**Blocked by:** 01: Execution mode factory and Direct mode.

**Status:** ready-for-agent

- [x] A shared Tool runtime provides one invocation-scoped MCP session, configured tool set, allowlist, forced working directory, trace integration, and deterministic cleanup for tool-enabled Execution modes.
- [x] Tool-agent executes no more than the first requested tool; extra requests are not executed and are traced as ignored, while a no-tool response uses the model text as its answer.
- [x] Text, structured, and multi-content tool results are deterministically rendered without a second model request; tool, validation, nonzero-result, or rendering failures return failed Execution answers.
- [x] Public-seam tests prove the single-call limit, host constraints, rendering behavior, ignored-call trace, lifecycle, and failure contract.

## Comments

- Implemented `ToolRuntime`, `ToolAgentMode`, deterministic tool-result rendering, and factory wiring in `agent/execution.py`.
- Added public-seam tests for first-call-only execution, stable structured rendering, no-tool text answers, and runtime cleanup.
