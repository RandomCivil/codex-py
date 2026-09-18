# 02: Component Provider requests and Structured-output

**What to build:** An Agent run uses its resolved Planner and Executor Component provider configurations independently. Both `json_schema` and `json_object` support valid, locally verified Plans and Execution results; Executor tool-selection requests use the Executor configuration while MCP Tool execution remains host-defined.

**Blocked by:** 01: Configuration file entry and validation.

**Status:** ready-for-agent

- [ ] Planner and Executor receive their own resolved endpoint, API key, and model name during runtime composition.
- [ ] Each component applies its selected Structured-output mode to its final structured response and retains local contract validation in both modes.
- [ ] Executor tool-selection remains compatible with MCP function calls and uses its resolved Component provider configuration.

