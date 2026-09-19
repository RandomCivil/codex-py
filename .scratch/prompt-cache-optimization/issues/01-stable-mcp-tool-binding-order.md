# 01: Stabilize MCP tool-set binding order

**What to build:** Host integrators receive the same canonical MCP tool-set order whenever the effective allowlist is the same, so tool-capable Model requests retain a stable tool-definition prefix and Tool execution authority remains unchanged.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] ReAct mode, Tool-agent mode, and the Plan-step Executor bind each filtered MCP tool set in canonical tool-name order.
- [ ] Differently ordered MCP server enumeration produces the same bound tool order without changing the effective allowlist, Tool execution, terminal Execution answer, or completion receipt behavior.

