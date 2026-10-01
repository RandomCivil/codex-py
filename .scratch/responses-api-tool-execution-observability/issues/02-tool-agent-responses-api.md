# 02: Run Tool-agent mode through the Responses API

**What to build:** Tool-agent mode produces the same completed answer or host-rendered first Tool result as before, but every request travels through the shared streamed Responses API path and receives independently attributable `tool_agent` usage and timing logs.

**Blocked by:** 01: Complete the Responses stream observability contract.

**Status:** ready-for-agent

- [ ] A no-tool Tool-agent answer, its Line Protocol repair request, and their final result retain the current user-visible behavior through the Responses API.
- [ ] A Tool-agent function-call response executes only the first requested MCP Tool and host-renders its result as the final answer.
- [ ] Each Tool-agent provider request has a separate, provider-backed timing and usage trace record.
