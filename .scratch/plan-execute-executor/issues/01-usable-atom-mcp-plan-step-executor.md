# 01: Deliver a usable Atom MCP Plan-step Executor

**What to build:** An application can configure an asynchronous Executor, or inject a tool-calling model for tests, and use its context-manager lifetime to execute one valid Plan step against Atom MCP. The Executor resolves the requested revision and Plan-step ID from Agent state, runs a model → ToolNode → model workflow using adapter-loaded MCP tools, and returns a completed Step execution after a successful tool-backed final result without changing Agent state or a Plan.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] An Executor can be configured for an OpenAI-compatible tool-calling model or supplied a test model, and owns a reusable Atom MCP stdio session through asynchronous context management.
- [ ] A valid revision and Plan-step ID execute through `langchain_mcp_adapters` and LangGraph `ToolNode`, returning a completed Step execution while leaving the supplied Agent state and Plan immutable.
- [ ] The model receives the selected Plan step and complete serializable Agent state, and the default Atom development configuration is usable without embedding Atom credentials or policy values.
