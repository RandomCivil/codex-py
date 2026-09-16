# ToolNode-backed Executor boundary

The Executor is limited to one Plan step and uses a LangChain `ChatOpenAI` tool-calling model plus a `model → ToolNode → model` StateGraph to select and invoke a host-configured MCP tool set. It accepts Agent state plus a revision and Plan-step ID, returns a completed or failed Step execution without changing a Plan, retries neither an MCP error nor a whole step, and allows at most ten model/tool rounds. A completed execution must include at least one successful tool call and a strict JSON result that addresses the completion criterion; an MCP error or an `exec` result with a nonzero exit code fails the step immediately. The model receives the complete serializable Agent state. Atom MCP is connected through an overridable stdio configuration, whose development default runs `poetry run atom-mcp` in `/home/xzp/workspace/atom-mcp`; the Executor owns its session through its async context manager, inherits its environment, and accepts supplemental environment values. This keeps tool authority and failure history at the boundary established by ADR-0001.

## Considered Options

- Reuse the Planner's text-only model seam — rejected because it cannot represent tool calls safely.
- Implement the top-level Agent loop now — rejected because planning/replanning coordination remains a distinct future boundary.
