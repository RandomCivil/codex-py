# ToolNode-backed Executor boundary

**Status: accepted; completion rule amended by ADR-0020.**

The Executor is limited to one Plan step and uses a LangChain `ChatOpenAI` tool-calling model plus a `model → ToolNode → model` StateGraph to select and invoke a host-configured MCP tool set when the selected Plan step needs one. It accepts Agent state plus a revision and Plan-step ID, returns a completed or failed Step execution without changing a Plan, retries neither an MCP error nor a whole step, and allows at most fifty model/tool rounds. A malformed tool-call argument, MCP error, or an `exec` result with a nonzero exit code is returned to the model as an error tool result so it can correct its next call; it does not fail the Step immediately. The original rule accepted a no-tool model response with nonempty text as a completed Step, with that text serving as the handoff to later Plan steps. ADR-0020 replaces that completion rule with separate judgment over evidence followed by a handoff. Freeform terminal content cannot safely populate structured files-read, files-modified, or observations fields, so the completed execution records an empty `ContextUpdate`. MCP tools retain their native provider bindings because their schemas are host-defined. The model receives the complete serializable Agent state. Atom MCP is connected through an overridable stdio configuration, whose development default runs `poetry run atom-mcp` in `/home/xzp/workspace/atom-mcp`; the Executor owns its session through its async context manager, inherits its environment, and accepts supplemental environment values. This keeps tool authority and failure history at the boundary established by ADR-0001.

## Considered Options

- Reuse the Planner's text-only model seam — rejected because it cannot represent tool calls safely.
- Implement the top-level Agent loop now — rejected because planning/replanning coordination remains a distinct future boundary.
