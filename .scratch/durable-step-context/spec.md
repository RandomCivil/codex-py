# Durable Step Context

Status: ready-for-agent

## Problem Statement

The durable Agent graph declares `files_modified`, `files_read`, and `observations`, but does not populate them or supply them to the Tool-calling model. Consequently, one Plan step cannot reliably hand its file activity and concise discoveries to the next Plan step, including after a Checkpoint resume.

## Solution

Add durable Step context to the Plan–Execute flow. Each successful Step execution returns a strict Context update together with its existing Execution result. The Durable Agent merges that update into its checkpointed Graph state, then supplies the accumulated context as a separate Markdown message to the next Step execution's Tool-calling model. Failed executions never infer or persist a Context update.

## User Stories

1. As an Agent developer, I want every later Plan step to receive the files previously read, so that it can avoid repeating exploration.
2. As an Agent developer, I want every later Plan step to receive the files previously modified, so that it can safely build on prior work.
3. As an Agent developer, I want every later Plan step to receive concise observations, so that relevant discoveries survive beyond the prior model conversation.
4. As an Agent developer, I want Step context to accumulate for the entire Agent run, so that a later Plan revision retains successful work from earlier revisions.
5. As an Agent developer, I want Step context Checkpointed with the Durable Agent graph, so that a resumed Agent run receives the same handoff context without re-executing completed work.
6. As an Executor consumer, I want a successful completion receipt to contain its Context update, so that the Execution result and the facts needed by later steps are captured atomically.
7. As an Executor consumer, I want the completion receipt to remain strictly machine-validatable, so that malformed context cannot be recorded as durable Agent state.
8. As an Agent developer, I want a failed Step execution to preserve its existing error behavior without a Context update, so that the system does not represent guessed or partial tool activity as fact.
9. As an Agent developer, I want Context updates to contain only relative, normalized POSIX paths within the Agent working directory, so that later model requests are portable and do not contain path traversal or machine-specific absolute paths.
10. As an Agent developer, I want duplicate file paths merged while preserving their first appearance, so that Step context remains stable and compact.
11. As an Agent developer, I want observations attributed by the system to the actual Plan revision and Plan-step ID, so that an execution model cannot misrepresent the source of a discovery.
12. As a Tool-calling model, I want Step context delivered as readable Markdown in a separate message, so that it is distinct from the JSON Model request containing Agent state and the selected Plan step.
13. As a Tool-calling model, I want the three Step-context sections to be present even when empty, so that the input contract is predictable on the first Plan step.
14. As a maintainer, I want the Executor to return an explicit outcome containing both the Step execution and Context update, so that it does not mutate Durable Graph state or hide cross-layer data in `StepExecution`.
15. As a maintainer of the non-durable Agent loop, I want it to retain its existing Agent-state recording behavior by consuming the outcome's Step execution, so that durable context does not leak into Agent state.
16. As an operator, I want a terminal or resumed Agent run to retain existing result serialization, so that this feature does not change the CLI's public Agent-state result contract.

## Implementation Decisions

- Treat Step context and Context update as defined in the project glossary. Step context is durable Graph state, not Agent state, raw model messages, or raw tool traces.
- Maintain the top-level Graph-state fields `files_read`, `files_modified`, and `observations` as cumulative lists. Initialize absent fields as empty lists and merge only Context updates from completed Step executions.
- Introduce an explicit immutable `ContextUpdate` domain value containing `files_read`, `files_modified`, and observation text, and an immutable `ExecutionOutcome` containing a `StepExecution` and an optional Context update. The Executor returns this outcome rather than directly changing Agent state or Graph state.
- Extend the existing forced, strict completion receipt rather than making another model call. A valid completed receipt includes non-null list fields for file reads, file modifications, and observations; empty lists are valid. A failed Step execution has no Context update.
- Keep the current completion truth requirements and result requirement. Invalid extra fields, invalid types, invalid completion flags, or an invalid context item make the completion receipt invalid and therefore preserve existing failed-execution behavior.
- Accept only non-empty, normalized relative POSIX file paths that remain within the configured Agent working directory. Reject absolute paths, parent-directory traversal, empty entries, and invalid path values. Deduplicate accepted paths across the Agent run while retaining the first-seen ordering.
- Accept non-empty observation text from the model, but attach revision and Plan-step attribution in the Durable Agent while merging it into Graph state. Preserve observations in their completed-step order and do not deduplicate their content.
- Supply Step context to every Executor invocation as a second `HumanMessage`, after the existing serialized JSON Model request. Its Markdown format has a fixed `## Step context` heading and `Files read`, `Files modified`, and `Observations` sections. The first step receives the same three empty sections.
- Pass the current Step context from the Durable Agent to the Executor without adding it to `AgentState`; update the Executor protocol and controlled collaborators accordingly. The ordinary in-memory Agent loop records only `outcome.execution` and does not accumulate durable Step context.
- Preserve the ADR-0002 Executor boundary: the Executor still performs exactly one bounded Step execution, chooses MCP tools at runtime, and does not change a Plan or itself write coordination state. Preserve ADR-0003's Checkpoint behavior by allowing LangGraph to persist the new Graph-state fields unredacted with the rest of the Agent run.
- Do not add an ADR for this feature. The message layout and context receipt shape are localized and reversible implementation contracts, while the existing Executor-boundary and Checkpoint ADRs already explain the architectural constraints.

## Testing Decisions

- The primary behavioral seam is a `DurableAgent.run()` execution driven by a controlled multi-step Planner and Executor. Assert that the second Executor invocation receives the accumulated context from the first successful outcome, and that the final checkpoint retains the same Graph-state context after a fresh Durable Agent resumes the run.
- Test only observable collaborations and values: Executor input context, returned status and serialized Agent state, and resumed behavior. Do not assert LangGraph node order, checkpoint IDs, raw message storage, or private merge helpers.
- Add focused Executor tests using the existing injected controlled Tool-calling model seam. Verify the strict completion receipt accepts valid Context updates, rejects malformed fields and unsafe paths, and emits the fixed separate Markdown Step-context message alongside the unchanged JSON request.
- Add coverage that a failed outcome contributes no files or observations, and that successful outcomes deduplicate file paths by first appearance while observations retain their system-provided revision and step attribution.
- Reuse the established `tests/test_durable.py` in-memory-checkpointer tests for durable lifecycle behavior and `tests/test_executor.py` controlled-model / controlled-MCP tests for strict receipt and request construction. Preserve existing Agent and Executor tests as regression coverage for the changed outcome boundary.

## Out of Scope

- Inferring files read, files modified, or observations from MCP tool names, arguments, raw output, model messages, or filesystem changes.
- Persisting Context updates from failed, interrupted, or malformed Step executions.
- Passing Step context to the Planner or adding it to `AgentState`, Plan revisions, `StepExecution`, CLI results, or Run-registry records.
- Replanning policy changes, Plan schema changes, tool permissions, MCP-server changes, tool retry/recovery changes, or changes to the three-revision budget.
- Pruning, token budgeting, summarization, encryption, redaction, retention, or user editing of accumulated Step context.
- Changes to Atom MCP path semantics beyond validating the context values reported by the Tool-calling model.

## Further Notes

The user confirmed the following design choices during the design review: Context updates are model-reported rather than inferred from MCP output; the context is cumulative across the whole Agent run; it is sent only to the next Plan step's initial Tool-calling-model request; it is human-readable Markdown; only successful Step executions update it; and file paths are normalized relative POSIX paths. The `ExecutionOutcome` boundary was selected to keep the Executor independent of both Agent state mutation and Durable Graph-state mutation.
