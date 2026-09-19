# Layered tool context for ReAct and Plan-step Executor

Status: ready-for-agent

## Problem Statement

ReAct and the Plan-step Executor currently retain their complete in-loop message histories as tool work proceeds. Long tool sessions therefore accumulate raw request arguments and raw tool results until they crowd out useful task context or exceed a provider input limit. The two loops need one consistent context-engineering policy that preserves exact evidence needed for immediate correction, retains a compact and attributable account of older work, respects Plan–execute recovery boundaries, and does not change ReAct's ephemeral nature.

## Solution

Introduce one shared runtime-context policy used by ReAct and the Plan-step Executor. For every model request in an active tool loop, it supplies applicable Durable State, exact Raw tool results for the most recent three Tool rounds, and Observations for all older Tool rounds. Every Tool-call batch produces a no-tool, strict-schema Observation request. Its Observation is held out while its raw result remains among the newest three and becomes model context only when that raw result expires. The default context policy budget is 128,000 tokens and is configurable per component. When the budget requires compaction, oldest Observations are merged by another strict-schema, no-tool model request; inability to generate or validate an Observation or merged Observation, or a provider oversized-input rejection, fails the active execution.

For Plan–execute, transient Raw tool results and Observations start empty for each Step execution. The Executor receives only existing Durable State: complete Agent state plus successful prior Context updates. Transient data must not be persisted in LangGraph Checkpoints, while existing `running` markers and successful Context updates remain durable. Complete raw results remain available through the explicit diagnostic trace but are not later model context or recovery state.

## User Stories

1. As an application integrator, I want ReAct and the Plan-step Executor to use the same Runtime context window policy, so that tool-driven behavior is predictable across execution modes.
2. As an application integrator, I want each tool-capable model request to receive the latest three Tool rounds as exact Raw tool results, so that the model can inspect recent arguments, outputs, and errors precisely.
3. As an application integrator, I want all older Tool rounds represented by Observations, so that useful historical evidence remains available without retaining an unbounded raw transcript.
4. As an application integrator, I want the final structured completion request to use the same Runtime context window, so that it can substantiate completion using the same evidence as operational requests.
5. As an application integrator, I want a Tool round to mean one tool-capable model request and its complete Tool-call batch, so that concurrent calls remain one coherent unit of history.
6. As an application integrator, I want a Raw tool result to retain complete final tool-call parameters and ordered results, including failures, so that recent corrective actions have their original evidence.
7. As an application integrator, I want all calls in a Tool-call batch represented in request order, so that the context preserves the existing concurrent batch contract.
8. As an application integrator, I want an Observation generated after every settled Tool-call batch, so that every Tool round has a validated compact representation before its raw data expires.
9. As an application integrator, I want Observation generation to use a no-tool, strict-schema model request, so that context maintenance cannot invoke host tools or produce unvalidated prose.
10. As an application integrator, I want Observation generation to count toward model use but not toward the Tool-round limit, so that `max_rounds` continues to bound side-effect-capable Tool-call batches.
11. As an application integrator, I want an Observation to distinguish confirmed facts, reported errors, and model inferences, so that model interpretation is never presented as tool-confirmed evidence.
12. As an application integrator, I want every Observation entry attributable to its originating tool-call IDs and round, so that compact history remains auditable after raw results expire.
13. As an application integrator, I want the three newest rounds to contribute raw results without duplicate Observations, so that exact evidence is not redundantly charged against the context budget.
14. As an application integrator, I want an expired round's Observation to enter context only after a fourth newer Tool round has completed, so that the Runtime context window follows the defined three-raw-round boundary.
15. As an application owner, I want the Runtime context budget to default to 128,000 tokens and be configurable independently for ReAct and the Executor, so that deployments can tune context capacity by component.
16. As an application owner, I want the oldest Observations to be combined when the budget would otherwise be exceeded, so that historical information is compacted before required current evidence is discarded.
17. As an application integrator, I want Observation compaction to use the same component's no-tool, strict-schema model capability, so that the merged record preserves Observation semantics and source attribution.
18. As an application owner, I want compaction to retain the merged Observation's source round range and tool-call IDs, so that a later model can assess the provenance of condensed history.
19. As an application owner, I want execution to fail if Observation generation, Observation validation, or Observation compaction fails, so that context-boundary failures cannot silently change the evidence model.
20. As an application owner, I want execution to fail when the provider rejects an oversized input, so that the absence of a cross-provider tokenizer contract does not create a fallback with altered behavior.
21. As a ReAct caller, I want the mode to remain an Ephemeral execution, so that adding Runtime context management does not introduce checkpoints, leases, resume behavior, or Durable State.
22. As a Plan–execute caller, I want each new Step execution to begin without prior steps' Raw tool results or Observations, so that the established Plan-step boundary remains intact.
23. As a Plan–execute caller, I want a new Step execution to inherit complete Agent state and accumulated successful Step context, so that durable goal, plan, execution, file, and observation facts remain available.
24. As a recovery operator, I want Raw tool results and runtime Observations excluded from LangGraph Checkpoints, so that an interrupted Step recovery begins only from trustworthy Durable State.
25. As a recovery operator, I want the existing `running` checkpoint marker and successful Context updates retained, so that the established fail-closed and recovery behavior remains operational.
26. As an operator, I want complete Raw tool results retained by the explicit diagnostic trace, so that diagnostic workflows retain their current evidence without making trace output recoverable model context.
27. As an operator, I want trace records never reintroduced into a later Runtime context window, so that diagnostics do not become an undocumented memory channel.
28. As a maintainer, I want tool errors and nonzero tool results preserved in both Raw results and their Observation error category, so that corrective model behavior remains compatible with current loops.
29. As a maintainer, I want the shared policy to preserve the existing Tool-call batch concurrency, request-order result, allowlist, forced working-directory, and tool-runtime lifecycle contracts.
30. As a maintainer, I want no changes to Planner ownership, Plan immutability, or the Agent's replanning decisions, so that this feature remains a context-engineering concern.

## Implementation Decisions

- Define a shared Runtime context policy as the only new composition seam. It owns the in-memory Tool-round history, Observation generation and validation, context assembly, budget-triggered Observation merging, and transient-state cleanup. ReAct and the Plan-step Executor delegate context construction to it rather than implementing separate retention logic.
- Preserve existing public execution seams: ReAct remains invoked through `ReactMode.run(goal)` and Plan-step work through `Executor.execute(state, revision, step_id, ...)`. The new policy is injected or configured at those construction boundaries without changing their whole-task or Plan-step responsibilities.
- Model each Tool round as the final tool-call arguments after host working-directory enforcement plus the full request-ordered settled batch results. Include successful results, malformed-call failures, MCP failures, and nonzero command results.
- Use a strict structured Observation contract with a required round identifier and always-present `confirmed_facts`, `reported_errors`, and `model_inferences` collections. Entries retain text plus originating tool-call IDs; inference entries name their evidence call IDs.
- Generate an Observation immediately after every Tool-call batch settles, using the same component provider configuration with tools unavailable. Generation requests count as model calls but do not increment the existing positive Tool-round budget.
- Assemble every operational and final-completion model request from Durable State, earlier Observations, and the exact raw results of at most the newest three Tool rounds. Do not include Observations for those recent raw rounds.
- Configure a component-level Runtime context budget with a default of 128,000 tokens. Do not introduce a universal provider tokenizer or token-counter interface. Treat the configured budget as the policy threshold and handle a provider oversized-input rejection as an execution failure.
- When assembly needs further space, merge the oldest Observations through a same-component, no-tool, strict-schema model request. The merged record preserves the complete source-round range and all retained source tool-call IDs. Do not truncate required Durable State or the three newest Raw tool results; fail if they cannot fit or any maintenance request fails.
- Keep ReAct Runtime context entirely in memory for its invocation lifetime.
- For Plan–execute, initialize the policy afresh for each Step execution. Supply existing Agent state plus cumulative successful `Step context` as Durable State, including goal, memory summary, Plan history, Step executions, files read, files modified, and durable observations.
- Refactor the checkpointed Executor graph so Raw tool results and runtime Observations do not enter LangGraph `MessagesState` checkpoint serialization. Retain the pre-tool `running` checkpoint, scheduling state, and existing successful Context update persistence.
- Leave diagnostic tracing of complete Raw tool results intact. Define trace output as diagnostic-only and prohibit its use as Runtime context, Step context, or recovery input.
- Preserve the current model/tool response contracts: ReAct still requires its structured goal-satisfied proof; the Executor still requires its structured completion receipt; tool-capable operational requests remain compatible with host-defined tool schemas.

## Testing Decisions

- Test observable behavior through the shared Runtime context policy seam and the existing public ReAct and Executor execution seams. Do not assert private message-list topology, prompt wording, or LangGraph node internals.
- Use controlled model doubles that capture submitted contexts and produce tool calls, strict Observation responses, merged Observations, completion responses, malformed responses, and provider-style failures without live credentials.
- Use controlled Tool runtime or MCP tool doubles to verify concurrent request-order Tool-call batches, successful results, tool errors, nonzero command results, and final enforced arguments.
- Verify both ReAct and Executor submit exactly the last three Tool rounds as complete raw data, move the fourth-oldest round to Observation context, and omit Observations for the three raw rounds.
- Verify Observation schemas require all three evidence categories and source IDs, and reject invalid or incomplete observations.
- Verify Observation creation happens once after every Tool-call batch, uses no tools, does not consume `max_rounds`, and fails the active execution when it fails or is invalid.
- Verify oldest-first Observation compaction when context budget policy requests it, including preserved round ranges and source IDs, no compaction of required raw data, and failure on invalid or failed merge responses.
- Verify default 128,000-token configuration and component-level overrides are supplied to both modes.
- Verify final structured ReAct completion and Executor completion receipt receive the assembled Runtime context window.
- Verify a new Executor Step starts without a previous Step's Raw results or runtime Observations while receiving its Agent state and cumulative successful `Step context`.
- Verify checkpointed Executor state excludes Raw results and runtime Observations but preserves current `running` and completed Context-update recovery behavior.
- Verify traces still receive complete Raw tool data, while no trace data is used to reconstitute a Runtime context or recovery context.
- Existing ReAct public-seam tests, Executor ToolNode/batch/recovery tests, DurableAgent recovery tests, and trace tests are the prior art. Preserve their documented behavior unless the new Runtime context assertions intentionally supersede transcript retention.

## Out of Scope

- Changing Direct mode or Tool-agent mode.
- Changing Planner behavior, Plan schemas, Plan revision budgets, Agent replanning, or the Agent's durable scheduling ownership.
- Making ReAct durable, resumable, checkpointed, leased, or recoverable.
- Persisting Raw tool results or runtime Observations as Step context, Context updates, recovery records, or LangGraph Checkpoint message state.
- Replacing or redacting the existing full diagnostic trace policy.
- Adding a provider-independent tokenizer, exact cross-provider token accounting, or a new token-counter plugin contract.
- Treating tool-output prompt injection as a new security policy in this feature.
- Changing MCP tool permission, allowlist, forced-working-directory, concurrency, or session-lifecycle policy.
- Adding CLI mode selection, live provider tests, live MCP tests, or live MySQL requirements.

## Further Notes

This specification uses the project glossary terms Raw tool result, Observation, Runtime context window, Tool round, Durable State, Step context, Context update, ReAct mode, and Executor. It implements ADR-0010 while preserving ADR-0002's Plan-step Executor boundary, ADR-0004's fresh recovery boundary, ADR-0005's concurrent Tool-call batch semantics, and ADR-0007's separation of ReAct from Plan–execute.
