# ReAct completion-criterion judgment

Status: ready-for-agent

## Problem Statement

ReAct completion criteria currently become instructions in the main ReAct Model request, and the ReAct model self-reports progress in its response. This mixes tool selection with verification, causes criteria and their status to dominate the tool-loop context, and lets the tool-calling model claim completion without a dedicated evaluation of the resulting evidence.

## Solution

Task Analyzer will continue to supply an ordered collection of independently verifiable completion criteria when deterministic routing selects ReAct. ReAct will keep the criteria as invocation-local state but remove them, their status, and `REACT_PROGRESS` entirely from its own Model requests and responses.

After every Tool-call batch settles, ReAct will make one separate, tool-free `completion_judge` Model request using the ReAct model's configured provider. The judge receives the complete current ReAct reasoning context, the successful calls and their Raw tool results, and an explicit criterion-progress context containing every ordered criterion and current status. It returns a locally validated `COMPLETION_PROGRESS` Line Protocol document, with concise evidence for every newly completed criterion. The host validates and monotonically updates the invocation-local progress only from this judgment. Failed tool results never constitute judge evidence; they remain immediate correction messages for the next ReAct request.

## User Stories

1. As an application integrator, I want ReAct work to retain explicit ordered completion criteria, so that multi-step tool work has a visible and testable completion standard.
2. As an application integrator, I want Task Analyzer to supply criteria only for ReAct-routed work, so that direct, tool-agent, and Plan–execute retain their existing contracts.
3. As an application integrator, I want deterministic routing to remain responsible for execution-mode selection, so that Task Analyzer stays descriptive.
4. As a ReAct model, I want a context free of completion criteria and progress instructions, so that I can focus on selecting and correcting tool actions.
5. As an application integrator, I want the ReAct model's accompanying tool-call text ignored for criterion progress, so that free-form reasoning cannot mutate authoritative state.
6. As an application integrator, I want successful tool evidence assessed by a separate completion judge, so that criterion completion is based on executed results rather than self-reporting.
7. As a completion judge, I want the complete current ReAct reasoning context, so that I can interpret a tool result in the goal's actual execution context.
8. As a completion judge, I want the complete ordered criterion list and current completion state in my own context, so that I can decide only which remaining outcomes this batch proves.
9. As a completion judge, I want every successful tool call and its Raw tool result from one settled Tool-call batch together, so that related concurrent results can establish an outcome jointly.
10. As an application integrator, I want failed tool results excluded from completion evidence, so that an error cannot be mistaken for proof of completion.
11. As a ReAct model, I want failed tool results immediately returned in my next context, so that I can correct the failed action in the next tool round.
12. As an application integrator, I want mixed-result batches to judge their successful results while returning their failed results to ReAct, so that valid evidence is not discarded because an unrelated concurrent call failed.
13. As an application integrator, I want one judge request per settled batch rather than one per individual call, so that progress does not depend on concurrent completion order.
14. As an application integrator, I want the completion judge to identify criteria by their stable one-based numbers, so that criterion order remains the authoritative reference.
15. As an application integrator, I want each newly completed criterion to include concise, directly checkable evidence, so that progress is auditable without retaining hidden reasoning.
16. As an application integrator, I want a judge to report no progress explicitly when a batch proves nothing, so that no evidence is not confused with an invalid response.
17. As an application integrator, I want duplicate, already-completed, out-of-range, malformed, and evidence-free judgments rejected locally, so that a model cannot skip, repeat, or fabricate progress.
18. As an application integrator, I want one repair attempt after invalid judge output, so that transient format mistakes do not immediately fail execution.
19. As an application integrator, I want the rejected judge output and validation error included only in the judge's repair context, so that the retry can correct itself without contaminating ReAct context.
20. As an application integrator, I want execution to continue with the next ReAct round when the repair attempt is still invalid, so that a judge formatting failure is recoverable.
21. As an application integrator, I want completion-judge requests traced and metered as a distinct `completion_judge` component, so that their cost and behavior are observable separately from ReAct and Runtime-context requests.
22. As an application integrator, I want judge requests excluded from the tool-round budget, so that the budget continues to represent ReAct progress through tools.
23. As an application integrator, I want a criterion-enabled no-tool ReAct response to use the normal `ANSWER` protocol without declaring criterion numbers, so that only the judge owns completion state.
24. As an application integrator, I want ReAct to reject a terminal answer while criteria remain pending, so that incomplete work is never returned as completed.
25. As a ReAct model, I want generic continuation feedback after a rejected terminal answer, so that I can keep working without being shown completion criteria.
26. As an application integrator, I want rejected no-tool responses to consume the existing ReAct round budget, so that repeated premature answers terminate safely.
27. As an application integrator, I want a successful final answer accepted only after every criterion has been recorded by the completion judge, so that completion is authoritative and monotonic.
28. As a maintainer, I want criterion state and evidence to remain invocation-local and ephemeral, so that ReAct retains its established non-resumable Execution-mode contract.
29. As an interactive Conversation user, I want routed ReAct turns to use the same criteria and completion-judge behavior as routed CLI runs, so that entry point does not affect completion.
30. As a maintainer, I want Runtime-context Observation generation to remain separate from completion judgment, so that Observation fallback and its provider configuration retain their existing semantics.

## Implementation Decisions

- Preserve the Task Analysis contract: ReAct analyses have ordered, nonempty, distinct, independently verifiable completion criteria; analyses for other modes omit them. The deterministic router passes the analysis through the existing Execution-mode factory seam, and only ReAct consumes its criteria.
- Remove the completion-criterion status section and all `REACT_PROGRESS` instructions, parsing, and state updates from main ReAct Model requests and native tool-call responses. ReAct without criteria retains its existing free-form terminal behavior.
- For criterion-enabled ReAct, a no-tool response must use the existing strict `ANSWER` Line Protocol block. It is accepted only if the locally held criterion set is already complete; otherwise the host supplies generic continuation feedback and proceeds to the next budgeted ReAct request.
- After each Tool-call batch settles, create one separate, tool-free `completion_judge` request. Reuse the ReAct component provider/model configuration but issue and trace it as a distinct component; do not add a provider configuration field and do not use the Runtime-context component.
- The judge context is a new request containing its judgment instructions, the complete current ReAct reasoning context, the ReAct response that produced the batch, all successful calls and rendered Raw tool results in request order, and a final criterion-progress section. That section lists every criterion by stable one-based number and whether it is completed or pending.
- The judge receives successful entries only. In a mixed batch, failed results are omitted from judge evidence and are added as error ToolMessages to the next ReAct context. A batch with no successful results makes no judge request.
- Define registered `COMPLETION_PROGRESS` and nested `COMPLETED_CRITERION` Line Protocol blocks. The outer block may be empty to report no newly completed criteria. Each nested item has exactly `NUMBER` and a concise nonempty `EVIDENCE`; numbers must be distinct, in range, and not already recorded.
- The host holds an invocation-local monotonic mapping from criterion number to evidence. Only a locally valid judge result can add entries. Neither ReAct prose, failed tool results, Runtime-context Observations, nor final-answer prose can update it.
- A malformed or invalid judge result triggers exactly one tool-free repair request. The repair request retains the original judge context and adds the rejected output plus local validation error. If the repair is still invalid, retain no new progress and continue the next ReAct round rather than failing the execution.
- Completion-judge requests do not increment the ReAct tool-round budget, but they are included in model-use, trace, and cost accounting under `completion_judge`.
- Preserve concurrent Tool-call batch execution, Runtime-context evidence selection, asynchronous Observation generation, routing, Conversation persistence, Plan–execute, recovery, and MCP authority boundaries. The judge is synchronous with post-batch completion-state update but does not replace or alter Observation work.
- Amend the affected ReAct, Line Protocol, layered-context, Observation, routing, and execution-mode documentation to remove the superseded self-reporting decision and distinguish completion judgment from Runtime-context processing.

## Testing Decisions

- The primary seam is a public Task Router invocation using controlled Task Analyzer output, a controlled ReAct model sequence, and a controlled Tool runtime. Assert externally observable request contexts, completion-judge evidence and protocol handling, terminal Execution answers, and routed criteria propagation without live providers or MCP servers.
- Use the existing controlled ReAct model/runtime tests as focused behavior coverage. Test that main ReAct requests have no criteria, that one settled successful batch produces one judge request with the full ReAct context and criterion state, and that parsed evidence updates the next judge context rather than ReAct context.
- Cover mixed successful/failed batches, all-failed batches, and failed-tool correction messages. Verify that failures do not enter judge evidence and remain visible to the next ReAct request.
- Add Line Protocol tests for empty progress, valid number/evidence pairs, duplicate numbers, already-completed numbers, out-of-range values, missing or blank evidence, undeclared fields, and invalid nested blocks.
- Cover the one-retry judge policy: an invalid first response is reissued with validation feedback; a second invalid response preserves progress and resumes ReAct rather than failing the execution.
- Cover terminal behavior: a completed judge state permits a strict `ANSWER`; pending state rejects a no-tool answer, emits no criterion data to ReAct, consumes budget, and eventually yields a failed Execution answer at exhaustion.
- Keep the focused routed Conversation-turn test to prove criteria reach its ReAct runner. It should assert the same public criterion-enabled behavior without testing database persistence internals.
- Tests assert requests, protocol-visible output, Execution answers, and routed outcomes, rather than private collections or incidental helper ordering. Existing Task Analyzer, Task Router, ReAct, Runtime-context, Conversation, trace, and Line Protocol tests are prior art.

## Out of Scope

- Changing deterministic routing or allowing Task Analyzer to choose an Execution mode.
- Adding completion criteria or completion judgment to direct, tool-agent, or Plan–execute mode.
- Persisting, checkpointing, resuming, or exposing ReAct criterion progress or evidence as durable Agent or Conversation state.
- Changing Plan, Plan revisions, Plan steps, Plan-step Executor receipts, replanning, recovery, or MCP tool authority.
- Changing concurrent Tool-call batch execution or Runtime-context Observation policy.
- Giving the completion judge an independently configurable provider or using the Runtime-context component as the judge.
- Exposing full model reasoning as progress evidence; judge evidence is a concise, checkable summary only.

## Further Notes

This replaces the earlier self-reported `REACT_PROGRESS` design. Criterion progress is intentionally absent from the main ReAct context: ReAct selects and corrects actions, while the completion judge evaluates completed tool work. The judge's criterion-progress context is the only model context that contains criterion text or status.
