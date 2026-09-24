# Plan-step completion judgment

Status: ready-for-agent

## Problem Statement

The Plan-step Executor currently treats a nonempty model response without native Tool calls as completion of the selected Plan step. Its prompt includes the step's completion criterion, but the host does not independently assess execution evidence against that criterion. This lets the Tool-calling model both select actions and declare its own work complete. The ReAct completion-criterion judgment design establishes a separate, locally validated judge over completed Tool work; the Plan-step Executor needs an analogous boundary while retaining serial Plan execution and durable recovery.

## Proposed Solution

For each Step execution attempt, the Executor owns one pending/completed state for the selected Plan step's single completion criterion. After each settled Tool-call batch containing a successful call, it issues one tool-free `completion_judge` request. A no-tool response while the criterion is pending also invokes the judge, including when no Tool has ever been called. The judge considers the current Step's accumulated execution evidence, with the newest successful batch's Raw results explicit, and returns a locally validated judgment with concise evidence if the criterion is newly proven. Only the host updates completion state.

The next operational Executor request receives the selected criterion and current `pending` or `completed` state. Once the judge records completion, the Executor asks the Tool-calling model for a final, nonempty Step handoff in a tool-free request. The completed Step execution durably records both the handoff and the judge's concise evidence. The Agent still completes the whole Plan only after every Plan step in its current revision has a completed Step execution.

This deliberately differs from ReAct's operational context: ReAct hides criterion state from its Tool-calling model, while the Plan-step Executor exposes its one selected criterion's status. It also differs from ReAct's terminal judgment: the Plan-step judge does not write the final handoff.

## Behavioral Contract

1. The judge assesses only the selected Plan step's criterion. It cannot mark another Plan step complete or change the Plan, its revision, or the Agent's step order.
2. After a settled batch, the Executor makes one judge request only when every Tool call succeeded. If any call failed, the batch skips judgment and returns failures as correction messages to the operational model; successful results from that batch do not establish completion.
3. The judge receives the selected Plan step and criterion, current Agent state and Step context, the current Runtime-context snapshot of accumulated evidence, the operational request and response that produced the batch, and all successful calls with their complete rendered Raw results in request order. A no-tool judgment uses the available Agent state, Step context, and accumulated Step evidence. Failed Tool results are identified as failures and excluded from the evidentiary set.
4. The existing Runtime-context budget applies. Required evidence is never silently truncated; an inability to fit it fails the Step. Runtime-context Observations support interpretation but cannot update criterion status themselves.
5. The Plan-step judge returns a `STEP_COMPLETION_PROGRESS` Line Protocol block. It uses ReAct's numbered `COMPLETED_CRITERION` evidence and `ALL_COMPLETED` validation semantics with a single criterion numbered `1`, but omits ReAct's `ANSWER`. Pending judgment reports no newly completed criterion and `ALL_COMPLETED=false`. Completion reports criterion `1`, nonempty directly checkable `EVIDENCE`, and `ALL_COMPLETED=true`. Duplicate, out-of-range, already completed, missing-evidence, malformed, and inconsistent verdicts are locally rejected.
6. An invalid judgment receives exactly one tool-free repair request containing the original judge context, rejected output, and validation error. A second invalid response records no progress and returns to the budgeted operational loop. A judge provider failure fails the Step.
7. If a no-tool operational response arrives while the criterion is pending, the Executor asks the judge. A pending verdict rejects that response and sends the selected criterion's pending state plus a continuation instruction to the operational model. This consumes a Tool-round budget unit. If the verdict completes the criterion, the earlier no-tool text is discarded and a new final handoff request is made.
8. Once the judge records completion, no later Tool call may execute for this Step. The Executor makes a tool-free handoff request with the completed criterion state. A nonempty text response without native Tool calls completes the Step; it does not require a second semantic judgment. An empty handoff or attempted Tool call gets one tool-free repair request. If repair is still invalid, the Step fails.
9. Judge requests and handoff repairs reuse the Executor's configured provider/model. Judge requests are traced and metered under a distinct `completion_judge` component and do not consume the operational Tool-round budget. The existing positive Tool-round budget still bounds tool selection and rejected premature no-tool responses.
10. Judge progress is local to a Step execution attempt. An interrupted Step starts a fresh attempt and rechecks external state; it does not trust the prior attempt's unconfirmed judge or Tool transcript. A completed Step retains its final judgment evidence and is skipped on recovery.
11. A completed `StepExecution` records the final judge evidence separately from its handoff result. The evidence is persisted with the completed Step and its recovery record, and remains attributable to the Plan revision and step ID. Pending progress is not persisted as completed work. Existing records must remain readable after the persistence schema change.
12. Invalid handoff, provider error, context-budget failure, and round-budget exhaustion fail the Step. The Agent's existing failed-Step replanning and three-revision limit continue to govern the whole Plan outcome.

## Acceptance Scenarios

- A Tool result proves the selected criterion: judge records evidence; the next model request contains the completed state, cannot execute Tools, and returns the Step handoff.
- Several successful batches jointly prove one criterion: the judge sees accumulated available evidence and the newest Raw results before completing it.
- A mixed batch invokes no judge; successful results from that batch do not establish completion, and failed calls remain operational correction context.
- A Step already satisfied by Agent state or Step context completes through a no-tool judgment, followed by a separate handoff request.
- A premature no-tool response with a pending judgment resumes operational work and consumes budget; repeated premature responses eventually fail the Step.
- Malformed judgment receives one repair attempt; repeated invalid output leaves the criterion pending. A judge provider failure fails the Step.
- A Tool call attempted after completion is not executed. One invalid handoff repair is allowed; a second invalid handoff fails the Step.
- An interrupted Step loses attempt-local progress, while a completed Step retains its persisted evidence and is not re-executed after recovery.

## Scope and Document Impact

- This proposal extends the design in [ReAct completion-criterion judgment](../react-completion-criteria/spec.md) but does not change ReAct's behavior or Task Analyzer's ReAct-only completion-criteria contract.
- In Plan-step execution, any failed call suppresses the batch's completion-judge request and all Observation requests for that batch, including requests for successful observation-class calls. ReAct retains its per-call policy.
- It amends ADR-0002's rule that a nonempty no-tool response alone completes a Plan step, and requires updates to the Executor and recovery boundary descriptions in `CONTEXT.md`.
- It does not change Plan-step cardinality, serial Agent orchestration, Planner ownership of the Plan, or the Agent's replanning budget.

## Confirmation

Interview decisions Q1–Q17 are recorded in [design notes](../react-completion-criteria/plan-execute-design-notes.md). The user confirmed the complete design. Implementation remains separate from this specification.
