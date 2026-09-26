# ReAct completion judgment at a proposed terminal response

Status: ready-for-agent

## Problem Statement

The current ReAct loop calls a Completion judge after each successful Tool-call batch. Judgment therefore depends on batch count rather than on ReAct's decision that its work is ready for final review. The judge can also supply the final answer, obscuring ownership of the candidate answer. ReAct invocations without completion criteria bypass this judgment entirely.

## Solution

Every ReAct invocation uses a strict `REACT_DECISION` Line Protocol block when its operational model requests no native Tool calls. Its `STATUS` is `completed`, `failed`, or `need_tool`. Only a no-tool `completed` response invokes a separate, tool-free Completion judge. ReAct supplies the candidate `ANSWER`; a valid positive judgment returns that answer. A negative judgment sends its validated structured verdict, evidence, and remaining gaps into the next ReAct context within the existing round budget.

Native Tool-call responses execute their batches without an immediate Completion judge. Accompanying ordinary text is reasoning description. A response that both requests Tool calls and declares a `REACT_DECISION` status is contradictory: the host executes none of those calls, reports the protocol error to ReAct, and uses the next budgeted round.

With criteria, the judge evaluates every ordered criterion against the current evidence window. The host retains historical confirmed evidence for the invocation, but a final answer is accepted only when the latest judgment verifies every criterion **now**. Without criteria, the judge evaluates the whole Goal. Runtime-context Observations are evidence, never Completion judgments.

## User Stories

1. As an application integrator, I want an explicit decision on every no-tool ReAct response, so an idle or malformed response cannot silently complete work.
2. As a ReAct model, I want to propose a candidate answer only when I believe the Goal is complete, so Judge calls occur at the completion boundary.
3. As an application owner, I want every ReAct invocation judged, including one without Task Analyzer criteria or Tool calls.
4. As a Tool host, I want contradictory status-and-Tool responses rejected before execution, so a completion claim cannot race with side effects.
5. As a ReAct model, I want ordinary text accompanying Tool calls to remain usable as reasoning description.
6. As a Completion judge, I want the candidate answer, Goal, current reasoning context, and policy-selected cumulative Tool evidence.
7. As a ReAct model, I want the judge's validated structured verdict, evidence, and remaining gaps after a rejection.
8. As an application integrator, I want historical criterion evidence retained while requiring a fresh current-state check, so later actions cannot rely on stale success.
9. As a maintainer, I want ReAct to receive ordered criteria with host-owned status, plus numbered verdicts and gaps after a rejected terminal proposal.
10. As an operator, I want judge requests traced and metered separately without consuming the ReAct round budget.
11. As a Conversation user, I want routed ReAct turns to follow the same judgment contract as CLI invocations.

## Behavioral Contract

1. A no-tool ReAct response must be one `REACT_DECISION` block. `STATUS="completed"` requires a nonempty `ANSWER` and no `ERROR`; `STATUS="failed"` requires a nonempty `ERROR` and no `ANSWER`; `STATUS="need_tool"` permits neither. Unknown status, missing or extra fields, empty required values, arbitrary prose, and incomplete blocks are invalid.
2. No-tool `completed` makes exactly one initial Completion judge request. It is tool-free, uses ReAct's configured provider/model, and is traced as `completion_judge`. A valid positive judgment returns ReAct's candidate `ANSWER` unchanged. The judge never writes the user-facing answer.
3. No-tool `failed` returns a failed Execution answer with the supplied safe error. No-tool `need_tool` adds feedback asking ReAct to choose a Tool or explicitly fail, then continues within the round budget. Neither calls the judge.
4. A native Tool-call response with ordinary accompanying text executes its calls as one concurrent, request-ordered batch; the text is reasoning description, not a completion claim. A response containing Tool calls and a `REACT_DECISION` block, including a malformed attempted block, executes no calls and adds a protocol error to the next ReAct context. This consumes a round.
5. An invalid no-tool decision adds the specific local validation error to ReAct context and consumes a round. Repeated `need_tool`, contradictory, invalid, and judge-rejected responses cannot exceed the configured round budget. Judge requests and judge repair do not consume it.
6. The judge sees the current operational request and response, Goal, candidate answer, current Runtime-context evidence snapshot, and host-held prior validated judgment state. The snapshot follows existing Raw/Observation retention and token-budget policy; the judge does not resurrect omitted Raw results. Failed Tool results may explain a gap but cannot alone prove completion. A zero-Tool invocation is valid: an explanatory answer may be judged from Goal and answer, whereas an unverified external effect must be rejected.
7. With criteria, the judge receives all ordered criteria and historical confirmed evidence. Its `COMPLETION_PROGRESS` verdict has `ALL_COMPLETED` and exactly one numbered `CRITERION_VERDICT` block per criterion. Each block contains `NUMBER` and either `VERIFIED=true` with nonempty `EVIDENCE`, or `VERIFIED=false` with nonempty `GAP`. Numbers are unique, ordered, one-based, and cover the criterion set. `ALL_COMPLETED` equals the conjunction of current `VERIFIED` values. Historical confirmed evidence is retained even if the current verdict finds that criterion false.
8. Without criteria, the judge returns `GOAL_JUDGMENT` with `COMPLETED=true` and nonempty `EVIDENCE`, or `COMPLETED=false` and nonempty `GAP`. It judges the whole Goal and candidate answer, not an invented empty criterion list.
9. Every ReAct tool-loop request contains the ordered criteria with host-owned `completed` or `pending` status. A negative judgment additionally sends the **validated, canonical structured verdict** into the next ReAct context: numbers, current states, evidence and gaps, or the whole-Goal verdict. ReAct does not receive hidden judge reasoning. Only the latest judge verdict remains in subsequent ReAct requests; the host retains historical confirmed evidence for later judge requests. Exhaustion after a rejected claim fails the invocation.
10. An invalid judge output gets exactly one tool-free repair request with rejected output and local validation error. If repair is still invalid, no completion is accepted and ReAct gets validation-failure feedback within its remaining budget. A judge provider failure fails the invocation.
11. Historical progress, verdicts, and feedback are invocation-local and ephemeral. Routing, Conversation persistence, concurrent Tool batches, Tool authority, and asynchronous Observation generation retain their existing boundaries. The Plan-step Executor's separate judgment and handoff contract is unchanged.

## Protocol Examples

ReAct proposes completion without Tool calls:

```text
BEGIN REACT_DECISION
STATUS="completed"
ANSWER="The requested file is ready."
END REACT_DECISION
```

The criterion-enabled judge rejects because current evidence does not prove the file still exists:

```text
BEGIN COMPLETION_PROGRESS
ALL_COMPLETED=false
BEGIN CRITERION_VERDICT
NUMBER=1
VERIFIED=false
GAP="Current file existence has not been verified."
END CRITERION_VERDICT
END COMPLETION_PROGRESS
```

Without criteria, the corresponding judge response uses `GOAL_JUDGMENT`:

```text
BEGIN GOAL_JUDGMENT
COMPLETED=false
GAP="No evidence shows that the requested file was created."
END GOAL_JUDGMENT
```

## Testing Decisions

- Use controlled ReAct model, Tool runtime, and judge responses through public ReAct and routed Task Router seams. Assert Tool effects, request contexts, judge counts, and Execution answers without live providers.
- Cover all three no-tool statuses, with and without criteria and with zero prior Tool calls. Assert only no-tool `completed` starts judgment and that a positive verdict returns the candidate answer.
- Cover Tool calls with ordinary reasoning text, Tool calls mixed with every status, mixed successful/failed batches, malformed decisions, and repeated `need_tool`; prove rejected calls have no Tool effects and continuations consume budget.
- Verify the judge receives the candidate answer and policy-selected cumulative evidence, without Raw history outside the Runtime-context window.
- Validate full per-criterion current verdicts: missing, duplicated, out-of-range, unordered, missing or blank evidence/gaps, inconsistent `ALL_COMPLETED`, and revalidation after later Tool work invalidates earlier success.
- Verify whole-Goal judgments without criteria, including a pure explanation and an unverified external side effect.
- Verify negative canonical verdicts enter ReAct context without criteria text; a later verdict replaces the earlier ReAct-facing one while host-held history reaches the judge.
- Cover judge repair, repeated invalid judgment, provider failure, final-round rejection, trace/model-use accounting, and routed Conversation parity.

## Out of Scope

- Changing Task Analyzer's criterion generation or deterministic routing.
- Adding judgment to Direct, tool-agent, or Plan–execute mode, or changing Plan-step Executor judgment.
- Persisting or resuming ReAct criterion progress, judge verdicts, or Tool transcripts.
- Changing Tool-call batch concurrency, Runtime-context Observation selection, Tool authority, or provider configuration.
- Giving the judge Tool access or authority to write the final answer.

## Further Notes

This supersedes the per-Tool-batch ReAct judge design. Historical criterion evidence records what was once verified; current completion is a fresh verdict that may be false after subsequent work. The validated verdict is correction context for ReAct, while criteria text remains confined to the judge.
