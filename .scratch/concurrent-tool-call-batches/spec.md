# Concurrent Tool-Call Batches

Status: ready-for-agent

## Problem Statement

When a Tool-calling model emits several function calls in one response, the Executor needs a clear, testable contract for executing the independent calls concurrently. It must not continue the Step execution until every requested call has completed, and the model must receive unambiguous, request-correlated results even when calls finish in a different order.

## Solution

Treat all function calls in one Tool-calling model response as one Tool-call batch. Within a single Step execution, start the entire batch concurrently; wait for every call to settle; then return ToolMessages in original request order, retaining each tool-call ID, before the next model invocation or completion processing. Continue to surface individual tool failures as error context after sibling calls settle. Propagate cancellation to unfinished calls and prevent later model work.

## User Stories

1. As an Agent developer, I want all independent function calls in one model response to begin concurrently, so that one Step execution does not serialize work the model declared independent.
2. As an Agent developer, I want an Executor to wait for the complete Tool-call batch before invoking the model again, so that later reasoning can use every requested result.
3. As a Tool-calling model, I want every ToolMessage correlated to my original tool-call ID, so that I can reliably associate results with requests.
4. As a Tool-calling model, I want ToolMessages returned in my request order rather than completion order, so that concurrent timing does not make the conversation nondeterministic.
5. As a tool owner, I want a failed call in a batch represented as that call's error result, so that the model can select a corrective action without losing sibling results.
6. As a tool owner, I want sibling calls to continue settling after an ordinary per-call failure, so that successfully completed work remains available to the model.
7. As an Executor consumer, I want completion processing to start only after Tool-call batch results are available, so that a Step cannot be completed from a partial batch.
8. As an application owner, I want cancellation to stop unfinished calls and block later model work, so that cancellation does not become a normal error-driven continuation.
9. As a model author, I want calls with ordering or data dependencies placed in later model responses, so that I do not accidentally rely on an ordering guarantee within a Tool-call batch.
10. As an application owner, I want Plan steps to remain serial even when a Tool-call batch is concurrent, so that Plan ordering and its durable execution history remain unchanged.
11. As a maintainer, I want the existing ToolNode execution boundary retained, so that host authority, tool schemas, error handling, and graph behavior remain centralized.
12. As a test author, I want a black-box test that requires two tools to overlap before either can finish, so that a future serial implementation cannot satisfy the contract accidentally.
13. As a test author, I want the test to observe the next model invocation only after both tool results exist, so that the batch barrier is verified at the Executor boundary.
14. As a test author, I want the test to verify request-order ToolMessages and IDs despite concurrent completion, so that the model conversation remains stable.

## Implementation Decisions

- Use the glossary term Tool-call batch for all function calls emitted in a single Tool-calling model response.
- Preserve the existing `model → ToolNode → model` Executor boundary. ToolNode supplies concurrent invocation and waits for the whole batch; the Executor must not introduce a second scheduler around it.
- A Tool-call batch is limited to one Step execution. It does not change the strictly serial Plan established by ADR-0001.
- Results are returned to the model in the source response's request order and retain their original tool-call IDs, regardless of tool completion order.
- Ordinary per-call failures remain error ToolMessages. The Executor waits for sibling calls before returning that batch to the model.
- Caller cancellation propagates to unfinished batch work and prevents a later model invocation or completion receipt request.
- Calls in the same batch have no execution-order guarantee. A model must emit dependent operations in separate responses.
- ADR-0005 records this externally observable concurrency and side-effect boundary.

## Testing Decisions

- Test at the existing injected Tool-calling-model and MCP-tool seam in the Executor test suite; do not test ToolNode internals, private graph state, or its use of a particular async primitive.
- Use two controlled asynchronous tools which cannot finish until both have started. A serial implementation will time out, while a concurrent batch completes.
- Observe that the model's next invocation occurs only after both controlled tools have completed.
- Verify the ToolMessages observed by that next model invocation retain original request order and tool-call IDs, even if their completion timing differs.
- Preserve existing tests for individual tool errors and nonzero Atom `exec` results as prior art for error-as-model-context behavior.

## Out of Scope

- Parallel, dependency-graph, or multi-step Plan execution.
- A configurable concurrency limit, batching across model responses, or changing the maximum model/tool-round budget.
- Ordering guarantees, locking, transactions, compensation, or automatic dependency detection among calls in one Tool-call batch.
- Changes to MCP tool authority, tool schemas, host configuration, retries, recovery, persistence, or completion-receipt validation.
- Executor concurrency across separate `execute` invocations.

## Further Notes

The current ToolNode dependency already implements the required concurrent batch behavior. This work makes it an explicit project contract through ADR-0005 and high-level regression coverage, rather than duplicating tool scheduling in the Executor. The terminology follows `CONTEXT.md`; the boundary is consistent with ADR-0002 and does not alter ADR-0001's serial Plan execution.
