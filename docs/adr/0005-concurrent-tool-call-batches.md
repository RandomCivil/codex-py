# Concurrent tool-call batches

**Status: accepted.** All function calls in one response from the Tool-calling model form a Tool-call batch. The Executor starts every call in that batch concurrently and waits for every call to settle before the next model invocation or completion processing. Tool results retain the response's request order and their originating tool-call IDs, independent of completion order. A per-call failure is returned as that call's error result after sibling calls settle; it does not abandon the rest of the batch. Cancellation propagates to unfinished calls and prevents subsequent model work.

This boundary is within a single Step execution. It does not permit parallel or dependency-graph execution of Plan steps. A model must put dependent calls in separate responses; calls in a single batch have no execution-order guarantee, including when they affect the same external resource.

## Considered Options

- Add Executor-owned parallel scheduling around `ToolNode` — rejected because `ToolNode` already supplies concurrent, ordered tool-call processing and its established per-call error and interruption behavior.
- Execute calls serially in model-request order — rejected because independent calls would unnecessarily lengthen a bounded Step execution.
- Permit concurrent Plan steps — rejected because the Plan remains strictly serial under ADR-0001.
