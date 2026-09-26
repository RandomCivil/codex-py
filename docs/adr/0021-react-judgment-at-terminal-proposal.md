# Judge ReAct completion only at a terminal proposal

**Status: accepted.** This supersedes the ReAct completion portions of ADR-0007 and ADR-0017; the Plan-step Executor decision in ADR-0020 remains unchanged.

Every ReAct invocation makes a locally validated, no-tool `REACT_DECISION` with status `completed`, `failed`, or `need_tool`. Only `completed` invokes a separate, tool-free Completion judge. ReAct supplies the candidate answer; a positive judge verdict releases that answer, while a negative verdict returns its validated structured judgment, evidence, and remaining gaps to the next ReAct round. Tool batches do not invoke the judge. A response that declares a decision while requesting Tools executes none of them and receives protocol-error feedback. This places the independent check at ReAct's claimed completion boundary and avoids repeated judge calls during ordinary Tool work.

With Task Analyzer criteria, the judge evaluates every criterion against the **current** Runtime-context evidence snapshot; without criteria, it evaluates the whole Goal. The host retains historical confirmed evidence within the invocation, but an earlier success cannot authorize completion after later work invalidates it. ReAct receives the ordered criteria together with host-owned `completed`/`pending` status on every tool-loop request, as well as numbered verdicts and gaps after a rejected terminal proposal. Judge requests and one malformed-output repair are separate from the bounded ReAct round budget. The existing Tool evidence policy and the Plan-step Executor's judgment and handoff contract remain intact.

The alternative was to continue judging after each successful Tool batch and let the judge supply the answer. That spends judge requests before ReAct proposes completion, cannot cover a zero-Tool ReAct invocation consistently, and combines verification with answer authorship.

The implementation contract is in [the ReAct completion spec](../../.scratch/react-completion-criteria/spec.md).
