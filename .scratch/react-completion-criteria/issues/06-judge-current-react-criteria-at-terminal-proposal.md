# 06: Judge current ReAct criteria at a terminal proposal

**What to build:** A ReAct invocation with ordered completion criteria waits until ReAct proposes no-tool `completed`, then judges every criterion's current truth. It retains prior confirmed evidence for later judgments, but releases ReAct's candidate answer only when the latest verdict verifies every criterion now. A rejected verdict gives ReAct numbered evidence and gaps without revealing criteria text.

**Blocked by:** 04: Judge a ReAct terminal proposal without criteria; 05: Continue ReAct after a whole-Goal judge rejection.

**Status:** ready-for-agent

- [ ] Successful and mixed Tool-call batches keep their existing Tool execution and Runtime-context behavior but make no Completion judge request; only a no-tool `completed` proposal starts criterion judgment.
- [ ] The `COMPLETION_PROGRESS` judgment contains exactly one ordered `CRITERION_VERDICT` for each criterion, with current `VERIFIED` state and nonempty evidence or gap; local validation rejects missing, duplicate, out-of-range, unordered, malformed, and `ALL_COMPLETED`-inconsistent verdicts.
- [ ] Historical confirmed criterion evidence remains invocation-local and reaches later judge requests. A later Tool action that invalidates an earlier success causes a negative current verdict, so historical evidence alone cannot authorize completion.
- [ ] The next ReAct request receives only the latest validated structured verdict, numbered evidence, and gaps, without criteria text; the final positive verdict returns ReAct's candidate answer rather than a judge-authored answer.
- [ ] Controlled routed CLI and Conversation tests prove criteria propagation, current-state revalidation, feedback visibility, zero-Tool proposals, repair/failure behavior, and unchanged non-ReAct mode contracts.
