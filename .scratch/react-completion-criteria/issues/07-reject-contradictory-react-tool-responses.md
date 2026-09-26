# 07: Reject contradictory ReAct Tool responses

**What to build:** ReAct continues to execute native Tool-call batches that carry ordinary reasoning text, but it never executes a Tool call from a response that also attempts to declare a `REACT_DECISION` status. Instead it reports the contradiction to ReAct and permits correction within the existing round budget.

**Blocked by:** 04: Judge a ReAct terminal proposal without criteria.

**Status:** ready-for-agent

- [ ] Native Tool calls accompanied by ordinary text execute as one concurrent, request-ordered batch; the text is treated as reasoning description and does not change completion state.
- [ ] A Tool-call response containing any `REACT_DECISION` status, including a malformed attempted block, executes none of its calls, records no Tool-result evidence or Observation for them, and receives a protocol-error continuation.
- [ ] Contradictory responses consume ReAct rounds; a corrected later Tool response can proceed, while repeated contradictions fail at the configured round limit.
- [ ] Controlled public ReAct tests prove the lack of Tool side effects for rejected responses, preserved Tool-call behavior for ordinary text, no Completion judge request from either Tool-call path, and correct feedback in the next model context.
