# 04: Judge a ReAct terminal proposal without criteria

**What to build:** A ReAct invocation without supplied completion criteria uses an explicit no-tool decision. A `completed` proposal returns ReAct's candidate answer only after a separate, tool-free whole-Goal judgment confirms it; `failed` and `need_tool` follow their own bounded paths without calling the judge. This works even when no Tool was ever called.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] Every no-tool ReAct response is locally validated as one `REACT_DECISION` with exactly the fields permitted by its `completed`, `failed`, or `need_tool` status; malformed, empty, unknown, and extra-field responses receive protocol-error feedback within the existing round budget.
- [ ] A no-tool `completed` proposal makes one initial, tool-free `completion_judge` request using the ReAct provider; a valid positive `GOAL_JUDGMENT` returns the candidate answer unchanged. Ordinary Tool-call batches do not invoke this judge.
- [ ] A no-tool `failed` response returns a failed Execution answer with its nonempty error; `need_tool` requests another ReAct round, and repeated invalid or `need_tool` responses fail when the budget is exhausted.
- [ ] Controlled public ReAct tests prove the zero-Tool explanatory-answer path, a Tool-backed positive path, rejected malformed decisions, judge request attribution, and absence of judge calls for `failed` and `need_tool`, without a live provider.
