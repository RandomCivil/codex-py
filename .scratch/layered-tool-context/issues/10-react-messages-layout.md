# 10: ReAct messages layout

**What to build:** Let ReAct consume the `messages` Runtime-context layout for
its operational requests and Completion-judge requests. Each request preserves
the agreed role order, puts fixed system instructions first, retains the
original structured user input, and places repair or completion-rejection
feedback after Observations and before Goal.

**Blocked by:** 09: Runtime-context messages rendering and budget.

**Status:** ready-for-agent

- [ ] A ReAct tool-loop request with the selected `messages` layout orders
  system instructions, original user input, retained Raw AI/Tool pairs,
  Durable State, Observations, feedback, Goal, and criteria exactly as specified.
- [ ] ReAct completion judgment receives the same selected layout and applicable
  acceptance criteria without changing its validation or terminal behavior.
- [ ] Protocol repair and the latest validated negative completion judgment are
  represented after Observations and before Goal.
- [ ] The existing `grouped` ReAct request remains unchanged when selected.

See [the feature specification](../spec.md) for the full contract.
