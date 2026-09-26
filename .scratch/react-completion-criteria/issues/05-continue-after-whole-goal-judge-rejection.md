# 05: Continue ReAct after a whole-Goal judge rejection

**What to build:** When a ReAct invocation without completion criteria proposes `completed` but the Completion judge finds the current Goal unproven, ReAct receives the validated structured verdict and remaining gap, can take corrective action, and can submit a new candidate answer within its remaining rounds.

**Blocked by:** 04: Judge a ReAct terminal proposal without criteria.

**Status:** ready-for-agent

- [x] A negative `GOAL_JUDGMENT` requires a nonempty gap, is locally validated and canonically rendered into the next ReAct context, and does not return the rejected candidate answer.
- [x] Only the latest validated judge verdict remains in subsequent ReAct requests; the judge sees the current candidate answer and the existing policy-selected cumulative evidence, without restoring Raw results outside the Runtime-context window.
- [x] An invalid judge response gets exactly one tool-free repair request with validation feedback; a second invalid response continues ReAct without accepting completion, while a judge provider failure fails execution.
- [x] Controlled ReAct tests prove rejection, corrective Tool work, later approval, evidence-window behavior, one repair, provider failure, and final-round exhaustion.
