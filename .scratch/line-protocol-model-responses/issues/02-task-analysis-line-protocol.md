# 02: Task-analysis Line Protocol slice

**What to build:** Task analysis obtains and locally validates `TASK_ANALYSIS` Line Protocol output, so deterministic routing continues to receive the same trustworthy analysis contract without provider JSON modes.

**Blocked by:** 01: Line Protocol core and Planner slice.

**Status:** ready-for-agent

- [ ] The task-analysis codec explicitly maps every analysis field and rejects unexpected, missing, invalid, or semantically incompatible values.
- [ ] The model request prompts for `TASK_ANALYSIS` and sends no provider-specific structured-output format.
- [ ] A valid protocol response produces the existing analysis result and routing behavior; invalid output fails without repair or retry.
- [ ] Tests cover valid routing-relevant analysis, protocol rejection, and the absence of provider formatting options.
