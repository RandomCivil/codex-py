# 02: ReAct judges completion from settled tool evidence

**What to build:** A criterion-enabled ReAct execution selects tools without seeing or reporting completion criteria. After each settled Tool-call batch, a separate completion-judge request evaluates successful Raw tool results against the ordered criteria and returns concise evidence for newly completed outcomes. Failed results remain correction messages for the next ReAct request, and only judge-verified progress can permit a final answer.

**Blocked by:** 01: Task Analysis supplies ReAct completion criteria.

**Status:** ready-for-agent

- [ ] Main ReAct Model requests and tool-call text contain no criteria, criterion status, or self-reported progress; a criterion-enabled terminal response uses strict `ANSWER` and succeeds only when judge state is complete.
- [ ] Every settled batch with successful calls creates one tool-free, traced `completion_judge` request using the ReAct provider, full current ReAct reasoning context, successful Raw results, and the full criterion-progress context; all-failed batches do not invoke it.
- [ ] The registered completion protocol requires distinct, new in-range criterion numbers and concise evidence; valid judgments update ephemeral monotonic progress, and invalid output is repaired once before ReAct continues with unchanged progress.
- [ ] Mixed batches send only successful evidence to the judge and preserve failed results as next-round ReAct correction messages; judge work is metered but does not consume the tool-round budget.
- [ ] Controlled public ReAct and router tests cover context separation, evidence updates, empty progress, mixed failures, repair/retry, rejected premature answers, budget exhaustion, trace attribution, and final success.
