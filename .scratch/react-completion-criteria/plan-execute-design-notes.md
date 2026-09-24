# Plan–execute completion judgment: design discussion

Status: confirmed

## Confirmed decisions

- The Plan-step Executor's judge evaluates only the currently selected Plan step's one completion criterion. It does not mark other Plan steps complete.
- After a valid completed judgment, the Tool-calling model produces the final Step handoff. No further Tool call may execute for that Step.
- A Step that needs no Tool calls still receives a judge judgment before it can complete, using the available Agent state and Step context as evidence.
- A judgment may use evidence accumulated during the current Step execution, with the newest successful Tool-call batch made explicit. Failed Tool results are correction context, not completion evidence.
- An invalid judge output receives one repair request with validation feedback. If the repair fails, the Executor records no progress and continues within its existing Tool-round budget. Exhaustion fails the Step and leaves replanning to the Agent.
- The judge uses ReAct's completion-evidence output convention. The next Tool-calling-model request receives the current Plan step's criterion and its `pending` or `completed` status, but not the judge's freeform reasoning. This intentionally differs from ReAct mode, which hides criterion state from its Tool-calling model.
- Judgment progress belongs to one Step execution attempt. An interrupted Step starts a fresh attempt that rechecks external state; a formally completed Step remains complete and is skipped.
- Judge requests reuse the Executor's provider/model configuration, have no Tool binding, and are traced and metered separately as `completion_judge`.
- The Plan-step judge uses ReAct's numbered evidence and `ALL_COMPLETED` validation conventions, but a Plan-step protocol variant omits `ANSWER`. Only the Tool-calling model writes the Step handoff.
- A no-tool response while the judge still reports `pending` is rejected. The Executor returns the current criterion's pending status and a continuation instruction to the Tool-calling model; this consumes the existing Tool-round budget.
- Once the judge records completion, any later Tool call is intercepted without execution. An empty or tool-calling handoff gets one tool-free repair request. A second invalid handoff fails the Step.
- The judge sees the current Runtime-context snapshot of accumulated Step evidence plus the newest successful batch's Raw results. Required evidence is never silently truncated; if it cannot fit the configured context budget, the Step fails explicitly.
- A judge provider failure fails the Step and leaves replanning to the Agent. An all-failed Tool-call batch skips judgment and sends errors back to the operational model.
- The final handoff needs no second semantic judgment; only the earlier valid judge judgment establishes completion. The handoff must be nonempty and contain no Tool call.
- A completed Step persists its final concise judge evidence with its handoff and recovery record. Incomplete, attempt-local progress is not carried across interruption.

## Final review

All interview branches are answered, and the user confirmed the complete design. The implementation contract is in [the Plan-step completion judgment spec](../plan-execute-completion-judgment/spec.md). ADR-0020 records the accepted decision; code implementation remains pending.
