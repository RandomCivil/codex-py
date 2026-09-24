# Judge Plan-step completion from execution evidence

**Status: accepted.** The Plan-step Executor uses a separate, tool-free completion judge to evaluate the selected step's single completion criterion against available execution evidence. The host records completion only from a locally validated judgment, then obtains a separate nonempty handoff from the Tool-calling model. This separates action selection from completion authority while preserving the Plan's serial step boundary. The operational model sees its selected criterion's current pending/completed state; the judgment's concise evidence is stored with a completed Step execution for durable audit and recovery. An interrupted attempt loses provisional judge progress and rechecks external state in a fresh attempt.

For a Tool-call batch, any failed call suppresses the batch's completion-judge request and every Observation request associated with that batch, including successful observation-class calls. The full batch remains Raw evidence for the correction loop. ReAct keeps its existing per-call behavior.

This decision amends ADR-0002, which permitted a nonempty no-tool response to complete a Step without separate judgment. It adapts ReAct's completion-judge protocol: the Plan-step judge reports criterion evidence and completion state but supplies no final `ANSWER`, because the Executor model produces the Step handoff. The implementation contract is in [the Plan-step completion judgment spec](../../.scratch/plan-execute-completion-judgment/spec.md).

## Considered Options

- Keep the current no-tool text rule — simpler, but the same Tool-calling model chooses actions and declares its own criterion complete.
- Let the judge write the Step handoff — one fewer model request, but it combines evidence judgment with the durable handoff and conflicts with the chosen Executor ownership of that explanation.
- Persist provisional judgment across an interruption — less repeated work, but recovery could trust unconfirmed evidence from a partially executed Tool transcript.
