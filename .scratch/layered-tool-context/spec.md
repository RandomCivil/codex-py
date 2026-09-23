# Runtime-context goal placement and file-grouped raw reads

Status: ready-for-agent

## Problem Statement

The Runtime context window currently renders a goal inside Durable State, before
Observations and Raw tool results. In a long tool loop, the evidence that the
model must act on separates it from the goal. Its Raw tool results are also
rendered as a chronological list of generic calls, forcing the model to search
across Tool rounds to find all evidence for one source file.

The application needs the goal to remain prominent at the end of every model
context without weakening Agent-state durability, and it needs permanent-raw
native read evidence to be readily usable by file while preserving the
authoritative Raw tool result contract.

## Solution

Render the goal as a dedicated final Runtime-context section. Remove it only
from the rendered Durable State payload; retain it unchanged in Agent state,
Checkpoints, planning, recovery, and all other Durable State uses.

Within Raw tool results, render only native permanent-raw `grep` and
`read_file` calls as file groups spanning the active invocation's Tool rounds.
Use a native read's explicit path when present; otherwise split standard grep
output of the form `path:matched content` into the corresponding file groups.
Keep each group's results in Tool-round and original request order. Preserve a
complete `(unfiled)` group for failed calls or calls whose file cannot be
resolved, including the native request, its complete result, and its error.
All other evidence retains its existing presentation and retention behavior.

## User Stories

1. As a tool-calling model, I want the goal at the end of every Runtime context window, so that the next action is anchored to the task after reviewing evidence.
2. As an application integrator, I want the goal removed only from the rendered Durable State section, so that Agent state still has its established durable contract.
3. As a recovery operator, I want goal persistence and Checkpoint contents unchanged, so that this presentation change cannot alter recovery behavior.
4. As a Planner, I want the goal to remain available through Agent state, so that planning behavior is unchanged.
5. As a tool-calling model, I want a distinct Goal section after Durable State, Observations, and Raw tool results, so that its location is stable regardless of the amount of evidence.
6. As a tool-calling model, I want `read_file` results grouped beneath their requested file path, so that all exact reads of a source file are easy to compare.
7. As a tool-calling model, I want `grep` matches grouped beneath the file identified by each match, so that search evidence for a file is colocated with its direct reads.
8. As a tool-calling model, I want one grep result that names multiple files split into their corresponding file groups, so that each file presents its own matching evidence.
9. As a tool-calling model, I want results for the same file from different Tool rounds collected in one group, so that revisiting a file does not require scanning chronological call blocks.
10. As a tool-calling model, I want entries within each file group to retain Tool-round and original request order, so that I can reason about the sequence of evidence.
11. As an application integrator, I want every failed native read or unresolvable file mapping shown under `(unfiled)`, so that grouping never hides authoritative evidence.
12. As a tool-calling model, I want `(unfiled)` entries to show the native tool name and arguments, complete result, and error where present, so that I can diagnose why no file group was possible.
13. As an operator, I want tool-call IDs and complete Raw tool-result data to remain available to diagnostics, so that rendering does not weaken provenance.
14. As an application owner, I want `grep` and `read_file` to remain permanent-raw evidence and never request Observations, so that this change does not make exact read evidence lossy.
15. As an application integrator, I want `list_dir`, `glob`, native reads, and failed calls to retain their existing rendering and evidence lifecycle, while successful observation-class calls use the decision-impact policy, so that native-read presentation remains lossless and observation evidence remains task-relevant.
16. As a ReAct caller, I want the same goal placement and native-read grouping on every operational and terminal model request, so that context formatting is consistent through the tool loop.
17. As a Plan–execute caller, I want the Plan-step Executor to use the same rendering behavior for its invocation-local Runtime context, so that execution modes do not diverge.
18. As an application owner, I want context-budget behavior to remain unchanged, so that required Raw evidence remains lossless and context-maintenance failure remains explicit.

## Implementation Decisions

- Amend the Runtime-context renderer while leaving the RuntimeContext data model and Tool-call evidence policy intact.
- Derive the displayed goal from applicable Durable State and omit that same goal field from the Durable State presentation. Support both ReAct's direct goal payload and the Agent-state goal nested in the Plan-step Executor's durable payload.
- Render `### Goal` last, after `### Raw tool results`; omit no other Runtime-context sections or reorder non-goal evidence.
- Treat only native tools named exactly `grep` and `read_file` as file-grouped native reads. Do not extend file grouping to `exec`, including shell commands classified as permanent raw.
- Resolve a `read_file` group from its explicit path argument. Resolve a `grep` group from an explicit single-file path argument when available; otherwise parse path-prefixed grep result lines and split their matched content by path.
- Collect file groups across all selected Raw tool results. Preserve each group's entries in ascending Tool-round order and original request order within a batch. Retain relevant round and tool-call provenance in the rendered entry.
- Place a native read into `(unfiled)` whenever it reports an error, has no safely resolvable path, or contains result content that cannot be assigned to a file. Render its native request, complete unmodified result, and error rather than discarding or summarizing it.
- Continue to render non-native-read Raw calls, Observations, and empty states under their existing contracts.
- For every successful observation-class call, provide the Runtime-context component a deterministic decision summary containing the Goal, execution mode, applicable Plan step and completion criterion, and the triggering tool name and arguments. Require `affects_current_decision` and `affected_targets` (`confirmed_facts`, `summary`, and/or `durable_state`) in its Observation. A positive Observation replaces its Raw fallback; a negative result keeps Raw evidence only through its existing window, then leaves Runtime context. Do not block either loop for this asynchronous decision.
- Retain Raw fallback for a pending, failed, or invalid decision-impact Observation. Require nonempty targets for a positive impact and no targets for a negative one; when budget merging positive Observations, retain positive impact and union their targets. Treat `durable_state` strictly as explanatory metadata, never as a state mutation.
- Preserve Raw tool-result retention, per-call classification, asynchronous Observation lifecycle, token budgeting, tracing, Checkpoint exclusion, and recovery boundaries defined by ADR-0018.

## Testing Decisions

- Use the existing public RuntimeContextPolicy assembly and message-rendering seam as the primary test boundary. Assert the prompt content supplied to a model rather than private collection layout or helper implementation.
- Extend existing Runtime-context prompt rendering tests as prior art for section placement, empty evidence, result/error rendering, and JSON-safe values.
- Add controlled native read calls across several Tool rounds and assert that the final Goal section follows every evidence section while no goal remains in the displayed Durable State payload.
- Test both direct ReAct-style Durable State and nested Plan-step Executor-style Agent state to verify the same displayed-goal contract without changing their stored values.
- Test repeated `read_file` calls for one path and verify a single file group with round/request-ordered entries.
- Test grep output containing path-prefixed matches for multiple files and verify each match appears in the appropriate file group.
- Test explicit grep paths, malformed or pathless grep output, tool errors, and read failures; verify that these retain native request, complete result, and error in `(unfiled)`.
- Test that native `grep` and `read_file` retain their permanent-raw lifecycle and do not invoke the Observation model.
- Test positive and negative decision-impact Observations for each existing observation-class tool, including `exec`; verify negative calls retain Raw evidence through their normal window before omission, invalid or failed judgments retain Raw fallback, and merged Observations union their affected targets without mutating Durable State.
- Test that non-native-read calls retain their existing rendering and that no existing context-budget, Raw-evidence, or provenance behavior regresses.
- Retain high-level ReAct and Plan-step Executor tests as confirmation that both modes pass the resulting Runtime context to later model requests.

## Out of Scope

- Changing Agent state, Plan, Checkpoint, recovery, or Planner schemas.
- Changing Raw tool-result retention, Observation generation, Tool-call batch concurrency, request settlement order, or context-budget policy.
- Grouping `exec` output or attempting to parse arbitrary shell commands or their output into files. The existing conservative static classification still decides whether an `exec` call is observation-class; this feature does not make permanent-raw read commands subject to decision-impact omission.
- Grouping `list_dir`, `glob`, write-class calls, or Observations by file.
- Adding a new diagnostic store, changing trace contents, or persisting Runtime-context evidence.
- Changing tool schemas, tool permissions, execution-mode selection, or provider configuration.

## Further Notes

This specification uses the glossary terms Agent state, Durable State, Goal,
Raw tool result, Tool round, Tool-call batch, Tool-call evidence policy,
Runtime context window, Observation, ReAct mode, and Plan-step Executor.
ADR-0018 remains authoritative for evidence lifecycle and provenance; this
feature changes only the Runtime-context presentation of durable goal and
native permanent-raw reads.
