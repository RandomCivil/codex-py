# Runtime-context goal placement and evidence presentation

Status: ready-for-agent

## Problem Statement

The Runtime context already renders Goal after evidence and groups native `grep` and
`read_file` results by file. Each resolved file result still has a per-call
wrapper with Round, tool name, arguments, and tool-call ID; native-only rounds
leave placeholder Round headings; and non-file Raw results appear before file
groups. Observations also repeat Round headings and source IDs in the
model-facing text. These wrappers make the evidence harder to scan without
adding information needed for the next decision.

The application needs a more direct model-facing presentation while retaining
the authoritative Raw tool result and Observation records for provenance,
diagnostics, and lifecycle management.

## Solution

Keep Goal as a dedicated Runtime-context section after the evidence. It remains omitted
only from the rendered Durable State payload and remains unchanged in Agent
state, Checkpoints, planning, recovery, and all other Durable State uses.
ReAct Completion criteria and their status remain in the Completion judge's
context. After a rejected terminal proposal, the latest validated judge
verdict, evidence, and gaps enter ReAct context without the criteria text.

Within Raw tool results, render only native permanent-raw `grep` and
`read_file` calls as file groups spanning the active invocation's Tool rounds.
Use a native read's explicit path when present; otherwise split standard grep
output of the form `path:matched content` into the corresponding file groups.
Show each resolved result directly beneath its file heading as a separate
first-level code block, in Tool-round and original request order. Strip the
redundant path prefix from a path-prefixed grep match when presenting it under
that file. Do not show the native call's round, tool name, arguments, or
tool-call ID in resolved file groups. Preserve a complete `(unfiled)` group
for failed calls or calls whose file cannot be resolved, including the native
request, its complete result, and its error. Remove empty Tool-round
placeholders for grouped native reads, and place all other Raw tool results
after the file groups, retaining their existing round-grouped format.

Render Observations as three direct categories—Confirmed facts, Reported
errors, and Model inferences—without per-Observation Round headings or
tool-call IDs in the model-facing text. Keep all provenance and the existing
evidence lifecycle in the internal data model and diagnostics.

For example, after a `read_file` and a path-prefixed `grep` on the same file,
followed by a non-native `exec` result, the relevant rendered sections are:

~~~text
### Observations
- Confirmed facts:
  - Login validation was updated
- Reported errors: (none)
- Model inferences: (none)

### Raw tool results
#### File: src/app.py
```text
def login(): pass
```
```text
12:login() is called here
```
#### Round 3
- exec (tool_call_id=exec-3)
  - arguments: {"command": "pytest tests/test_login.py"}
  - result: "1 passed"
~~~

## User Stories

1. As a tool-calling model, I want the goal after the evidence in every Runtime context window, so that the next action is anchored to the task after reviewing evidence.
2. As an application integrator, I want the goal removed only from the rendered Durable State section, so that Agent state still has its established durable contract.
3. As a recovery operator, I want goal persistence and Checkpoint contents unchanged, so that this presentation change cannot alter recovery behavior.
4. As a Planner, I want the goal to remain available through Agent state, so that planning behavior is unchanged.
5. As a tool-calling model, I want a distinct Goal section after Durable State, Observations, and Raw tool results, so that its location is stable regardless of the amount of evidence.
6. As a tool-calling model, I want `read_file` results grouped beneath their requested file path, so that all exact reads of a source file are easy to compare.
7. As a tool-calling model, I want `grep` matches grouped beneath the file identified by each match, so that search evidence for a file is colocated with its direct reads.
8. As a tool-calling model, I want one grep result that names multiple files split into their corresponding file groups, so that each file presents its own matching evidence.
9. As a tool-calling model, I want results for the same file from different Tool rounds collected in one group, so that revisiting a file does not require scanning chronological call blocks.
10. As a tool-calling model, I want results within each file group to retain Tool-round and original request order without per-call metadata in the displayed text, so that I can read the file evidence in sequence.
11. As an application integrator, I want every failed native read or unresolvable file mapping shown under `(unfiled)`, so that grouping never hides authoritative evidence.
12. As a tool-calling model, I want `(unfiled)` entries to show the native tool name and arguments, complete result, and error where present, so that I can diagnose why no file group was possible.
13. As an operator, I want tool-call IDs, rounds, arguments, and complete Raw tool-result data to remain available to diagnostics, so that a simpler model-facing presentation does not weaken provenance.
14. As an application owner, I want `grep` and `read_file` to remain permanent-raw evidence and never request Observations, so that this change does not make exact read evidence lossy.
15. As an application integrator, I want all Tool-call evidence lifecycles to remain unchanged, while this specification revises only the model-facing rendering of resolved native reads and Observations.
16. As a ReAct caller, I want the same goal placement and native-read grouping on every operational and terminal model request, so that context formatting is consistent through the tool loop.
17. As a Plan–execute caller, I want the Plan-step Executor to use the same rendering behavior for its invocation-local Runtime context, so that execution modes do not diverge.
18. As an application owner, I want context-budget behavior to remain unchanged, so that required Raw evidence remains lossless and context-maintenance failure remains explicit.
19. As a tool-calling model, I want resolved read results shown directly beneath each file heading, with one block per result and no call metadata, so that file contents are the first thing I see.
20. As a tool-calling model, I want Observations combined under three category headings with no displayed Round or tool-call ID, so that the evidence reads as one concise section.
21. As a tool-calling model, I want non-native Raw tool results after all file groups, retaining their own Round headings and call details, so that they remain diagnosable without interrupting file evidence.
22. As a ReAct caller, I want the latest validated Completion judge verdict, evidence, and gaps after a rejected terminal proposal, so that I can correct the shortfall without receiving criteria text.

## Implementation Decisions

- Amend the Runtime-context renderer while leaving the RuntimeContext data model and Tool-call evidence policy intact.
- Derive the displayed goal from applicable Durable State and omit that same goal field from the Durable State presentation. Support both ReAct's direct goal payload and the Agent-state goal nested in the Plan-step Executor's durable payload.
- Render `### Goal` after `### Raw tool results`; retain the other top-level Runtime-context sections. Keep ReAct Completion criteria and their status out of operational Runtime context. After judge rejection, append only the latest validated structured verdict with evidence and gaps, without criteria text.
- Treat only native tools named exactly `grep` and `read_file` as file-grouped native reads. Do not extend file grouping to `exec`, including shell commands classified as permanent raw.
- Resolve a `read_file` group from its explicit path argument. Resolve a `grep` group from an explicit single-file path argument when available; otherwise parse path-prefixed grep result lines and split their matched content by path.
- Collect file groups across all selected Raw tool results. Preserve each group's entries in ascending Tool-round order and original request order within a batch. Append each result as a separate first-level code block directly under `#### File: <path>`; use a fence that cannot collide with the result text. Show complete multi-line content without an extra `result:` label. For path-prefixed grep output, split the matched lines by file, remove each line's path prefix in the displayed block, and append each block to its corresponding file group. Retain round, tool name, arguments, tool-call ID, and complete original result internally and in diagnostics, but omit them from resolved file groups in the model-facing text.
- Place a native read into `(unfiled)` whenever it reports an error, has no safely resolvable path, or contains result content that cannot be assigned to a file. Render its native request, complete unmodified result, and error rather than discarding or summarizing it.
- Do not render empty `#### Round N` placeholders for rounds containing only grouped native reads. Render file groups first, including `(unfiled)`; then render non-native-read Raw calls under their existing `#### Round N` headings with their existing call details. Show `- (none)` under Raw tool results only when neither file groups nor other Raw calls are visible.
- Under `### Observations`, render exactly three category headings in this order: Confirmed facts, Reported errors, Model inferences. Flatten evidence from all visible Observations into each category in Observation/source order without deduplication. Show `(none)` for an empty category, including when no Observations are visible. Omit displayed Round headings and tool-call IDs; retain both in Observation data and diagnostics.
- For every successful observation-class call, provide the Runtime-context component a deterministic decision summary containing the Goal, execution mode, applicable Plan step and completion criterion, and the triggering tool name and arguments. Require `affects_current_decision` and `affected_targets` (`confirmed_facts`, `summary`, and/or `durable_state`) in its Observation. A positive Observation replaces its Raw fallback; a negative result keeps Raw evidence only through its existing window, then leaves Runtime context. Do not block either loop for this asynchronous decision.
- Retain Raw fallback for a pending, failed, or invalid decision-impact Observation. Require nonempty targets for a positive impact and no targets for a negative one; when budget merging positive Observations, retain positive impact and union their targets. Treat `durable_state` strictly as explanatory metadata, never as a state mutation.
- Preserve Raw tool-result retention, per-call classification, asynchronous Observation lifecycle, token budgeting, tracing, Checkpoint exclusion, and recovery boundaries defined by ADR-0018.

## Testing Decisions

- Use the existing public RuntimeContextPolicy assembly and message-rendering seam as the primary test boundary. Assert the prompt content supplied to a model rather than private collection layout or helper implementation.
- Extend existing Runtime-context prompt rendering tests as prior art for section placement, empty evidence, result/error rendering, and JSON-safe values.
- Add controlled native read calls across several Tool rounds and assert that Goal follows every evidence section while no goal remains in the displayed Durable State payload.
- Test both direct ReAct-style Durable State and nested Plan-step Executor-style Agent state to verify the same displayed-goal contract without changing their stored values.
- Test repeated `read_file` calls for one path and verify a single file group with one first-level code block per complete result, in round/request order, without call metadata or empty Round placeholders.
- Test grep output containing path-prefixed matches for multiple files and verify each match appears at the end of the appropriate file group without its redundant path prefix.
- Test explicit grep paths, malformed or pathless grep output, tool errors, and read failures; verify that these retain native request, complete result, and error in `(unfiled)`.
- Test that native `grep` and `read_file` retain their permanent-raw lifecycle and do not invoke the Observation model.
- Test positive and negative decision-impact Observations for each existing observation-class tool, including `exec`; verify negative calls retain Raw evidence through their normal window before omission, invalid or failed judgments retain Raw fallback, and merged Observations union their affected targets without mutating Durable State.
- Test that Observations flatten into three always-present categories with no displayed Round or tool-call IDs, preserving evidence order and duplicates while keeping their internal provenance.
- Test that non-native-read calls appear after the file groups with their existing round-grouped rendering, and that no existing context-budget, Raw-evidence, or diagnostic provenance behavior regresses.
- Test that a ReAct request does not render Completion criteria text or status; after a rejected completion proposal, it receives only the latest validated structured verdict with evidence and gaps.
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
ADR-0018 remains authoritative for evidence lifecycle and internal provenance.
The model-facing rendering intentionally omits some resolved native-read
request metadata and redundant grep path prefixes while preserving the
complete original Raw tool result internally. Goal remains after the evidence;
this revision changes the presentation of native
reads, other Raw results, and Observations.
