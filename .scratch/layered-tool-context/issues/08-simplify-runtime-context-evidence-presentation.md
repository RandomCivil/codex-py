# 08: Simplify Runtime-context evidence presentation

**What to build:** Simplify the model-facing Runtime context: show resolved native read results directly beneath file headings, move all other Raw tool results after file groups, and flatten Observations into three categories without displayed Round headings or tool-call IDs. Keep internal evidence, provenance, diagnostics, and retention behavior unchanged.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

## Acceptance criteria

- [ ] Each resolved `read_file` or `grep` result appears as a separate first-level code block directly beneath `#### File: <path>`; complete multiline content and result order are preserved, while call metadata and empty native-read Round placeholders are absent.
- [ ] Path-prefixed grep matches are split by file, appended to the corresponding file group, and displayed without the redundant path prefix. Their complete original results remain available internally and in diagnostics.
- [ ] Failed or unresolvable native reads still appear in `(unfiled)` with tool name, arguments, complete result, and error.
- [ ] Non-native Raw tool results appear after all file groups, preserving their existing Round headings and call details.
- [ ] `### Observations` always shows Confirmed facts, Reported errors, and Model inferences in that order. Evidence from all visible Observations is flattened in source order without deduplication; Round headings and tool-call IDs are absent from the model-facing text, including when categories are empty.
- [ ] ReAct and Plan-step Executor use the same revised presentation; internal Observation/Raw tool-result records, diagnostics, evidence lifecycle, token budgeting, and Checkpoint boundaries remain unchanged.

See [the feature specification](../spec.md) for the full rendering contract and examples.
