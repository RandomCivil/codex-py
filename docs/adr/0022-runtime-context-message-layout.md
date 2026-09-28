# Native message layout for Runtime context

**Status: accepted.** Runtime context has an opt-in `messages` layout selected
by `--runtime-context-layout`, while `grouped` remains the default for existing
callers. The messages layout preserves retained Raw evidence as native AI
tool-call and Tool result pairs, rather than regrouping native reads by file,
then appends Durable State, Observations, repair or correction feedback, Goal,
and acceptance criteria in a fixed order. The component system prompt is
always first. The layout is part of durable-run configuration identity and its
token budget covers the full model request, so recovery and bounded-context
behavior cannot silently change when a caller selects the layout.

Each model role retains its existing evidence filter. In particular, a
Plan-step Completion judge receives only its established successful evidence
view, while failed Tool results remain in the operational loop for correction.

## Considered Options

- Replace `grouped` for all callers — rejected because prompt organization is
  externally observable and existing integrations depend on the file-grouped
  rendering.
- Keep the native message transcript alongside grouped evidence — rejected
  because it creates a second historical evidence channel and can show tool
  calls without their retained result.
