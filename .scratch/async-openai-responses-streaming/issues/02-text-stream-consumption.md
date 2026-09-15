# 02 — Text Stream Consumption

**What to build:** A text-rendering layer can consume a Text stream from the configured LLM and receive only successive generated text deltas, ending normally when the model response completes, without needing to interpret other Event stream values.

**Blocked by:** 01 — Event Stream LLM Session.

**Status:** ready-for-agent

- [x] A Text stream yields text deltas in their original order and excludes reasoning, function-call, completion, usage, and other non-text events.
- [x] The Text stream terminates normally after completion and releases stream resources when its consumer exits early.
- [x] Tests exercise the public LLM seam with a controlled provider/SDK boundary and do not require live credentials.

## Comments

- Implemented `LLM.stream_text`, which yields only `response.output_text.delta` payloads while reusing the event stream's provider-resource lifecycle.
- Added public-interface tests for ordered text filtering and early-consumer cleanup; no live credentials are required.
