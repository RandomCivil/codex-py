# 01: Per-call native read evidence

**What to build:** Make the Runtime context window select evidence independently for each native Tool call, so recent directory discovery stays exact for three Tool rounds while `grep` and `read_file` evidence remains exact for the invocation. A mixed Tool-call batch must retain each native read call according to its own lifecycle, without changing how the batch settles.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [x] A context containing mixed native `list_dir`, `glob`, `grep`, and `read_file` calls retains each call according to its own Tool-call evidence policy rather than its batch's lifecycle.
- [x] `list_dir` and `glob` Raw tool results are omitted after their Tool round leaves the newest-three-round window, without requesting an Observation.
- [x] `grep` and `read_file` Raw tool results remain visible after that window and never request an Observation.
- [x] Required retained Raw tool results remain lossless and produce the existing explicit context-maintenance failure if they cannot fit with Durable State.
- [x] Existing Tool-call batch ordering, errors, and concurrent settlement behavior remain unchanged.
