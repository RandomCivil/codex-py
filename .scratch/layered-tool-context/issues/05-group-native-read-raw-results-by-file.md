# 05: Group native read Raw tool results by file

**What to build:** Permanent-raw native `grep` and `read_file` evidence is grouped by file across Tool rounds, with ordered entries and a lossless `(unfiled)` fallback for errors or unresolvable paths; all other evidence behavior remains unchanged.

**Blocked by:** None (can start immediately).

**Status:** resolved

- [x] Repeated native reads and path-prefixed grep matches appear under one resolved file group, ordered by Tool round and original request order.
- [x] A failed or unresolvable native read appears in `(unfiled)` with its tool name, arguments, complete result, and reported error.
- [x] Native `grep` and `read_file` remain permanent raw without Observation requests; non-native-read evidence keeps its existing rendering and retention behavior.

## Answer

Implemented file grouping for permanent-raw native `grep` and `read_file` results across Tool rounds, including explicit paths, path-prefixed grep splitting, ordered provenance, and lossless `(unfiled)` rendering for failures or unresolved paths. Native reads remain permanent raw and do not request Observations; other evidence retains its existing lifecycle.

Verification: `.venv/bin/pytest -q` — 269 passed, 7 skipped.
