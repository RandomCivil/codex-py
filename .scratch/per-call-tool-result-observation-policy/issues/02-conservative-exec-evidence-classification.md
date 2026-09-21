# 02: Conservative exec evidence classification

**What to build:** Give `exec` a deterministic, conservative Tool-call evidence classification so pure shell reads use the corresponding directory-discovery or durable-read lifecycle, while writes and any syntactically uncertain command safely retain write-class Raw fallback evidence.

**Blocked by:** 01: Per-call native read evidence.

**Status:** ready-for-human

- [x] Pure top-level `ls`, `find`, and `fd` commands use the directory-discovery lifecycle, while pure top-level `rg`, `grep`, `cat`, `sed`, `head`, and `tail` commands use the durable-read lifecycle.
- [x] Pure top-level `apply_patch`, `tee`, `cp`, `mv`, and `touch` commands, and commands containing redirection, are write class.
- [x] Pipes, command-list operators, subshells, variable expansion, unknown commands, malformed input, and all other uncertain syntax are write class without running or expanding the command.
- [x] Classification behavior is covered directly and command results appear in the Runtime context under their selected raw lifecycle.
