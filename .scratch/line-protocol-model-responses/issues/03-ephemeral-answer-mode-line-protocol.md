# 03: Ephemeral answer-mode Line Protocol slice

**What to build:** Direct mode and tool-agent mode without a native tool call return decoded `ANSWER.TEXT` as the existing public Execution answer, while native function-tool handling remains unchanged.

**Blocked by:** 01: Line Protocol core and Planner slice.

**Status:** ready-for-agent

- [ ] Direct mode completes only from a valid, nonempty `ANSWER` response and exposes decoded text rather than Line Protocol framing.
- [ ] Tool-agent mode with no native tool call has the same valid `ANSWER` requirement; successful native tool results retain their existing deterministic renderer path.
- [ ] A reply combining native tool calls with Line Protocol text, or an incomplete/invalid `ANSWER`, fails closed.
- [ ] Tests verify decoded public answers, tool-result compatibility, mixed-response rejection, and provider-format omission.
