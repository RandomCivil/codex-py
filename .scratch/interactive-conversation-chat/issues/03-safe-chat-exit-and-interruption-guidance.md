# 03: Safe chat exit and interruption guidance

**What to build:** Let a terminal user leave an Interactive Conversation safely and receive precise reconnect guidance if input, a Conversation turn, or Conversation recovery is interrupted.

**Blocked by:** 01: Interactive new Conversation loop; 02: Reconnect and Conversation recovery.

**Status:** ready-for-agent

- [ ] EOF and `Ctrl+C` while input is pending exit cleanly without creating or submitting an additional Conversation turn.
- [ ] When an already identified Conversation is left or execution/recovery is interrupted, output includes a usable `chat --conv-id` reconnect command.
- [ ] A `KeyboardInterrupt` during a Conversation turn or recovery is not rendered as a fabricated terminal result; the next invocation can use the established recovery flow.
- [ ] CLI-seam tests cover normal exit, EOF, pending-input interruption, run interruption, recovery interruption, reconnect guidance, and applicable existing exit-status behavior.
