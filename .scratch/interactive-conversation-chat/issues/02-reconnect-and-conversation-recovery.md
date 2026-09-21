# 02: Reconnect and Conversation recovery

**What to build:** Let a terminal user reconnect with `agent conversation chat --conv-id …`, automatically reconcile/retry an Active Conversation turn before further input, view its recovered result, and use the established `fail` or `abort` recovery dispositions when needed.

**Blocked by:** 01: Interactive new Conversation loop.

**Status:** ready-for-agent

- [ ] `--conv-id` selects the existing Conversation for recovery and every newly submitted turn rather than creating a new one.
- [ ] Chat performs default retry recovery before its first prompt, supports `--recovery fail|abort`, and renders the recovered Conversation-turn result before accepting input.
- [ ] Recovery reuses the established Conversation and Execution-mode recovery semantics; it does not create a new store, lifecycle, or recovery path.
- [ ] CLI-seam tests verify default and explicit recovery dispositions, recovered-result ordering, forwarded IDs/options, and the existing handling of unrecoverable Ephemeral active turns.
