# 01: Interactive new Conversation loop

**What to build:** Let a terminal user run `agent conversation chat --config …` and conduct a new Interactive Conversation through `input("> ")`. The first nonempty input creates the Conversation, each terminal Conversation-turn result is rendered before the next input is accepted, and the user can keep interacting after a failed or blocked turn.

**Blocked by:** None (can start immediately).

**Status:** resolved

- [x] `chat` accepts the agreed configuration and new-turn options, rejects incompatible one-shot options, and preserves the existing one-shot Conversation commands.
- [x] A blank input is ignored; the first nonempty input creates a Conversation; later inputs continue that same `conv_id` sequentially; no prompt is shown until the preceding result has been rendered.
- [x] Default output is terminal-readable and `--json` renders the public result for interactive debugging while retaining the stdout prompt.
- [x] Exact `/exit` and `/quit` leave without submitting a turn, while lookalike text remains normal Conversation input.
- [x] CLI-seam tests cover the observable loop, result rendering, options, first-turn creation, serial ordering, terminal failed/blocked continuation, and slash-command behavior using controlled input and Conversation boundaries.

## Answer

Implemented the interactive `conversation chat` CLI loop at the existing `main()` seam. It validates chat-specific invocation, ignores blank input, creates and retains the Conversation ID across turns, renders completed/failed/blocked results, supports readable and JSON output, forwards new-turn options, and handles exact `/exit`/`/quit` commands. Existing one-shot Conversation commands remain unchanged.

Verification: `poetry run pytest -q` — 251 passed, 7 skipped.
