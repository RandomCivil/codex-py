# Interactive terminal Conversations

Status: ready-for-agent

## Problem Statement

Conversation callers currently have to invoke a separate one-shot command for every user input, copy the returned `conv_id`, and manually arrange the next invocation. This makes a normal terminal conversation cumbersome and offers no interactive way to resume an existing Conversation while preserving its strict turn ordering and recovery behavior.

## Solution

Add `agent conversation chat`, a terminal-oriented interface driven by `input("> ")`. It creates a Conversation on its first nonempty submitted input when no `--conv-id` is supplied, or reconnects to the supplied Conversation when one is supplied. It displays each terminal Conversation-turn result before accepting the next input, handles deliberate exit and interruption safely, automatically recovers an Active Conversation turn by default, and retains the existing one-shot JSON commands for automation.

## User Stories

1. As a terminal user, I want to start `agent conversation chat` without a Conversation ID, so that I can begin a new continuing interaction without manually invoking one command per turn.
2. As a terminal user, I want the chat command to wait for input with an `input("> ")` prompt, so that I have a familiar single-line interaction model.
3. As a terminal user, I want a new Conversation created only after I submit a nonempty first input, so that opening and immediately exiting chat does not persist an empty Conversation.
4. As a terminal user, I want the newly created `conv_id` shown with the first result, so that I can reconnect to that Conversation later.
5. As a terminal user, I want to supply `--conv-id` to reconnect to an existing Conversation, so that a later terminal process continues the same history rather than starting a new one.
6. As a terminal user, I want each submitted input to create exactly one next Conversation turn, so that the durable ordering and history remain unambiguous.
7. As a terminal user, I want the full result of a turn displayed before another prompt appears, so that I cannot submit another input while the prior turn is still running or its output is incomplete.
8. As a terminal user, I want a completed answer displayed in the interactive transcript, so that I can decide what to ask next.
9. As a terminal user, I want a failed or blocked turn's status and safe error displayed, so that I can immediately clarify or correct my request.
10. As a terminal user, I want to continue after a terminal failed or blocked turn, so that a single unsuccessful request does not end the Conversation.
11. As a terminal user, I want an empty input line ignored, so that accidental Enter presses do not create an invalid turn or end the chat.
12. As a terminal user, I want a line exactly equal to `/exit` or `/quit` to end chat, so that I can leave deliberately without changing Conversation history.
13. As a terminal user, I want text merely beginning with `/exit` or `/quit` to remain normal Conversation input, so that I can ask the model about those strings.
14. As a terminal user, I want EOF or `Ctrl+C` while waiting for input to exit cleanly, so that standard terminal controls do not corrupt the Conversation.
15. As a terminal user, I want an interruption while a turn is executing to print the Conversation recovery command, so that I can safely reconnect to the Active Conversation turn instead of guessing how to resume it.
16. As a terminal user reconnecting to a Conversation with Active durable work, I want chat to retry the existing Conversation recovery by default, so that I can continue without a separate recovery command in ordinary cases.
17. As a terminal user, I want `--recovery fail` and `--recovery abort` available on reconnect, so that I can explicitly choose an established non-retry recovery disposition when necessary.
18. As a terminal user, I want the recovered turn's complete result displayed before the next prompt, so that I know the precise state that precedes my next input.
19. As a terminal user, I want the command to retain the existing treatment of interrupted Ephemeral executions, so that chat never claims to recover a turn that the selected Execution mode cannot recover.
20. As a terminal user, I want normal chat output to be readable rather than constrained to a one-line machine protocol, so that multi-line answers are comfortable to use in a terminal.
21. As a terminal user, I want an optional `--json` display mode that emits each result as JSON while retaining the prompt on stdout, so that I can inspect the established result shape during interactive debugging.
22. As an automation author, I want `conversation run`, `conversation resume`, and `conversation show` unchanged, so that the existing one-shot JSON contracts remain the proper noninteractive interface.
23. As a terminal user, I want `--config` required and the existing `--cwd`, `--log-level`, and `--execution` options supported, so that chat uses the same model, tool-working-directory, logging, and Execution-mode controls as a one-shot Conversation turn.
24. As a terminal user, I want an explicit `--execution` to affect only newly submitted turns, so that recovery never changes the already persisted mode of Active work.
25. As a terminal user, I want unsupported combinations such as `--input`, `--goal`, and `--run-id` rejected for chat, so that a command cannot silently mix interactive and one-shot semantics.
26. As an operator, I want the existing active-turn serialization to remain in force, so that another process cannot append work between chat turns or run concurrently with its current turn.

## Implementation Decisions

- Use the glossary terms Interactive Conversation, Conversation, Conversation turn, Active Conversation turn, Conversation recovery, and Conversation-turn result. An Interactive Conversation is a terminal interface over the existing durable Conversation aggregate; it is not a new persistence model or an Agent run.
- Add `chat` as a Conversation CLI action. Keep the existing `run`, `resume`, and `show` actions and their behavior intact.
- The chat loop must call the built-in `input()` with the exact prompt `> `. It submits one nonempty input at a time, waits synchronously for the existing Conversation operation to produce its terminal result, renders that result, and only then asks for another input.
- When `--conv-id` is omitted, generate the Conversation identity in the interactive adapter and use it for the first submitted turn. Do not invoke the Conversation service or persist anything until a nonempty input is submitted. After the first result, retain that ID for all later turns and show it in the transcript and exit guidance.
- When `--conv-id` is supplied, use it for every submitted turn. Before accepting new input, invoke existing Conversation recovery/reconciliation semantics. Default recovery is retry; `--recovery fail` and `--recovery abort` forward the established disposition. Render a recovered turn exactly as another turn result before prompting.
- Delegate submitted inputs and recovery to the existing MySQL Conversation entry points. Do not create a second Conversation service, store, schema, lifecycle, mode-selection mechanism, or serialization mechanism. Existing Active Conversation turn rules continue to determine busy and terminal behavior.
- A line exactly matching `/exit` or `/quit` exits without submitting a turn. Empty or whitespace-only input is ignored. All other text, including strings that merely start with those commands, is submitted unchanged.
- EOF and `KeyboardInterrupt` while input is pending exit cleanly and print reconnect guidance when a Conversation ID exists. A `KeyboardInterrupt` during recovery or a turn run exits with reconnect guidance for the known Conversation ID and does not report the interrupted work as a terminal result.
- Default rendering is terminal-readable and includes the Conversation ID, turn sequence, execution mode, status, answer when present, and error when present. `--json` renders the same per-turn public result as JSON for interactive debugging only; because the prompt remains stdout, it is not a pipe-safe protocol. Runtime logs retain their existing stderr behavior.
- `chat` requires `--config`; accepts `--conv-id`, `--recovery`, `--cwd`, `--log-level`, `--execution`, and `--json`; and rejects `--input`, `--goal`, and `--run-id`. `--execution` configures only turns started by the chat loop, never recovery of an existing turn.
- Preserve existing exit-status conventions for configuration, busy, invalid invocation, persistence, failed, and blocked outcomes where the command terminates. A failed or blocked submitted turn is rendered and the loop remains available for another input.
- Respect ADR-0011 through ADR-0015: Conversations remain Agent-run aggregates with serialized Active turns, use existing Execution modes, retain structured history, and own recovery at the Conversation boundary.

## Testing Decisions

- Test exclusively at the existing CLI `main()` seam. Replace interactive input, capture standard output/error, and replace the existing Conversation run/recovery boundaries. Good tests assert externally visible command behavior and forwarded public options, not loop helper structure, SQL, private calls, model prompts, or persistence internals.
- Verify an omitted `--conv-id` waits for the first nonempty input before invoking the Conversation boundary, then retains and displays the returned Conversation ID for later submitted turns and exit guidance.
- Verify a supplied Conversation ID is used for recovery and all later submitted turns; verify default recovery and explicit `fail`/`abort` dispositions are forwarded, and a recovered result is rendered before the first prompt.
- Verify exactly one input is processed at a time: the loop does not request the next input until its prior run result has been rendered. Verify completed, failed, and blocked results are rendered, and that failed/blocked turns do not terminate the loop.
- Verify blank input is ignored; exact `/exit` and `/quit` exit without a Conversation submission; and lookalike text remains a submitted input.
- Verify EOF and `KeyboardInterrupt` while waiting exit cleanly. Verify interruption during a run or recovery prints reconnect guidance containing the known Conversation ID and leaves the recovery path to a subsequent invocation.
- Verify default readable rendering contains the public result fields, while `--json` emits the equivalent JSON result while still exercising an stdout prompt. Do not characterize `--json` as pipe-safe.
- Verify required, accepted, and rejected CLI arguments; verify `--execution`, `--cwd`, and log level are forwarded for new turns; and verify invalid combinations return the established invalid-invocation result.
- Use the existing CLI migration tests as prior art for direct `main()` invocation, monkeypatching downstream boundaries, captured output, argument validation, and stable JSON. Conversation service, ephemeral-mode, Plan–execute recovery, and MySQL integration tests remain the prior art and coverage for downstream lifecycle, persistence, ordering, and recovery behavior; do not duplicate them in chat tests.

## Out of Scope

- Changing Conversation storage, migrations, history structure, serialization, mode selection, or recovery semantics.
- A graphical UI, terminal history navigation, multiline editor, Conversation list/search/delete/export, or command completion.
- Pipe-safe JSON streaming, a new machine-readable interactive protocol, or changing the existing one-shot JSON commands.
- Changing the behavior or durability of Direct, tool-agent, ReAct, or Plan–execute modes.
- Authentication, authorization, multi-user Conversation ownership, and Conversation identity changes.
- Conversation-history truncation, compaction, or context-budget policy.

## Further Notes

The feature intentionally layers terminal ergonomics over the existing Conversation boundary instead of adding a new execution path. The first submitted turn creates a new Conversation so that an abandoned prompt produces no empty durable aggregate. Reconnect guidance must include the command form with `--conv-id`, allowing a user to resume after any deliberate exit or execution-time interruption.
