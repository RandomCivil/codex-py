"""Command-line entry point for durable Agent runs."""

import argparse
import json
import os
import sys
import uuid
from collections.abc import Sequence

from agent.durable import resume_agent, run_agent
from agent.agent import RecoveryDecisionError
from agent.configuration import ConfigurationError, load_configuration
from agent.migration import MigrationConfigurationError, migrate_database
from agent.registry import ConfigurationMismatchError, RunBusyError
from agent.conversation import (
    ConversationBusyError,
    ConversationError,
    ConversationNotFoundError,
    run_mysql_conversation,
    resume_mysql_conversation,
    show_mysql_conversation,
)


EXIT_COMPLETED = 0
EXIT_BLOCKED = 1
EXIT_BUSY = 2
EXIT_CONFIGURATION = 3
EXIT_PERSISTENCE = 4
EXIT_INVALID_INVOCATION = 5


class InvalidInvocationError(ValueError):
    """The command line cannot be interpreted as a supported invocation."""


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise InvalidInvocationError(message)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _ArgumentParser(prog="agent", add_help=False)
    parser.add_argument("command", choices=("migrate", "run", "resume", "conversation"))
    parser.add_argument("conversation_action", nargs="?", choices=("run", "resume", "show", "chat"))
    parser.add_argument("--goal")
    parser.add_argument("--run-id")
    parser.add_argument("--recovery", choices=("fail", "abort"))
    parser.add_argument("--cwd", help="working directory supplied to Atom MCP tool calls")
    parser.add_argument("--log-level", "--level", dest="log_level", choices=("info", "error"))
    parser.add_argument("--config")
    parser.add_argument("--execution", choices=("direct", "tool_agent", "react", "plan_execute"))
    parser.add_argument("--conv-id")
    parser.add_argument("--input", "--user-input", dest="input")
    parser.add_argument("--json", action="store_true", dest="json_output")
    try:
        args = parser.parse_args(argv)
    except InvalidInvocationError as exc:
        return _emit({"command": None, "status": "invalid", "error": str(exc)}, EXIT_INVALID_INVOCATION)

    if args.json_output and (args.command != "conversation" or args.conversation_action != "chat"):
        return _emit(
            {"command": args.command, "status": "invalid", "error": "--json is only supported by conversation chat"},
            EXIT_INVALID_INVOCATION,
        )
    if args.command == "run" and (not args.goal or args.run_id or args.recovery or not args.config):
        return _emit({"command": "run", "status": "invalid", "error": "run requires --goal and --config and does not accept --run-id or --recovery"}, EXIT_INVALID_INVOCATION)
    if args.command == "resume" and (not args.run_id or args.goal or not args.config or args.execution):
        return _emit({"command": "resume", "status": "invalid", "error": "resume requires --run-id and --config and does not accept --goal"}, EXIT_INVALID_INVOCATION)
    if args.command == "resume":
        try:
            if uuid.UUID(args.run_id).version != 4:
                raise ValueError
        except (ValueError, AttributeError):
            return _emit({"command": "resume", "status": "invalid", "error": "resume requires a UUIDv4 --run-id"}, EXIT_INVALID_INVOCATION)
    if args.command == "migrate" and (args.goal or args.run_id or args.recovery or args.cwd or args.log_level or args.config or args.execution):
            return _emit({"command": "migrate", "status": "invalid", "error": "migrate does not accept --goal, --run-id, --recovery, --cwd, --log-level, --config, or --execution"}, EXIT_INVALID_INVOCATION)
    if args.command == "conversation":
        if args.conversation_action == "run" and (not args.input or not args.config or args.goal or args.run_id or args.recovery):
            return _emit({"command": "conversation", "status": "invalid", "error": "conversation run requires --input and --config"}, EXIT_INVALID_INVOCATION)
        if args.conversation_action == "resume" and (not args.conv_id or not args.config or args.goal or args.run_id or args.input or args.cwd or args.log_level or args.execution):
            return _emit({"command": "conversation", "status": "invalid", "error": "conversation resume requires --conv-id and --config"}, EXIT_INVALID_INVOCATION)
        if args.conversation_action == "show" and (not args.conv_id or args.goal or args.run_id or args.recovery or args.config or args.input or args.cwd or args.log_level or args.execution):
            return _emit({"command": "conversation", "status": "invalid", "error": "conversation show requires --conv-id and does not accept run options"}, EXIT_INVALID_INVOCATION)
        if args.conversation_action == "chat" and (not args.config or args.goal or args.run_id or args.input):
            return _emit({"command": "conversation", "status": "invalid", "error": "conversation chat requires --config and does not accept --goal, --run-id, or --input"}, EXIT_INVALID_INVOCATION)
        if args.conversation_action is None:
            return _emit({"command": "conversation", "status": "invalid", "error": "conversation requires run or show"}, EXIT_INVALID_INVOCATION)

    try:
        configuration = load_configuration(args.config) if args.config else None
    except ConfigurationError as exc:
        return _emit({"command": args.command, "status": "configuration", "error": str(exc)}, EXIT_CONFIGURATION)

    if not os.environ.get("CODEX_MYSQL_URL"):
        return _emit(
            {
                "command": args.command,
                "status": "configuration",
                "error": "CODEX_MYSQL_URL is required",
            },
            EXIT_CONFIGURATION,
        )

    try:
        if args.command == "migrate":
            migrate_database(os.environ["CODEX_MYSQL_URL"])
            result = {"command": args.command, "status": "completed"}
        elif args.command == "run":
            options = {"configuration": configuration}
            if args.cwd:
                options["cwd"] = args.cwd
            if args.log_level:
                options["log_level"] = args.log_level
            if args.execution:
                options["execution_mode"] = args.execution
            run = run_agent(os.environ["CODEX_MYSQL_URL"], args.goal, **options)
            result = {"command": args.command, **run}
            result.setdefault("run_id", str(uuid.uuid4()))
        elif args.command == "resume":
            options = {"configuration": configuration}
            if args.recovery:
                options["recovery"] = args.recovery
            if args.cwd:
                options["cwd"] = args.cwd
            if args.log_level:
                options["log_level"] = args.log_level
            resumed = resume_agent(os.environ["CODEX_MYSQL_URL"], args.run_id, **options)
            result = {"command": args.command, **resumed}
        elif args.conversation_action == "run":
            conversation_options = {
                "user_input": args.input,
                "conv_id": args.conv_id,
                "configuration": configuration,
                "cwd": args.cwd,
                "log_level": args.log_level or "info",
            }
            if args.execution:
                conversation_options["execution_mode"] = args.execution
            result = {
                "command": "conversation",
                **run_mysql_conversation(
                    os.environ["CODEX_MYSQL_URL"],
                    **conversation_options,
                ).as_dict(),
            }
        elif args.conversation_action == "resume":
            result = {
                "command": "conversation",
                **resume_mysql_conversation(
                    os.environ["CODEX_MYSQL_URL"],
                    conv_id=args.conv_id,
                    recovery=args.recovery,
                    configuration=configuration,
                ).as_dict(),
            }
        elif args.conversation_action == "chat":
            return _run_chat(
                os.environ["CODEX_MYSQL_URL"],
                configuration=configuration,
                conv_id=args.conv_id,
                cwd=args.cwd,
                log_level=args.log_level or "info",
                execution_mode=args.execution,
                recovery=args.recovery,
                json_output=args.json_output,
            )
        else:
            result = {
                "command": "conversation",
                "conv_id": args.conv_id,
                "history": show_mysql_conversation(os.environ["CODEX_MYSQL_URL"], args.conv_id),
            }
    except (MigrationConfigurationError, ConfigurationMismatchError) as exc:
        return _emit(
            {"command": args.command, "status": "configuration", "error": str(exc)},
            EXIT_CONFIGURATION,
        )
    except ConversationBusyError as exc:
        return _emit(
            {
                "command": args.command,
                "status": "busy",
                "error": str(exc),
                "conv_id": exc.conv_id,
                "sequence": exc.sequence,
                "run_id": exc.run_id,
            },
            EXIT_BUSY,
        )
    except RunBusyError as exc:
        return _emit({"command": args.command, "status": "busy", "error": str(exc)}, EXIT_BUSY)
    except RecoveryDecisionError as exc:
        return _emit({"command": args.command, "status": "invalid", "error": str(exc)}, EXIT_INVALID_INVOCATION)
    except (ConversationNotFoundError, ValueError) as exc:
        return _emit({"command": args.command, "status": "invalid", "error": str(exc)}, EXIT_INVALID_INVOCATION)
    except Exception as exc:
        return _emit(
            {"command": args.command, "status": "persistence", "error": str(exc)},
            EXIT_PERSISTENCE,
        )
    result.pop("state", None)
    status = EXIT_BLOCKED if result.get("status") in {"blocked", "failed"} else EXIT_COMPLETED
    return _emit(result, status)


def _emit(result: dict[str, object], exit_code: int) -> int:
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return exit_code


def _run_chat(
    url: str,
    *,
    configuration: object,
    conv_id: str | None,
    cwd: str | None,
    log_level: str,
    execution_mode: str | None,
    recovery: str | None,
    json_output: bool,
) -> int:
    current_conv_id = conv_id
    if current_conv_id is not None:
        history = show_mysql_conversation(url, current_conv_id)
        _emit_chat_history(history, json_output=json_output)
        try:
            recovered = resume_mysql_conversation(
                url,
                conv_id=current_conv_id,
                recovery=recovery,
                configuration=configuration,
            )
        except KeyboardInterrupt:
            _emit_reconnect_guidance(current_conv_id)
            return EXIT_COMPLETED
        except ConversationError as exc:
            if str(exc) not in {
                "conversation has no active turn",
                "only a plan_execute turn can be resumed",
            }:
                raise
        else:
            current_conv_id = recovered.conv_id
            _emit_chat_result(recovered.as_dict(), json_output=json_output)
    while True:
        try:
            user_input = input("> ")
        except (EOFError, KeyboardInterrupt):
            _emit_reconnect_guidance(current_conv_id)
            return EXIT_COMPLETED
        if user_input in {"/exit", "/quit"}:
            _emit_reconnect_guidance(current_conv_id)
            return EXIT_COMPLETED
        if not user_input.strip():
            continue
        creating = current_conv_id is None
        if creating:
            current_conv_id = str(uuid.uuid4())
        try:
            result = run_mysql_conversation(
                url,
                user_input=user_input,
                conv_id=current_conv_id,
                configuration=configuration,
                cwd=cwd,
                log_level=log_level,
                create=creating,
                **({"execution_mode": execution_mode} if execution_mode else {}),
            )
        except KeyboardInterrupt:
            _emit_reconnect_guidance(current_conv_id)
            return EXIT_COMPLETED
        except Exception as exc:
            _emit_chat_error(exc)
            continue
        current_conv_id = result.conv_id
        _emit_chat_result(result.as_dict(), json_output=json_output)


def _emit_chat_result(result: dict[str, object], *, json_output: bool) -> None:
    if json_output:
        _emit({"command": "conversation", **result}, EXIT_COMPLETED)
        return
    print(f"Conversation: {result['conv_id']}")
    print(f"Turn {result['sequence']} ({result['execution_mode']}) — {result['status']}")
    if result.get("answer") is not None:
        print(result["answer"])
    if result.get("error") is not None:
        print(f"Error: {result['error']}")


def _emit_chat_history(history: list[dict[str, object]], *, json_output: bool) -> None:
    if not history:
        return
    if json_output:
        print(
            json.dumps(
                {"command": "conversation", "history": history},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        return
    print("Conversation history:")
    for turn in history:
        print(f"Turn {turn['sequence']} ({turn['execution_mode']}) — {turn['status']}")
        print(f"User: {turn['user_input']}")
        if turn.get("answer") is not None:
            print(f"Answer: {turn['answer']}")
        if turn.get("error") is not None:
            print(f"Error: {turn['error']}")


def _emit_reconnect_guidance(conv_id: str | None) -> None:
    if conv_id is not None:
        print(f"Reconnect with: agent conversation chat --conv-id {conv_id}")


def _emit_chat_error(error: Exception) -> None:
    """Keep interactive chat alive while making unexpected errors visible."""
    print(f"Error: {error}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    sys.exit(main())
