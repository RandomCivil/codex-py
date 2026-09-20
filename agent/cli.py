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
    parser.add_argument("conversation_action", nargs="?", choices=("run", "resume", "show"))
    parser.add_argument("--goal")
    parser.add_argument("--run-id")
    parser.add_argument("--recovery", choices=("fail", "abort"))
    parser.add_argument("--cwd", help="working directory supplied to Atom MCP tool calls")
    parser.add_argument("--log-level", "--level", dest="log_level", choices=("info", "error"))
    parser.add_argument("--config")
    parser.add_argument("--conv-id")
    parser.add_argument("--input", "--user-input", dest="input")
    try:
        args = parser.parse_args(argv)
    except InvalidInvocationError as exc:
        return _emit({"command": None, "status": "invalid", "error": str(exc)}, EXIT_INVALID_INVOCATION)

    if args.command == "run" and (not args.goal or args.run_id or args.recovery or not args.config):
        return _emit({"command": "run", "status": "invalid", "error": "run requires --goal and --config and does not accept --run-id or --recovery"}, EXIT_INVALID_INVOCATION)
    if args.command == "resume" and (not args.run_id or args.goal or not args.config):
        return _emit({"command": "resume", "status": "invalid", "error": "resume requires --run-id and --config and does not accept --goal"}, EXIT_INVALID_INVOCATION)
    if args.command == "resume":
        try:
            if uuid.UUID(args.run_id).version != 4:
                raise ValueError
        except (ValueError, AttributeError):
            return _emit({"command": "resume", "status": "invalid", "error": "resume requires a UUIDv4 --run-id"}, EXIT_INVALID_INVOCATION)
    if args.command == "migrate" and (args.goal or args.run_id or args.recovery or args.cwd or args.log_level or args.config):
            return _emit({"command": "migrate", "status": "invalid", "error": "migrate does not accept --goal, --run-id, --recovery, --cwd, --log-level, or --config"}, EXIT_INVALID_INVOCATION)
    if args.command == "conversation":
        if args.conversation_action == "run" and (not args.input or not args.config or args.goal or args.run_id or args.recovery):
            return _emit({"command": "conversation", "status": "invalid", "error": "conversation run requires --input and --config"}, EXIT_INVALID_INVOCATION)
        if args.conversation_action == "resume" and (not args.conv_id or not args.config or args.goal or args.run_id or args.input or args.cwd or args.log_level):
            return _emit({"command": "conversation", "status": "invalid", "error": "conversation resume requires --conv-id and --config"}, EXIT_INVALID_INVOCATION)
        if args.conversation_action == "show" and (not args.conv_id or args.goal or args.run_id or args.recovery or args.config or args.input or args.cwd or args.log_level):
            return _emit({"command": "conversation", "status": "invalid", "error": "conversation show requires --conv-id and does not accept run options"}, EXIT_INVALID_INVOCATION)
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
            result = {
                "command": "conversation",
                **run_mysql_conversation(
                    os.environ["CODEX_MYSQL_URL"],
                    user_input=args.input,
                    conv_id=args.conv_id,
                    configuration=configuration,
                    cwd=args.cwd,
                    log_level=args.log_level or "info",
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


if __name__ == "__main__":
    sys.exit(main())
