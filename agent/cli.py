"""Command-line entry point for durable Agent runs."""

import argparse
import json
import os
import sys
import uuid
from collections.abc import Sequence

from agent.durable import resume_agent, run_agent
from agent.agent import RecoveryDecisionError
from agent.migration import MigrationConfigurationError, migrate_database
from agent.registry import ConfigurationMismatchError, RunBusyError


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
    parser.add_argument("command", choices=("migrate", "run", "resume"))
    parser.add_argument("--goal")
    parser.add_argument("--run-id")
    parser.add_argument("--recovery", choices=("retry", "fail", "abort"))
    try:
        args = parser.parse_args(argv)
    except InvalidInvocationError as exc:
        return _emit({"command": None, "status": "invalid", "error": str(exc)}, EXIT_INVALID_INVOCATION)

    if args.command == "run" and (not args.goal or args.run_id or args.recovery):
        return _emit({"command": "run", "status": "invalid", "error": "run requires --goal and does not accept --run-id or --recovery"}, EXIT_INVALID_INVOCATION)
    if args.command == "resume" and (not args.run_id or args.goal):
        return _emit({"command": "resume", "status": "invalid", "error": "resume requires --run-id and does not accept --goal"}, EXIT_INVALID_INVOCATION)
    if args.command == "resume":
        try:
            if uuid.UUID(args.run_id).version != 4:
                raise ValueError
        except (ValueError, AttributeError):
            return _emit({"command": "resume", "status": "invalid", "error": "resume requires a UUIDv4 --run-id"}, EXIT_INVALID_INVOCATION)
    if args.command == "migrate" and (args.goal or args.run_id or args.recovery):
        return _emit({"command": "migrate", "status": "invalid", "error": "migrate does not accept --goal, --run-id, or --recovery"}, EXIT_INVALID_INVOCATION)

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
            result = {"command": args.command, **run_agent(os.environ["CODEX_MYSQL_URL"], args.goal)}
            result.setdefault("run_id", str(uuid.uuid4()))
        else:
            if args.recovery is None:
                resumed = resume_agent(os.environ["CODEX_MYSQL_URL"], args.run_id)
            else:
                resumed = resume_agent(os.environ["CODEX_MYSQL_URL"], args.run_id, args.recovery)
            result = {"command": args.command, **resumed}
    except (MigrationConfigurationError, ConfigurationMismatchError) as exc:
        return _emit(
            {"command": args.command, "status": "configuration", "error": str(exc)},
            EXIT_CONFIGURATION,
        )
    except RunBusyError as exc:
        return _emit({"command": args.command, "status": "busy", "error": str(exc)}, EXIT_BUSY)
    except RecoveryDecisionError as exc:
        return _emit({"command": args.command, "status": "invalid", "error": str(exc)}, EXIT_INVALID_INVOCATION)
    except Exception as exc:
        return _emit(
            {"command": args.command, "status": "persistence", "error": str(exc)},
            EXIT_PERSISTENCE,
        )
    result.pop("state", None)
    status = EXIT_BLOCKED if result.get("status") == "blocked" else EXIT_COMPLETED
    return _emit(result, status)


def _emit(result: dict[str, object], exit_code: int) -> int:
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
