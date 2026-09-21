import json

import pytest

from agent.cli import main
from agent.conversation import ConversationError, ConversationResult


def _configuration_file(tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text(
        "model:\n"
        "  base_url: https://provider.test/v1\n"
        "  api_key: key\n"
        "  model_name: model\n"
    )
    return str(path)


def test_chat_creates_a_conversation_only_after_first_nonempty_input(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")
    monkeypatch.setattr(
        "agent.cli.uuid.uuid4", lambda: "550e8400-e29b-41d4-a716-446655440000"
    )
    submitted = []

    def run(url, **kwargs):
        submitted.append(kwargs)
        return ConversationResult(
            "550e8400-e29b-41d4-a716-446655440000",
            1,
            "550e8400-e29b-41d4-a716-446655440001",
            "direct",
            "completed",
            "Hello back",
        )

    monkeypatch.setattr("agent.cli.run_mysql_conversation", run)
    answers = iter(["", "Hello", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))

    assert main(["conversation", "chat", "--config", _configuration_file(tmp_path)]) == 0

    assert len(submitted) == 1
    assert submitted[0]["user_input"] == "Hello"
    assert submitted[0]["conv_id"] == "550e8400-e29b-41d4-a716-446655440000"
    assert submitted[0]["create"] is True
    assert submitted[0]["configuration"] is not None
    assert submitted[0]["cwd"] is None
    assert submitted[0]["log_level"] == "info"
    output = capsys.readouterr().out
    assert "550e8400-e29b-41d4-a716-446655440000" in output
    assert "Hello back" in output


def test_chat_exact_exit_does_not_submit_a_turn(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")
    monkeypatch.setattr(
        "agent.cli.run_mysql_conversation",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("submitted")),
    )
    monkeypatch.setattr("builtins.input", lambda prompt: "/quit")

    assert main(["conversation", "chat", "--config", _configuration_file(tmp_path)]) == 0
    assert capsys.readouterr().out == ""


def test_chat_renders_each_result_before_requesting_the_next_input(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")
    results = iter(
        [
            ConversationResult("conv", 1, "run-1", "direct", "failed", error="try again"),
            ConversationResult("conv", 2, "run-2", "direct", "blocked", error="blocked"),
        ]
    )
    events = []

    def run(url, **kwargs):
        events.append(("run", kwargs["user_input"]))
        return next(results)

    def get_input(prompt):
        events.append(("prompt", capsys.readouterr().out))
        return ["first", "second", "/exit"][len([event for event in events if event[0] == "prompt"]) - 1]

    monkeypatch.setattr("agent.cli.run_mysql_conversation", run)
    monkeypatch.setattr("builtins.input", get_input)

    assert main(["conversation", "chat", "--config", _configuration_file(tmp_path)]) == 0

    assert [event[0:2] for event in events] == [
        ("prompt", ""),
        ("run", "first"),
        ("prompt", "Conversation: conv\nTurn 1 (direct) — failed\nError: try again\n"),
        ("run", "second"),
        ("prompt", "Conversation: conv\nTurn 2 (direct) — blocked\nError: blocked\n"),
    ]


def test_chat_prints_unexpected_turn_errors_to_stderr_and_continues(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")
    calls = []

    def run(url, **kwargs):
        calls.append(kwargs["user_input"])
        if len(calls) == 1:
            raise RuntimeError("provider connection failed")
        return ConversationResult("conv", 1, "run-1", "direct", "completed", "recovered")

    monkeypatch.setattr("agent.cli.run_mysql_conversation", run)
    answers = iter(["first", "second", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))

    assert main(["conversation", "chat", "--config", _configuration_file(tmp_path)]) == 0

    captured = capsys.readouterr()
    assert "Error: provider connection failed" in captured.err
    assert "recovered" in captured.out


def test_chat_forwards_new_turn_options_and_retains_conversation_id(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")
    calls = []

    def run(url, **kwargs):
        calls.append((url, kwargs))
        return ConversationResult("conv", len(calls), f"run-{len(calls)}", "react", "completed", "ok")

    monkeypatch.setattr("agent.cli.run_mysql_conversation", run)
    answers = iter(["first", "second", "/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))

    assert main(
        [
            "conversation",
            "chat",
            "--config",
            _configuration_file(tmp_path),
            "--cwd",
            "/workspace/project",
            "--log-level",
            "error",
            "--execution",
            "react",
        ]
    ) == 0

    assert calls[0][0] == "mysql://user:pass@localhost/db"
    assert calls[0][1]["conv_id"] is not None
    assert calls[0][1]["create"] is True
    assert calls[0][1]["cwd"] == "/workspace/project"
    assert calls[0][1]["log_level"] == "error"
    assert calls[0][1]["execution_mode"] == "react"
    assert calls[1][1]["conv_id"] == "conv"
    assert calls[1][1]["create"] is False


def test_chat_json_renders_public_turn_result(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")
    monkeypatch.setattr(
        "agent.cli.run_mysql_conversation",
        lambda *args, **kwargs: ConversationResult("conv", 1, "run", "direct", "completed", "answer"),
    )
    answers = iter(["hello", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))

    assert main(["conversation", "chat", "--config", _configuration_file(tmp_path), "--json"]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert '"conv_id":"conv"' in lines[0]
    assert '"answer":"answer"' in lines[0]


def test_chat_submits_slash_lookalikes_as_normal_input(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")
    inputs = []

    def run(url, **kwargs):
        inputs.append(kwargs["user_input"])
        return ConversationResult("conv", 1, "run", "direct", "completed", "answer")

    monkeypatch.setattr("agent.cli.run_mysql_conversation", run)
    answers = iter(["/exit please", "/quit now", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))

    assert main(["conversation", "chat", "--config", _configuration_file(tmp_path)]) == 0
    assert inputs == ["/exit please", "/quit now"]


@pytest.mark.parametrize("recovery", [None, "fail", "abort"])
def test_chat_recovers_reconnected_conversation_before_first_prompt(
    monkeypatch, capsys, tmp_path, recovery
):
    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")
    events = []

    def resume(url, **kwargs):
        events.append(("resume", url, kwargs))
        return ConversationResult("conv", 2, "run-2", "plan_execute", "completed", "Recovered")

    def run(url, **kwargs):
        events.append(("run", url, kwargs))
        return ConversationResult("conv", 3, "run-3", "direct", "completed", "Next answer")

    def get_input(prompt):
        events.append(("prompt", capsys.readouterr().out))
        return ["next input", "/exit"][len([event for event in events if event[0] == "prompt"]) - 1]

    monkeypatch.setattr("agent.cli.resume_mysql_conversation", resume)
    monkeypatch.setattr("agent.cli.run_mysql_conversation", run)
    monkeypatch.setattr("builtins.input", get_input)

    argv = [
        "conversation",
        "chat",
        "--conv-id",
        "conv",
        "--config",
        _configuration_file(tmp_path),
        "--execution",
        "direct",
    ]
    if recovery is not None:
        argv.extend(["--recovery", recovery])

    assert main(argv) == 0

    assert events[0][0:2] == ("resume", "mysql://user:pass@localhost/db")
    assert events[0][2]["conv_id"] == "conv"
    assert events[0][2]["recovery"] is recovery
    assert events[0][2]["configuration"] is not None
    assert events[1][0] == "prompt"
    assert "Recovered" in events[1][1]
    assert events[2][0] == "run"
    assert events[2][2]["conv_id"] == "conv"
    assert events[2][2]["execution_mode"] == "direct"


def test_chat_keeps_existing_ephemeral_recovery_handling(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")
    calls = []

    def resume(url, **kwargs):
        calls.append(("resume", kwargs))
        raise ConversationError("only a plan_execute turn can be resumed")

    def run(url, **kwargs):
        calls.append(("run", kwargs))
        return ConversationResult("conv", 2, "run-2", "direct", "failed", error="could not recover")

    monkeypatch.setattr("agent.cli.resume_mysql_conversation", resume)
    monkeypatch.setattr("agent.cli.run_mysql_conversation", run)
    answers = iter(["continue", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))

    assert main(
        ["conversation", "chat", "--conv-id", "conv", "--config", _configuration_file(tmp_path)]
    ) == 0

    assert [call[0] for call in calls] == ["resume", "run"]
    assert calls[1][1]["conv_id"] == "conv"
    assert "could not recover" in capsys.readouterr().out


def test_chat_eof_exits_cleanly_with_reconnect_guidance(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")
    monkeypatch.setattr(
        "agent.cli.resume_mysql_conversation",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ConversationError("conversation has no active turn")
        ),
    )
    monkeypatch.setattr(
        "agent.cli.run_mysql_conversation",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("submitted")),
    )
    monkeypatch.setattr("builtins.input", lambda prompt: (_ for _ in ()).throw(EOFError))

    assert main(
        ["conversation", "chat", "--conv-id", "conv", "--config", _configuration_file(tmp_path)]
    ) == 0

    assert "agent conversation chat --conv-id conv" in capsys.readouterr().out


def test_chat_recovery_interrupt_exits_with_reconnect_guidance(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")
    monkeypatch.setattr(
        "agent.cli.resume_mysql_conversation",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt),
    )
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: (_ for _ in ()).throw(AssertionError("prompted")),
    )

    assert main(
        ["conversation", "chat", "--conv-id", "conv", "--config", _configuration_file(tmp_path)]
    ) == 0

    output = capsys.readouterr().out
    assert "agent conversation chat --conv-id conv" in output
    assert '"status"' not in output


def test_chat_run_interrupt_exits_with_reconnect_guidance(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")
    monkeypatch.setattr(
        "agent.cli.resume_mysql_conversation",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ConversationError("conversation has no active turn")
        ),
    )
    monkeypatch.setattr(
        "agent.cli.run_mysql_conversation",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt),
    )
    monkeypatch.setattr("builtins.input", lambda prompt: "continue")

    assert main(
        ["conversation", "chat", "--conv-id", "conv", "--config", _configuration_file(tmp_path)]
    ) == 0

    output = capsys.readouterr().out
    assert "agent conversation chat --conv-id conv" in output
    assert '"status"' not in output


def test_chat_first_turn_interrupt_exits_with_generated_reconnect_guidance(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")
    monkeypatch.setattr("agent.cli.uuid.uuid4", lambda: "new-conversation")
    monkeypatch.setattr(
        "agent.cli.run_mysql_conversation",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt),
    )
    monkeypatch.setattr("builtins.input", lambda prompt: "continue")

    assert main(["conversation", "chat", "--config", _configuration_file(tmp_path)]) == 0

    output = capsys.readouterr().out
    assert "agent conversation chat --conv-id new-conversation" in output
    assert '"status"' not in output


@pytest.mark.parametrize("pending_input", ["/quit", KeyboardInterrupt])
def test_chat_leaves_existing_conversation_without_submitting(
    monkeypatch, capsys, tmp_path, pending_input
):
    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")
    monkeypatch.setattr(
        "agent.cli.resume_mysql_conversation",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ConversationError("conversation has no active turn")
        ),
    )
    monkeypatch.setattr(
        "agent.cli.run_mysql_conversation",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("submitted")),
    )
    if pending_input is KeyboardInterrupt:
        monkeypatch.setattr(
            "builtins.input", lambda prompt: (_ for _ in ()).throw(KeyboardInterrupt)
        )
    else:
        monkeypatch.setattr("builtins.input", lambda prompt: pending_input)

    assert main(
        ["conversation", "chat", "--conv-id", "conv", "--config", _configuration_file(tmp_path)]
    ) == 0

    assert "agent conversation chat --conv-id conv" in capsys.readouterr().out


def test_chat_rejects_one_shot_options_and_missing_config(monkeypatch, capsys):
    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")

    assert main(["conversation", "chat", "--input", "Hello"]) == 5
    assert json.loads(capsys.readouterr().out)["status"] == "invalid"

    assert main(["conversation", "chat", "--goal", "Hello"]) == 5
    assert json.loads(capsys.readouterr().out)["status"] == "invalid"

    assert main(["conversation", "chat", "--run-id", "run"]) == 5
    assert json.loads(capsys.readouterr().out)["status"] == "invalid"


def test_chat_rejects_json_for_one_shot_conversation_commands(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")

    assert main(
        ["conversation", "run", "--config", _configuration_file(tmp_path), "--input", "Hello", "--json"]
    ) == 5

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "invalid"
    assert result["error"] == "--json is only supported by conversation chat"
