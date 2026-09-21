import asyncio

from agent.runtime_context import RuntimeContextPolicy, classify_exec_command


class NoObservationModel:
    def bind(self, **kwargs):
        return self

    async def ainvoke(self, messages):
        raise AssertionError("directory and durable exec reads must not request an Observation")


def test_exec_classifier_accepts_only_pure_top_level_commands():
    assert classify_exec_command("ls -la") == "recent_raw"
    assert classify_exec_command("find src -name '*.py'") == "recent_raw"
    assert classify_exec_command("fd runtime agent") == "recent_raw"
    assert classify_exec_command("rg policy agent/runtime_context.py") == "permanent_raw"
    assert classify_exec_command("grep -n policy README.md") == "permanent_raw"
    assert classify_exec_command("cat README.md") == "permanent_raw"
    assert classify_exec_command("sed -n '1,20p' README.md") == "permanent_raw"
    assert classify_exec_command("head -n 20 README.md") == "permanent_raw"
    assert classify_exec_command("tail -n 20 README.md") == "permanent_raw"
    assert classify_exec_command("apply_patch") == "observation"
    assert classify_exec_command("tee output.txt") == "observation"
    assert classify_exec_command("cp source target") == "observation"
    assert classify_exec_command("mv source target") == "observation"
    assert classify_exec_command("touch output.txt") == "observation"


def test_exec_classifier_treats_uncertain_shell_syntax_as_write_class():
    uncertain = (
        "ls | cat",
        "ls && cat README.md",
        "ls || true",
        "ls; cat README.md",
        "(ls)",
        "ls $(pwd)",
        "ls $PATH",
        'ls "$PATH"',
        "ls > output.txt",
        "unknown-command README.md",
        "ls 'unterminated",
        "",
        None,
    )

    assert all(classify_exec_command(command) == "observation" for command in uncertain)


def test_exec_results_use_the_selected_raw_lifecycle():
    async def run():
        policy = RuntimeContextPolicy(NoObservationModel())
        await policy.record_tool_round(
            1,
            [
                {"id": "list", "name": "exec", "args": {"command": "ls ."}},
                {"id": "read", "name": "exec", "args": {"command": "cat README.md"}},
                {"id": "write", "name": "exec", "args": {"command": "touch marker"}},
            ],
            ["listing", "contents", "written"],
        )

        calls = policy.assemble({}).raw_tool_results[0].calls
        assert [call.lifecycle for call in calls] == [
            "recent_raw",
            "permanent_raw",
            "observation",
        ]

    asyncio.run(run())


def test_exec_directory_results_expire_without_expiring_durable_reads():
    async def run():
        policy = RuntimeContextPolicy(NoObservationModel())
        await policy.record_tool_round(
            1,
            [
                {"id": "listing", "name": "exec", "args": {"command": "ls ."}},
                {"id": "file", "name": "exec", "args": {"command": "cat README.md"}},
            ],
            ["old listing", "old contents"],
        )
        for round_number in range(2, 5):
            await policy.record_tool_round(
                round_number,
                [{"id": f"new-{round_number}", "name": "exec", "args": {"command": "fd ."}}],
                [f"listing {round_number}"],
            )

        calls = [
            call.tool_call_id
            for result in policy.assemble({}).raw_tool_results
            for call in result.calls
        ]
        assert calls == ["file", "new-2", "new-3", "new-4"]

    asyncio.run(run())
