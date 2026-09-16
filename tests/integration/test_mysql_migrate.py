import json
import os

import pytest

from agent.cli import main


MYSQL_URL = os.environ.get("CODEX_TEST_MYSQL_URL")


@pytest.mark.integration
@pytest.mark.skipif(
    MYSQL_URL is None,
    reason="CODEX_TEST_MYSQL_URL is not configured; skipping MySQL integration test",
)
def test_migrate_initializes_mysql_persistence_and_is_idempotent(monkeypatch, capsys):
    monkeypatch.setenv("CODEX_MYSQL_URL", MYSQL_URL)

    assert main(["migrate"]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first == {"command": "migrate", "status": "completed"}

    assert main(["migrate"]) == 0
    assert json.loads(capsys.readouterr().out) == first
