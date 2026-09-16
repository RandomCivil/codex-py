import pytest

from agent.migration import MigrationConfigurationError, validate_mysql_version


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("8.0.19", (8, 0, 19)),
        ("9.5.2-MySQL Community Server", (9, 5, 2)),
    ],
)
def test_mysql_version_accepts_supported_server_versions(raw, expected):
    assert validate_mysql_version(raw) == expected


def test_mysql_version_rejects_unparseable_server_version():
    with pytest.raises(MigrationConfigurationError, match="could not determine MySQL version"):
        validate_mysql_version("unknown")


@pytest.mark.parametrize("raw", ["8.0.18", "9.6.0", "10.11.6-MariaDB"])
def test_mysql_version_rejects_unsupported_servers(raw):
    with pytest.raises(MigrationConfigurationError, match="unsupported MySQL version"):
        validate_mysql_version(raw)
