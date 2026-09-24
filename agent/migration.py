"""Explicit schema migration for Agent persistence."""

import asyncio
import re
from collections.abc import Callable
from contextlib import asynccontextmanager
from urllib.parse import unquote, urlparse


MIN_MYSQL_VERSION = (8, 0, 19)
MAX_MYSQL_VERSION = (9, 5, 99)
PROJECT_SCHEMA_VERSION = 3


class MigrationConfigurationError(Exception):
    """The database cannot be used by the persistence layer."""


def migrate_database(
    url: str,
    *,
    connection_factory: Callable | None = None,
    saver_factory: Callable | None = None,
) -> None:
    asyncio.run(
        _migrate_database(
            url,
            connection_factory=connection_factory,
            saver_factory=saver_factory,
        )
    )


async def _migrate_database(
    url: str,
    *,
    connection_factory: Callable | None,
    saver_factory: Callable | None,
) -> None:
    connection_factory = connection_factory or _connection_pool
    parsed = _parse_url(url)
    async with connection_factory(parsed) as pool:
        async with pool.acquire() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute("SELECT VERSION()")
                version = (await cursor.fetchone())[0]
        validate_mysql_version(version)
        async with pool.acquire() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS codex_migration_history (
                        version INT NOT NULL PRIMARY KEY,
                        applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                await cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_runs (
                        run_id CHAR(36) NOT NULL PRIMARY KEY,
                        thread_id CHAR(36) NOT NULL,
                        status VARCHAR(16) NOT NULL,
                        config_snapshot JSON NOT NULL,
                        config_fingerprint CHAR(64) NOT NULL,
                        terminal_result JSON NULL,
                        lease_owner CHAR(36) NULL,
                        lease_expires_at DATETIME(6) NULL,
                        created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
                        updated_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
                            ON UPDATE CURRENT_TIMESTAMP(6)
                    )
                    """
                )
                await cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS step_recovery_attempts (
                        record_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
                        run_id CHAR(36) NOT NULL,
                        revision TINYINT UNSIGNED NOT NULL,
                        step_id VARCHAR(255) NOT NULL,
                        attempt INT UNSIGNED NOT NULL,
                        status VARCHAR(16) NOT NULL,
                        result TEXT NULL,
                        error TEXT NULL,
                        completion_evidence TEXT NULL,
                        context_update JSON NULL,
                        recovery BOOLEAN NOT NULL DEFAULT FALSE,
                        common_version BIGINT UNSIGNED NOT NULL,
                        created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
                        PRIMARY KEY (record_id),
                        UNIQUE KEY step_recovery_event (
                            run_id, revision, step_id, attempt, status
                        ),
                        INDEX step_recovery_run_order (run_id, revision, step_id, attempt)
                    )
                    """
                )
                await cursor.execute(
                    """
                    SELECT COUNT(*) FROM information_schema.columns
                    WHERE table_schema=DATABASE()
                      AND table_name='step_recovery_attempts'
                      AND column_name='completion_evidence'
                    """
                )
                if (await cursor.fetchone())[0] == 0:
                    await cursor.execute(
                        "ALTER TABLE step_recovery_attempts ADD COLUMN completion_evidence TEXT NULL"
                    )
                await cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversations (
                        conv_id CHAR(36) NOT NULL PRIMARY KEY,
                        created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
                    )
                    """
                )
                await cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversation_turns (
                        conv_id CHAR(36) NOT NULL,
                        sequence INT UNSIGNED NOT NULL,
                        run_id CHAR(36) NOT NULL,
                        execution_mode VARCHAR(32) NOT NULL,
                        user_input TEXT NOT NULL,
                        status VARCHAR(16) NOT NULL,
                        answer TEXT NULL,
                        error TEXT NULL,
                        created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
                        updated_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
                            ON UPDATE CURRENT_TIMESTAMP(6),
                        PRIMARY KEY (conv_id, sequence),
                        UNIQUE KEY conversation_turn_run (run_id),
                        INDEX conversation_active (conv_id, status),
                        CONSTRAINT conversation_turn_conversation
                            FOREIGN KEY (conv_id) REFERENCES conversations(conv_id)
                    )
                    """
                )
                await cursor.execute(
                    """
                    INSERT INTO codex_migration_history (version)
                    VALUES (%s)
                    ON DUPLICATE KEY UPDATE version=VALUES(version)
                    """,
                    (PROJECT_SCHEMA_VERSION,),
                )
            await connection.commit()

    saver_factory = saver_factory or _saver
    async with saver_factory(url) as saver:
        await saver.setup()


async def ensure_schema_initialized(pool) -> None:
    """Reject execution before both project and saver schemas are initialized."""
    try:
        async with pool.acquire() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "SELECT version FROM codex_migration_history WHERE version=%s",
                    (PROJECT_SCHEMA_VERSION,),
                )
                if await cursor.fetchone() is None:
                    raise MigrationConfigurationError("persistence schema is not initialized; run agent migrate")
                await cursor.execute("SELECT 1 FROM agent_runs LIMIT 1")
                await cursor.execute("SELECT 1 FROM step_recovery_attempts LIMIT 1")
                await cursor.execute("SELECT 1 FROM conversations LIMIT 1")
                await cursor.execute("SELECT 1 FROM conversation_turns LIMIT 1")
                # The third-party saver records its completed setup in this table.
                await cursor.execute("SELECT 1 FROM checkpoint_migrations LIMIT 1")
    except MigrationConfigurationError:
        raise
    except Exception as exc:
        raise MigrationConfigurationError(
            "persistence schema is not initialized; run agent migrate"
        ) from exc


@asynccontextmanager
async def _connection_pool(parsed):
    try:
        import aiomysql
    except ImportError as exc:  # pragma: no cover - packaging failure
        raise MigrationConfigurationError(
            "langgraph-checkpoint-mysql[aiomysql] is not installed"
        ) from exc
    pool = await aiomysql.create_pool(**parsed, autocommit=True)
    try:
        yield pool
    finally:
        pool.close()
        await pool.wait_closed()


@asynccontextmanager
async def _saver(url):
    try:
        from langgraph.checkpoint.mysql.aio import AIOMySQLSaver
    except ImportError as exc:  # pragma: no cover - packaging failure
        raise MigrationConfigurationError(
            "langgraph-checkpoint-mysql[aiomysql] is not installed"
        ) from exc
    async with AIOMySQLSaver.from_conn_string(url) as saver:
        yield saver


def _parse_url(url: str) -> dict[str, object]:
    parsed = urlparse(url)
    if parsed.scheme not in {"mysql", "mysql+aiomysql"} or not parsed.hostname:
        raise MigrationConfigurationError("CODEX_MYSQL_URL must be a MySQL URL")
    if not parsed.path.strip("/"):
        raise MigrationConfigurationError("CODEX_MYSQL_URL must name a database")
    return {
        "host": parsed.hostname,
        "port": parsed.port or 3306,
        "user": unquote(parsed.username or ""),
        "password": unquote(parsed.password or ""),
        "db": parsed.path.lstrip("/"),
    }


def _mysql_version(raw: str) -> tuple[int, int, int]:
    match = re.search(r"(?<!\d)(\d+)\.(\d+)\.(\d+)", raw)
    if not match:
        raise MigrationConfigurationError("could not determine MySQL version")
    return tuple(map(int, match.groups()))


def validate_mysql_version(raw: str) -> tuple[int, int, int]:
    """Return a supported MySQL version or explain why it cannot be used."""
    if "mariadb" in raw.lower():
        raise MigrationConfigurationError("unsupported MySQL version: " + raw)
    version = _mysql_version(raw)
    if not MIN_MYSQL_VERSION <= version <= MAX_MYSQL_VERSION:
        raise MigrationConfigurationError("unsupported MySQL version: " + raw)
    return version
