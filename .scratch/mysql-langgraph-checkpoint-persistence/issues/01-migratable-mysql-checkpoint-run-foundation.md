# 01: Migratable MySQL Checkpoint run foundation

**What to build:** An operator can run `agent migrate` against a supported, dedicated local MySQL schema to prepare durable Agent-run storage. The command validates the database version, applies project-owned versioned migrations, initializes the LangGraph MySQL checkpointer schema, and returns stable JSON with the defined configuration outcome when MySQL configuration, schema state, or version is invalid.

**Blocked by:** None (can start immediately).

**Status:** resolved

- [x] `agent migrate` uses only `CODEX_MYSQL_URL`, verifies MySQL 8.0.19 through 9.5, and initializes both the saver-owned schema and the Run registry without an execution command performing implicit migration.
- [x] Repeated migration is idempotent, records project migration history, and integration coverage uses only `CODEX_TEST_MYSQL_URL`, skipping with a clear reason when absent.
- [x] Successful and configuration-failure migration results emit one stable JSON record and the documented exit-code class.

## Answer

Implemented the explicit `agent migrate` foundation with MySQL URL validation, supported-version checks, idempotent project migrations, LangGraph saver setup, stable JSON output, and exit-code mapping. Added a MySQL integration test that skips clearly without `CODEX_TEST_MYSQL_URL`.
