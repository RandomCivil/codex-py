# 03: Exclusive Agent-run ownership and configuration compatibility

**What to build:** An operator can safely invoke the Agent-run CLI from competing processes. Exactly one owner holds a renewable MySQL lease for an active Agent run; competing work receives `run busy`, and a run can be resumed after lease expiry. Resume also verifies that the non-secret model and MCP configuration still matches the run's saved fingerprint, refusing semantic drift.

**Blocked by:** 02: Resumable normal Agent run CLI.

**Status:** resolved

- [x] The Run registry persists Agent-run identity, top-level thread identity, terminal status, non-secret configuration snapshot/fingerprint, and lease state separately from LangGraph saver tables.
- [x] A 60-second lease is renewed every 15 seconds and every lease decision uses MySQL server time; valid lease contention returns the documented busy result and expiry allows safe takeover.
- [x] Resume rejects changed configuration fingerprints without execution, and local-MySQL integration tests cover contention, expiry, takeover, and configuration mismatch.

## Answer

Implemented the project-owned MySQL Agent-run registry with UUID-backed run/thread metadata, canonical non-secret configuration snapshots and SHA-256 fingerprints, 60-second MySQL-server-time leases, 15-second renewal, contention/expiry takeover behavior, and terminal result persistence. Integrated registry acquisition and compatibility checks into `agent run` and `agent resume`; CLI now emits dedicated busy and configuration outcomes and validates UUIDv4 run IDs. Added controlled registry/CLI tests and an explicitly configured local-MySQL integration test covering contention, expiry takeover, and configuration mismatch.
