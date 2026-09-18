# 01: Configuration file entry and validation

**What to build:** An operator can start or resume an Agent run only by supplying an explicit YAML configuration file. Shared defaults and Planner/Executor overrides resolve into complete Component provider configurations, while malformed, incomplete, or unsupported configuration is rejected as a configuration error before model or Tool execution. Database migration remains independent of this file.

**Blocked by:** None (can start immediately).

**Status:** claimed

- [ ] `run` and `resume` require an explicit configuration-file argument, while `migrate` remains configuration-file independent.
- [ ] Shared defaults and component overrides resolve each Planner and Executor Component provider configuration completely.
- [ ] Invalid YAML, unknown fields, invalid values, and absent effective required values return the stable configuration result before work begins.
