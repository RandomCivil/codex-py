# 04: YAML configuration documentation and regression coverage

**What to build:** Operators receive accurate YAML configuration instructions, a shared-default/override example, and plaintext-key permission guidance. The complete CLI workflow is covered by regression tests so documented behavior for configuration, model requests, and resumption remains stable.

**Blocked by:** 02: Component Provider requests and Structured-output; 03: Effective Component provider configuration resume compatibility.

**Status:** ready-for-agent

- [ ] Operator documentation describes explicit YAML configuration, inheritance and overrides, both Structured-output modes, and API-key file-protection responsibility.
- [ ] Documentation no longer instructs operators to configure Planner or Executor model settings through environment variables.
- [ ] Regression coverage demonstrates the documented configuration entry point, per-component request behavior, and resume compatibility without live provider credentials.
