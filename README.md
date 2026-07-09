# azo-plugin-directives

Userspace agent-zoo orchestrator plugin that appends directive-specific prompt guidance for user messages containing one or more directives such as `@work` or `@goal`.

Install with:

```bash
azo-plugin install /path/to/azo-plugin-directives
```

Install initializes `~/.local/share/agent-zoo/plugin-configs/azo-plugin-directives/directives.yaml` from the default `directives.yaml` if it is missing. Edit that installed config, or set `AZO_DIRECTIVES_REGISTRY` to point at another registry file. Normal uninstall keeps the config; use `azo-plugin uninstall azo-plugin-directives --purge-config` for full removal.
