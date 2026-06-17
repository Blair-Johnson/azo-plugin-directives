# azo-plugin-directives

Userspace agent-zoo orchestrator plugin that appends directive-specific prompt guidance for user messages beginning with directives such as `@work` or `@goal`.

Install with:

```bash
azo-plugin-install /path/to/azo-plugin-directives
```

Edit `directives.yaml` to customize directive behavior, or set `AZO_DIRECTIVES_REGISTRY` to point at another registry file.
