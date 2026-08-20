"""Userspace @directive prompt expansion for the orchestrator pipeline.

A user message that contains registered directives such as ``@work`` is
rendered to the model with extra user-role system-labeled blocks appended.
The stored entry content is not changed, so the appended block stays out of the
TUI transcript projection.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

_PLUGIN_NAME = "azo-plugin-directives"
_AGENT_UTILS_MIN_VERSION = "0.3.0"


class _PluginCompatibilityError(RuntimeError):
    agent_zoo_plugin_compatibility = True


try:
    from agent_utils import Feature, RENDER_CHANNEL_API_VERSION
    from agent_utils.components import (
        CompressToolResults,
        ExcludeForgotten,
        MessageRenderer,
        RenderTransformRegistrar,
    )
    from agent_utils.types import Entry, MODEL_RENDER_CHANNEL
except (ImportError, AttributeError) as exc:
    raise _PluginCompatibilityError(
        f"{_PLUGIN_NAME} requires agent-utils>={_AGENT_UTILS_MIN_VERSION} with "
        "RenderTransformRegistrar, MODEL_RENDER_CHANNEL, and "
        "Entry.register_render_transform; upgrade agent-utils before reloading."
    ) from exc

if (
    RENDER_CHANNEL_API_VERSION != _AGENT_UTILS_MIN_VERSION
    or MODEL_RENDER_CHANNEL != "model"
    or not hasattr(Entry, "register_render_transform")
):
    raise _PluginCompatibilityError(
        f"{_PLUGIN_NAME} requires agent-utils>={_AGENT_UTILS_MIN_VERSION} with "
        "RenderTransformRegistrar, MODEL_RENDER_CHANNEL, and "
        "Entry.register_render_transform; upgrade agent-utils before reloading."
    )

log = logging.getLogger("agent_zoo_user_plugin.directives")
_DIRECTIVE_RE = re.compile(r"(?<![\w@])@([A-Za-z][A-Za-z0-9_-]*)(?:\b|$)")



def _agent_zoo_data_root() -> Path:
    for env_name in ("AGENT_ZOO_HOME", "AGENT_ZOO_STATE_ROOT"):
        env_path = os.environ.get(env_name, "").strip()
        if env_path:
            return Path(env_path).expanduser()
    return Path.home() / ".local" / "share" / "agent-zoo"


def _user_config_dir() -> Path:
    return _agent_zoo_data_root() / "plugin-configs" / _PLUGIN_NAME
def _default_registry_path() -> Path:
    env_path = os.environ.get("AZO_DIRECTIVES_REGISTRY", "").strip()
    if env_path:
        return Path(env_path).expanduser()
    user_config = _user_config_dir() / "directives.yaml"
    if user_config.exists():
        return user_config
    return Path(__file__).resolve().parents[1] / "directives.yaml"


class DirectiveRegistry:
    """Hot-refreshing YAML registry for directive prompt blocks."""

    def __init__(self, path: Path | None = None):
        self.path = path or _default_registry_path()
        self._mtime_ns: int | None = None
        self._directives: dict[str, dict[str, Any]] = {}

    def get(self, name: str) -> dict[str, Any] | None:
        self.refresh()
        return self._directives.get(name.lower())

    def refresh(self) -> None:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            if self._mtime_ns is not None or self._directives:
                log.warning("Directive registry disappeared: %s", self.path)
            self._mtime_ns = None
            self._directives = {}
            return
        except OSError as exc:
            log.warning("Could not stat directive registry %s: %s", self.path, exc)
            return

        if self._mtime_ns == stat.st_mtime_ns:
            return

        try:
            raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
            self._directives = self._normalize(raw)
            self._mtime_ns = stat.st_mtime_ns
        except Exception as exc:  # keep last good registry on parse errors
            log.warning("Could not load directive registry %s: %s", self.path, exc)

    @staticmethod
    def _normalize(raw: Any) -> dict[str, dict[str, Any]]:
        if not isinstance(raw, dict):
            return {}
        directives = raw.get("directives", raw)
        if not isinstance(directives, dict):
            return {}

        normalized: dict[str, dict[str, Any]] = {}
        for key, value in directives.items():
            name = str(key).lstrip("@").strip().lower()
            if not name:
                continue
            if isinstance(value, str):
                entry = {"message": value, "enabled": True, "label": "system"}
            elif isinstance(value, dict):
                entry = dict(value)
            else:
                continue
            if entry.get("enabled", True) is False:
                continue
            message = str(entry.get("message") or entry.get("prompt") or "").strip()
            if not message:
                continue
            label = str(entry.get("label") or "system").strip() or "system"
            normalized[name] = {"message": message, "label": label}
            for alias in entry.get("aliases") or []:
                alias_name = str(alias).lstrip("@").strip().lower()
                if alias_name:
                    normalized[alias_name] = normalized[name]
        return normalized


class DirectivePromptRenderer(RenderTransformRegistrar):
    """Register directive prompt expansion for the model channel."""

    reads = {"entries"}
    writes = {"entries"}

    def __init__(self, registry: DirectiveRegistry | None = None):
        self.registry = registry or DirectiveRegistry()

    def __call__(self, state):
        """Refresh configuration and append fresh entry-local transforms."""
        self.registry.refresh()
        for entry in getattr(state, "entries", []) or []:
            if (
                getattr(entry, "role", "") != "user"
                or getattr(entry, "system_generated", False)
            ):
                continue

            configs = self._find_directives(self._entry_content(entry))
            if not configs:
                continue

            entry.register_render_transform(
                MODEL_RENDER_CHANNEL,
                self._make_transform(configs),
                name="directive prompts",
            )
        return state

    @staticmethod
    def _make_transform(configs):
        """Create a pure copy-on-write transform for one entry's directives."""
        blocks = [
            f"=== [{label}] ===\n"
            f"The user has included the @{name} directive.\n\n{message}"
            for name, label, message in configs
        ]
        block = "\n\n" + "\n\n".join(blocks)

        def transform(messages, _state):
            if not messages:
                return messages

            rendered = []
            appended = False
            for message in messages:
                if (
                    not appended
                    and isinstance(message, dict)
                    and message.get("role") == "user"
                ):
                    rendered.append({
                        **message,
                        "content": str(message.get("content") or "") + block,
                    })
                    appended = True
                else:
                    rendered.append(message)
            return rendered

        return transform

    @staticmethod
    def _entry_content(entry) -> str:
        messages = getattr(entry, "messages", None) or []
        if not messages or not isinstance(messages[0], dict):
            return ""
        return str(messages[0].get("content") or "")

    def _find_directives(self, content: str) -> tuple[tuple[str, str, str], ...]:
        configs: list[tuple[str, str, str]] = []
        seen_prompts: set[tuple[str, str]] = set()
        for match in _DIRECTIVE_RE.finditer(self._user_message_body(content)):
            name = match.group(1).lower()
            directive = self.registry.get(name)
            if not directive:
                continue
            prompt_key = (directive["label"], directive["message"])
            if prompt_key in seen_prompts:
                continue
            configs.append((name, *prompt_key))
            seen_prompts.add(prompt_key)
        return tuple(configs)

    @staticmethod
    def _user_message_body(content: str) -> str:
        lines = content.splitlines()
        index = 0
        while index < len(lines) and not lines[index].strip():
            index += 1
        if index < len(lines) and lines[index].strip() == "--- USER ---":
            index += 1
            end = index
            while end < len(lines) and lines[end].strip() != "--- END USER ---":
                end += 1
            return "\n".join(lines[index:end])
        return content


def register_features(builder, *, session, config):
    builder.add(Feature(
        name="directive_prompts",
        components=[DirectivePromptRenderer()],
        order=[CompressToolResults, DirectivePromptRenderer, ExcludeForgotten, MessageRenderer],
    ))
