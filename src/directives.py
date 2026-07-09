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

from agent_utils import Feature
from agent_utils.components import CompressToolResults, ExcludeForgotten, MessageRenderer

log = logging.getLogger("agent_zoo_user_plugin.directives")

_PLUGIN_NAME = "azo-plugin-directives"
_DIRECTIVE_RE = re.compile(r"(?<![\w@])@([A-Za-z][A-Za-z0-9_-]*)(?:\b|$)")
_MARKER_ATTR = "_directive_prompt_wrapper"
_PREVIOUS_RENDER_ATTR = "_directive_prompt_previous_render"
_WRAPPER_RENDER_ATTR = "_directive_prompt_render_fn"
_CONFIG_ATTR = "_directive_prompt_config"



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


class DirectivePromptRenderer:
    """Append registered directive text to model rendering only."""

    reads = {"entries"}
    writes = {"entries"}

    def __init__(self, registry: DirectiveRegistry | None = None):
        self.registry = registry or DirectiveRegistry()

    def __call__(self, state):
        self.registry.refresh()
        for entry in getattr(state, "entries", []) or []:
            self._refresh_entry(entry)
        return state

    def _refresh_entry(self, entry) -> None:
        if getattr(entry, "role", "") != "user" or getattr(entry, "system_generated", False):
            self._unwrap_if_current(entry)
            return

        content = self._entry_content(entry)
        configs = self._find_directives(content)
        if not configs:
            self._unwrap_if_current(entry)
            return

        if getattr(entry, _MARKER_ATTR, False) and getattr(entry, _CONFIG_ATTR, None) == configs:
            return

        self._unwrap_if_current(entry)
        previous_render = getattr(entry, "render_fn", None)

        def render(state, _previous=previous_render, _entry=entry, _configs=configs):
            if _previous is None:
                messages = _entry.render_default(state)
            else:
                messages = _previous(state)
            if not messages:
                return messages
            blocks = []
            for name, label, message in _configs:
                blocks.append(f"=== [{label}] ===\nThe user has included the @{name} directive.\n\n{message}")
            block = "\n\n" + "\n\n".join(blocks)
            rendered = []
            appended = False
            for msg in messages:
                if not appended and isinstance(msg, dict) and msg.get("role") == "user":
                    rendered.append({**msg, "content": str(msg.get("content") or "") + block})
                    appended = True
                else:
                    rendered.append(msg)
            return rendered

        entry.render_fn = render
        setattr(entry, _MARKER_ATTR, True)
        setattr(entry, _PREVIOUS_RENDER_ATTR, previous_render)
        setattr(entry, _WRAPPER_RENDER_ATTR, render)
        setattr(entry, _CONFIG_ATTR, configs)

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

    @staticmethod
    def _unwrap_if_current(entry) -> None:
        if not getattr(entry, _MARKER_ATTR, False):
            return
        previous = getattr(entry, _PREVIOUS_RENDER_ATTR, None)
        wrapper = getattr(entry, _WRAPPER_RENDER_ATTR, None)
        if getattr(entry, "render_fn", None) is wrapper:
            entry.render_fn = previous
        for attr in (_MARKER_ATTR, _PREVIOUS_RENDER_ATTR, _WRAPPER_RENDER_ATTR, _CONFIG_ATTR):
            if hasattr(entry, attr):
                delattr(entry, attr)


def register_features(builder, *, session, config):
    builder.add(Feature(
        name="directive_prompts",
        components=[DirectivePromptRenderer()],
        order=[CompressToolResults, DirectivePromptRenderer, ExcludeForgotten, MessageRenderer],
    ))
