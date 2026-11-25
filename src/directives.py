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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_PLUGIN_NAME = "azo-plugin-directives"
_AGENT_UTILS_MIN_VERSION = "0.4.0"


class _PluginCompatibilityError(RuntimeError):
    agent_zoo_plugin_compatibility = True


def _version_tuple(raw: str) -> tuple[int, ...]:
    return tuple(int(part) for part in str(raw).split(".") if part.isdigit())


try:
    from agent_utils import RENDER_CHANNEL_API_VERSION
    from agent_utils.core.events import (
        BeforeLLM,
        Event,
        IoCompleted,
        IoFailed,
        RenderRequested,
        SessionRestored,
        SessionStarted,
    )
    from agent_utils.core.io import FsRead, FsStat, OpId
    from agent_utils.core.pipeline import EmitFn, Feature
    from agent_utils.core.state import Entry, MODEL_RENDER_CHANNEL, State
    from agent_utils.features.core import (
        IoFence,
        MessageRenderer,
        RenderPlanFinalizer,
        RenderPlanInitializer,
        RenderTransformRegistrar,
    )
except (ImportError, AttributeError) as exc:
    raise _PluginCompatibilityError(
        f"{_PLUGIN_NAME} requires agent-utils>={_AGENT_UTILS_MIN_VERSION} with "
        "RenderTransformRegistrar, MODEL_RENDER_CHANNEL, and "
        "Entry.append_render_transform; upgrade agent-utils before reloading."
    ) from exc

if (
    _version_tuple(RENDER_CHANNEL_API_VERSION) < _version_tuple(_AGENT_UTILS_MIN_VERSION)
    or MODEL_RENDER_CHANNEL != "model"
    or not hasattr(Entry, "append_render_transform")
):
    raise _PluginCompatibilityError(
        f"{_PLUGIN_NAME} requires agent-utils>={_AGENT_UTILS_MIN_VERSION} with "
        "RenderTransformRegistrar, MODEL_RENDER_CHANNEL, and "
        "Entry.append_render_transform; upgrade agent-utils before reloading."
    )

from agent_zoo.durability import PluginStorage

log = logging.getLogger("agent_zoo_user_plugin.directives")
_DIRECTIVE_RE = re.compile(r"(?<![\w@])@([A-Za-z][A-Za-z0-9_-]*)(?:\b|$)")
_RUNTIME_KEY = "directives"
_IO_TAG = "directives.refresh"


def _user_config_dir() -> Path:
    return PluginStorage(_PLUGIN_NAME).config_dir


def _default_registry_path() -> Path:
    env_path = os.environ.get("AZO_DIRECTIVES_REGISTRY", "").strip()
    if env_path:
        return Path(env_path).expanduser()
    user_config = _user_config_dir() / "directives.yaml"
    if user_config.exists():
        return user_config
    return Path(__file__).resolve().parents[1] / "directives.yaml"


@dataclass
class _DirectivesRuntime:
    directives: dict[str, dict[str, Any]] = field(default_factory=dict)
    mtime_ns: int | None = None
    stat_op: OpId | None = None
    read_op: OpId | None = None
    pending_mtime_ns: int | None = None
    hydrating_readiness: bool = False
    initialized: bool = False


class DirectiveRegistry:
    """YAML directive configuration loader (preloaded at build time; refreshed via FsDriver)."""

    def __init__(self, path: Path | None = None, *, preload: bool = True) -> None:
        self.path = path or _default_registry_path()
        self._mtime_ns: int | None = None
        self._directives: dict[str, dict[str, Any]] = {}
        if preload:
            self.refresh()

    def get(self, name: str) -> dict[str, Any] | None:
        """Pure in-memory lookup (never performs disk I/O)."""
        return self._directives.get(name.lower())

    @property
    def directives(self) -> dict[str, dict[str, Any]]:
        return self._directives

    @property
    def mtime_ns(self) -> int | None:
        return self._mtime_ns

    def load_text(self, text: str, *, mtime_ns: int | None = None) -> bool:
        """Parse YAML text purely in memory; keeps last good directives on parse error."""
        try:
            raw = yaml.safe_load(text) or {}
            self._directives = self._normalize(raw)
            if mtime_ns is not None:
                self._mtime_ns = mtime_ns
            return True
        except Exception as exc:
            log.warning("Could not load directive registry %s: %s", self.path, exc)
            return False

    def clear_missing(self) -> None:
        if self._mtime_ns is not None or self._directives:
            log.warning("Directive registry disappeared: %s", self.path)
        self._mtime_ns = None
        self._directives = {}

    def refresh(self) -> None:
        """Host-side synchronous load (called only at build/init time, never inside handle())."""
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            self.clear_missing()
            return
        except OSError as exc:
            log.warning("Could not stat directive registry %s: %s", self.path, exc)
            return

        if self._mtime_ns == stat.st_mtime_ns:
            return

        try:
            raw_text = self.path.read_text(encoding="utf-8")
        except OSError as exc:
            log.warning("Could not read directive registry %s: %s", self.path, exc)
            return
        self.load_text(raw_text, mtime_ns=stat.st_mtime_ns)

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


class DirectiveRefreshHandler:
    """Pure Sans-I/O hot-reloader for ``directives.yaml`` via ``FsDriver``."""

    handles = (SessionStarted, SessionRestored, BeforeLLM, IoCompleted, IoFailed)

    def __init__(self, registry: DirectiveRegistry) -> None:
        self.registry = registry

    def _runtime(self, state: State) -> _DirectivesRuntime:
        rt = state.runtime(_RUNTIME_KEY, _DirectivesRuntime)
        if not rt.initialized:
            rt.directives = dict(self.registry.directives)
            rt.mtime_ns = self.registry.mtime_ns
            rt.initialized = True
        return rt

    def _submit_stat(
        self,
        state: State,
        rt: _DirectivesRuntime,
        *,
        hold_readiness: str | bool | None = None,
    ) -> None:
        if rt.stat_op is not None or rt.read_op is not None:
            return
        io_table = getattr(state, "io", None)
        if io_table is None:
            return
        rt.stat_op = io_table.submit(
            FsStat(str(self.registry.path)),
            tag=_IO_TAG,
            hold_readiness=hold_readiness,
        )

    def handle(self, event: Event, state: State, emit: EmitFn) -> None:
        rt = self._runtime(state)
        if isinstance(event, (SessionStarted, SessionRestored, BeforeLLM)):
            if isinstance(event, SessionRestored):
                rt.stat_op = None
                rt.read_op = None
                rt.hydrating_readiness = True
                self._submit_stat(state, rt, hold_readiness="directives")
                if rt.stat_op is None:
                    rt.hydrating_readiness = False
                return
            self._submit_stat(state, rt)
            return

        if isinstance(event, IoCompleted) and event.tag == _IO_TAG:
            if event.op_id == rt.stat_op:
                rt.stat_op = None
                hold_readiness = "directives" if rt.hydrating_readiness else None
                info = event.value
                exists = (
                    info.get("exists", True)
                    if isinstance(info, dict)
                    else getattr(info, "exists", True)
                )
                if not exists:
                    rt.directives = {}
                    rt.mtime_ns = None
                    rt.hydrating_readiness = False
                    return
                mtime_ns = (
                    info.get("mtime_ns")
                    if isinstance(info, dict)
                    else getattr(info, "mtime_ns", None)
                )
                if mtime_ns is not None and mtime_ns == rt.mtime_ns:
                    rt.hydrating_readiness = False
                    return
                rt.pending_mtime_ns = mtime_ns
                rt.read_op = state.io.submit(
                    FsRead(str(self.registry.path)),
                    tag=_IO_TAG,
                    hold_readiness=hold_readiness,
                )
                return

            if event.op_id == rt.read_op:
                rt.read_op = None
                rt.hydrating_readiness = False
                text = (
                    event.value.decode("utf-8")
                    if isinstance(event.value, bytes)
                    else str(event.value or "")
                )
                try:
                    raw = yaml.safe_load(text) or {}
                    rt.directives = DirectiveRegistry._normalize(raw)
                    rt.mtime_ns = rt.pending_mtime_ns
                    emit(RenderRequested(channels=(MODEL_RENDER_CHANNEL,)))
                except Exception as exc:
                    log.warning("Could not parse directive registry %s: %s", self.registry.path, exc)
                return

        if isinstance(event, IoFailed) and event.tag == _IO_TAG:
            if event.op_id == rt.stat_op:
                rt.stat_op = None
                rt.hydrating_readiness = False
                if isinstance(event.error, FileNotFoundError):
                    rt.directives = {}
                    rt.mtime_ns = None
            elif event.op_id == rt.read_op:
                rt.read_op = None
                rt.hydrating_readiness = False


class DirectivePromptRenderer(RenderTransformRegistrar):
    """Register directive prompt expansion for the model channel (pure Sans-I/O)."""

    reads = {"entries"}
    writes = {"entries"}

    def __init__(self, registry: DirectiveRegistry | None = None) -> None:
        self.registry = registry or DirectiveRegistry()

    def _active_directives(self, state: State) -> dict[str, dict[str, Any]]:
        runtime_fn = getattr(state, "runtime", None)
        if callable(runtime_fn):
            rt = runtime_fn(_RUNTIME_KEY, _DirectivesRuntime)
            if rt.initialized:
                return rt.directives
        return self.registry.directives

    def handle(self, event: Event, state: State, emit: EmitFn) -> None:
        """Append fresh entry-local transforms from in-memory directives."""
        del emit
        assert isinstance(event, RenderRequested)
        if event.channels is not None and MODEL_RENDER_CHANNEL not in event.channels:
            return
        active = self._active_directives(state)
        for entry in getattr(state, "entries", []) or []:
            if (
                getattr(entry, "role", "") != "user"
                or getattr(entry, "system_generated", False)
            ):
                continue

            configs = self._find_directives(self._entry_content(entry), active)
            if not configs:
                continue

            entry.append_render_transform(
                MODEL_RENDER_CHANNEL,
                self._make_transform(configs),
                name="directive prompts",
            )

    @staticmethod
    def _make_transform(configs: tuple[tuple[str, str, str], ...]):
        """Create a pure copy-on-write transform for one entry's directives."""
        blocks = [
            f"=== [{label}] ===\n"
            f"The user has included the @{name} directive.\n\n{message}"
            for name, label, message in configs
        ]
        block = "\n\n" + "\n\n".join(blocks)

        def transform(
            messages: list[dict[str, Any]],
            _state: State,
        ) -> list[dict[str, Any]]:
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
                    rendered.append(
                        {
                            **message,
                            "content": str(message.get("content") or "") + block,
                        }
                    )
                    appended = True
                else:
                    rendered.append(message)
            return rendered

        return transform

    @staticmethod
    def _entry_content(entry: Any) -> str:
        messages = getattr(entry, "messages", None) or []
        if not messages or not isinstance(messages[0], dict):
            return ""
        return str(messages[0].get("content") or "")

    def _find_directives(
        self,
        content: str,
        active: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[tuple[str, str, str], ...]:
        table = self.registry.directives if active is None else active
        configs: list[tuple[str, str, str]] = []
        seen_prompts: set[tuple[str, str]] = set()
        for match in _DIRECTIVE_RE.finditer(self._user_message_body(content)):
            name = match.group(1).lower()
            directive = table.get(name)
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


def register_features(builder: Any, *, config: Any) -> None:
    del config
    registry = DirectiveRegistry()
    builder.add(
        Feature(
            name="directive_prompts",
            components=[
                DirectiveRefreshHandler(registry),
                DirectivePromptRenderer(registry),
            ],
            order=[
                IoFence,
                DirectiveRefreshHandler,
                RenderPlanInitializer,
                DirectivePromptRenderer,
                RenderPlanFinalizer,
                MessageRenderer,
            ],
        )
    )
