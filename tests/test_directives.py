from __future__ import annotations

import os

from agent_utils.core.events import RenderRequested
from agent_utils.core.pipeline import PipelineBuilder
from agent_utils.core.session import Session
from agent_utils.core.state import Entry, MODEL_RENDER_CHANNEL, State
from agent_utils.features.core import (
    CompressToolResults,
    ExcludeForgotten,
    MessageRenderer,
    RenderPlanFinalizer,
    RenderPlanInitializer,
    RenderTransformRegistrar,
    core_feature,
)
from agent_zoo.pipelines.default import build_default_builder

from directives import DirectivePromptRenderer, DirectiveRegistry, register_features


def _noop(_event):
    return None


def write_registry(path, text: str, *, bump: bool = False) -> None:
    path.write_text(text, encoding="utf-8")
    if bump:
        current = path.stat().st_mtime_ns
        os.utime(path, ns=(current + 1, current + 1))


def make_state(content: str) -> State:
    state = State()
    state.entries = [
        Entry(messages=[{"role": "user", "content": content}], index=0, step=0)
    ]
    return state


def prepare(state: State, registry: DirectiveRegistry) -> None:
    event = RenderRequested()
    RenderPlanInitializer().handle(event, state, _noop)
    DirectivePromptRenderer(registry).handle(event, state, _noop)
    RenderPlanFinalizer().handle(event, state, _noop)


def render(state: State, *, channel: str = MODEL_RENDER_CHANNEL) -> list[dict]:
    MessageRenderer(channel=channel).handle(RenderRequested(), state, _noop)
    return state.rendered_messages


def test_repeated_preparation_does_not_accumulate_transforms_or_output(tmp_path):
    registry_path = tmp_path / "directives.yaml"
    write_registry(
        registry_path,
        "directives:\n  work:\n    message: do the work\n",
    )
    registry = DirectiveRegistry(registry_path)
    state = make_state("please @work")

    prepare(state, registry)
    first = render(state)
    prepare(state, registry)
    second = render(state)

    assert first == second
    assert first[0]["content"].count("The user has included the @work directive.") == 1
    assert state.entries[0].render_transform_names() == ("directive prompts",)


def test_directive_order_deduplication_and_removal(tmp_path):
    registry_path = tmp_path / "directives.yaml"
    write_registry(
        registry_path,
        """directives:
  first:
    label: system
    message: first guidance
  duplicate:
    label: system
    message: first guidance
  second:
    label: developer
    message: second guidance
""",
    )
    registry = DirectiveRegistry(registry_path)
    state = make_state("@second then @first and @duplicate")

    prepare(state, registry)
    content = render(state)[0]["content"]
    assert content.index("second guidance") < content.index("first guidance")
    assert content.count("first guidance") == 1
    assert content.count("The user has included the @first directive.") == 1
    assert content.count("The user has included the @duplicate directive.") == 0

    state.entries[0].messages[0]["content"] = "no registered directives"
    prepare(state, registry)
    rendered = render(state)
    assert rendered == state.entries[0].messages
    assert state.entries[0].render_transform_names() == ()


def test_parse_error_keeps_last_good_configuration(tmp_path):
    registry_path = tmp_path / "directives.yaml"
    write_registry(
        registry_path,
        "directives:\n  work:\n    message: last good guidance\n",
    )
    registry = DirectiveRegistry(registry_path)
    state = make_state("@work")

    prepare(state, registry)
    assert "last good guidance" in render(state)[0]["content"]

    write_registry(registry_path, "directives: [invalid", bump=True)
    prepare(state, registry)
    rendered = render(state)
    assert "last good guidance" in rendered[0]["content"]
    assert state.entries[0].messages[0]["content"] == "@work"


def test_directives_render_only_on_explicit_model_channel_and_isolate_storage(tmp_path):
    registry_path = tmp_path / "directives.yaml"
    write_registry(
        registry_path,
        "directives:\n  work:\n    message: model-only guidance\n",
    )
    registry = DirectiveRegistry(registry_path)
    state = make_state("@work")
    prepare(state, registry)

    rendered = render(state, channel=MODEL_RENDER_CHANNEL)
    assert "model-only guidance" in rendered[0]["content"]
    assert rendered[0] is not state.entries[0].messages[0]
    assert state.entries[0].messages[0]["content"] == "@work"


def test_registrar_is_public_model_transform_and_pipeline_builds_in_order():
    session = Session()
    builder = PipelineBuilder().add(core_feature())
    register_features(builder, session=session, config={})
    pipeline = builder.build(suppress_warnings=True)
    types = [type(component) for component in pipeline]

    registrar = next(
        component for component in pipeline if isinstance(component, DirectivePromptRenderer)
    )
    assert isinstance(registrar, RenderTransformRegistrar)
    renderer_idx = next(
        idx for idx, component in enumerate(pipeline) if isinstance(component, MessageRenderer)
    )
    assert types.index(CompressToolResults) < types.index(DirectivePromptRenderer)
    assert types.index(DirectivePromptRenderer) < types.index(ExcludeForgotten)
    assert types.index(ExcludeForgotten) < renderer_idx
    assert registrar.reads == {"entries"}
    assert MODEL_RENDER_CHANNEL == "model"


def test_user_wrapper_payload_parsing_is_preserved(tmp_path):
    registry_path = tmp_path / "directives.yaml"
    write_registry(
        registry_path,
        "directives:\n  work:\n    message: parsed guidance\n",
    )
    registry = DirectiveRegistry(registry_path)
    state = make_state("--- USER ---\nplease @work\n--- END USER ---\nignored @work")
    prepare(state, registry)

    rendered = state.entries[0].render(state, channel=MODEL_RENDER_CHANNEL)
    assert "parsed guidance" in rendered[0]["content"]
    assert rendered[0]["content"].count("parsed guidance") == 1
    assert state.entries[0].messages[0]["content"].startswith("--- USER ---")


def test_composes_onto_default_builder():
    session = Session()
    builder = build_default_builder(
        session,
        config={},
        include_plugins=False,
        include_llm_drivers=False,
    )
    register_features(builder, session=session, config={})
    pipeline = builder.build()
    assert any(isinstance(comp, DirectivePromptRenderer) for comp in pipeline)
