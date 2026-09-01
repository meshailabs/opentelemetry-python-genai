# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Agent lifecycle telemetry captured from unmodified LangGraph applications.

Every id asserted here comes from LangGraph itself: interrupt ids from the
``Interrupt`` objects the graph returns, checkpoint ids from the checkpointer's
own return values.
"""

from __future__ import annotations

import asyncio
from typing import Any, TypedDict

import pytest

InMemorySaver = pytest.importorskip(
    "langgraph.checkpoint.memory"
).InMemorySaver
pytest.importorskip("langgraph.callbacks")
_graph = pytest.importorskip("langgraph.graph")
_types = pytest.importorskip("langgraph.types")
END = _graph.END
START = _graph.START
StateGraph = _graph.StateGraph
Command = _types.Command
interrupt = _types.interrupt

PAUSED = "gen_ai.agent.paused"
CHECKPOINTED = "gen_ai.agent.checkpointed"
RESUMED = "gen_ai.agent.resumed"
LIFECYCLE_EVENTS = (PAUSED, CHECKPOINTED, RESUMED)


class _State(TypedDict, total=False):
    value: str
    approval: str


def _build_graph():
    def start(_state: _State) -> _State:
        return {"value": "prepared"}

    def approval(_state: _State) -> _State:
        return {"approval": str(interrupt({"question": "approve?"}))}

    def finish(_state: _State) -> _State:
        return {"value": "submitted"}

    builder = StateGraph(_State)
    builder.add_node("start", start)
    builder.add_node("approval", approval)
    builder.add_node("finish", finish)
    builder.add_edge(START, "start")
    builder.add_edge("start", "approval")
    builder.add_edge("approval", "finish")
    builder.add_edge("finish", END)
    return builder


def _events(log_exporter, event_name: str) -> list[Any]:
    return [
        item.log_record
        for item in log_exporter.get_finished_logs()
        if item.log_record.event_name == event_name
    ]


def _workflow_spans(span_exporter) -> list[Any]:
    return [
        span
        for span in span_exporter.get_finished_spans()
        if span.attributes
        and span.attributes.get("gen_ai.operation.name") == "invoke_workflow"
    ]


def test_interrupt_and_resume_emit_correlated_lifecycle_events(
    start_instrumentation,
    log_exporter,
    span_exporter,
):
    graph = _build_graph().compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "lifecycle-test"}}

    paused_result = graph.invoke({"value": "new"}, config=config)
    interrupts = paused_result["__interrupt__"]
    assert len(interrupts) == 1
    real_interrupt_id = interrupts[0].id

    # paused carries the id LangGraph minted for this interrupt.
    paused = _events(log_exporter, PAUSED)
    assert len(paused) == 1
    assert paused[0].attributes["gen_ai.agent.pause.id"] == real_interrupt_id
    pause_checkpoint = paused[0].attributes["gen_ai.agent.checkpoint.id"]
    # No pause reason: LangGraph does not report why execution paused.
    assert "gen_ai.agent.pause.reason" not in paused[0].attributes
    # No execution id: LangGraph has no id spanning suspend and resume.
    assert "gen_ai.agent.execution.id" not in paused[0].attributes

    # LangGraph writes one checkpoint per superstep, so checkpointed is a
    # per-step record: the input checkpoint plus one per completed superstep.
    first_run_checkpoints = [
        record.attributes["gen_ai.agent.checkpoint.id"]
        for record in _events(log_exporter, CHECKPOINTED)
    ]
    assert len(first_run_checkpoints) == 3
    assert len(set(first_run_checkpoints)) == 3
    # The graph pauses at the checkpoint it last persisted.
    assert pause_checkpoint == first_run_checkpoints[-1]

    resumed_result = graph.invoke(Command(resume="approved"), config=config)
    assert resumed_result["approval"] == "approved"

    resumed = _events(log_exporter, RESUMED)
    assert len(resumed) == 1
    assert (
        resumed[0].attributes["gen_ai.agent.resumed_from.type"] == "checkpoint"
    )
    # The resume continues from exactly the checkpoint the pause reported.
    assert (
        resumed[0].attributes["gen_ai.agent.resumed_from.id"]
        == pause_checkpoint
    )

    second_run_checkpoints = [
        record.attributes["gen_ai.agent.checkpoint.id"]
        for record in _events(log_exporter, CHECKPOINTED)
    ][len(first_run_checkpoints) :]
    assert len(second_run_checkpoints) == 2

    # Each event is correlated with the workflow span of its own invoke call.
    workflow_spans = _workflow_spans(span_exporter)
    assert len(workflow_spans) == 2
    paused_run, resumed_run = workflow_spans
    for record in _events(log_exporter, PAUSED) + [
        record
        for record in _events(log_exporter, CHECKPOINTED)
        if record.attributes["gen_ai.agent.checkpoint.id"]
        in first_run_checkpoints
    ]:
        assert record.trace_id == paused_run.context.trace_id
        assert record.span_id == paused_run.context.span_id
    for record in _events(log_exporter, RESUMED) + [
        record
        for record in _events(log_exporter, CHECKPOINTED)
        if record.attributes["gen_ai.agent.checkpoint.id"]
        in second_run_checkpoints
    ]:
        assert record.trace_id == resumed_run.context.trace_id
        assert record.span_id == resumed_run.context.span_id

    # The two invoke calls are separate traces, and nothing in the telemetry
    # ties them together: LangGraph mints no end-to-end execution id.
    assert paused_run.context.trace_id != resumed_run.context.trace_id


def test_graph_without_checkpointer_emits_no_durability_events(
    start_instrumentation,
    log_exporter,
):
    graph = _build_graph().compile()

    graph.invoke({"value": "new"}, config={"configurable": {"thread_id": "x"}})

    assert _events(log_exporter, CHECKPOINTED) == []
    assert _events(log_exporter, RESUMED) == []
    # The interrupt still happens, so paused is still reported. LangGraph
    # supplies a checkpoint id for the in-memory loop checkpoint even though
    # nothing persisted it.
    assert len(_events(log_exporter, PAUSED)) == 1


def test_plain_graph_run_emits_no_lifecycle_events(
    start_instrumentation,
    log_exporter,
):
    builder = StateGraph(_State)
    builder.add_node("only", lambda _state: {"value": "done"})
    builder.add_edge(START, "only")
    builder.add_edge("only", END)
    graph = builder.compile()

    assert graph.invoke({"value": "new"}) == {"value": "done"}

    for event_name in LIFECYCLE_EVENTS:
        assert _events(log_exporter, event_name) == []


def test_async_interrupt_and_resume_emit_lifecycle_events(
    start_instrumentation,
    log_exporter,
):
    graph = _build_graph().compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "async-lifecycle-test"}}

    async def run() -> str:
        paused_result = await graph.ainvoke({"value": "new"}, config=config)
        await graph.ainvoke(Command(resume="approved"), config=config)
        return paused_result["__interrupt__"][0].id

    real_interrupt_id = asyncio.run(run())

    paused = _events(log_exporter, PAUSED)
    assert len(paused) == 1
    assert paused[0].attributes["gen_ai.agent.pause.id"] == real_interrupt_id

    # aput is wrapped alongside put, so the per-superstep volume matches the
    # synchronous run.
    assert len(_events(log_exporter, CHECKPOINTED)) == 5
    resumed = _events(log_exporter, RESUMED)
    assert len(resumed) == 1
    assert (
        resumed[0].attributes["gen_ai.agent.resumed_from.id"]
        == paused[0].attributes["gen_ai.agent.checkpoint.id"]
    )


def test_uninstrument_restores_the_checkpointer(
    start_instrumentation,
    log_exporter,
):
    checkpointer = InMemorySaver()
    graph = _build_graph().compile(checkpointer=checkpointer)
    graph.invoke({"value": "new"}, config={"configurable": {"thread_id": "u"}})
    assert _events(log_exporter, CHECKPOINTED)

    start_instrumentation.uninstrument()

    assert "put" not in checkpointer.__dict__
    assert "aput" not in checkpointer.__dict__
