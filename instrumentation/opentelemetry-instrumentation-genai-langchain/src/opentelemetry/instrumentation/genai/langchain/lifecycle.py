# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Agent lifecycle telemetry for LangGraph durable executions.

Every name in this module is a candidate semantic convention proposed in
open-telemetry/semantic-conventions-genai#445 (agent lifecycle events for async
and long running executions). None of them is stable, so they are kept here as
plain string constants instead of being imported from the semconv package.

Two interception points feed these events, both of them generic over any
LangGraph application:

* ``langgraph.callbacks`` lifecycle dispatch. LangGraph calls ``on_interrupt``
  and ``on_resume`` on the handlers registered in the run's callback manager,
  passing the real ``Interrupt`` objects and the checkpoint id the graph
  paused at or resumed from.
* ``StateGraph.compile(checkpointer=...)``. The checkpointer instance is
  wrapped so that every ``put``/``aput`` (LangGraph writes one checkpoint per
  superstep) reports the id it persisted.
"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Protocol
from weakref import WeakSet

from langchain_core.runnables import RunnableConfig
from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.utils import unwrap

if TYPE_CHECKING:  # pragma: no cover - typing only
    from opentelemetry.instrumentation.genai.langchain.callback_handler import (
        OpenTelemetryLangChainCallbackHandler,
    )

__all__ = [
    "ATTR_CHECKPOINT_ID",
    "ATTR_PAUSE_ID",
    "ATTR_PAUSE_REASON",
    "ATTR_RESUMED_FROM_ID",
    "ATTR_RESUMED_FROM_TYPE",
    "EVENT_AGENT_CHECKPOINTED",
    "EVENT_AGENT_PAUSED",
    "EVENT_AGENT_RESUMED",
    "RESUMED_FROM_TYPE_CHECKPOINT",
    "instrument_checkpointers",
    "uninstrument_checkpointers",
]

# Candidate event names, pending open-telemetry/semantic-conventions-genai#445.
EVENT_AGENT_PAUSED = "gen_ai.agent.paused"
EVENT_AGENT_CHECKPOINTED = "gen_ai.agent.checkpointed"
EVENT_AGENT_RESUMED = "gen_ai.agent.resumed"

# Candidate attribute names, pending the same proposal.
ATTR_PAUSE_ID = "gen_ai.agent.pause.id"
ATTR_PAUSE_REASON = "gen_ai.agent.pause.reason"
ATTR_CHECKPOINT_ID = "gen_ai.agent.checkpoint.id"
ATTR_RESUMED_FROM_TYPE = "gen_ai.agent.resumed_from.type"
ATTR_RESUMED_FROM_ID = "gen_ai.agent.resumed_from.id"

# Only the ``checkpoint`` member of ``gen_ai.agent.resumed_from.type`` is
# observable in LangGraph: the resume payload LangGraph reports is always a
# checkpoint id, never a pause id.
RESUMED_FROM_TYPE_CHECKPOINT = "checkpoint"

_WRAPPED_MARKER = "_otel_genai_lifecycle_wrapped"

# Savers are free to implement ``aput`` by delegating to ``put`` (LangGraph's
# own ``InMemorySaver`` does), which would report the same checkpoint twice.
# Only the outermost wrapped call reports.
_in_checkpoint_write: ContextVar[bool] = ContextVar(
    "otel_genai_in_checkpoint_write", default=False
)

# Checkpointer instances patched by this instrumentation, so that
# ``uninstrument`` can restore them.
_wrapped_checkpointers: WeakSet[Any] = WeakSet()


class _Reporter(Protocol):
    def checkpoint_written(
        self, thread_id: str, checkpoint_id: str
    ) -> None: ...


def _configurable_value(config: RunnableConfig | None, key: str) -> str | None:
    """Return one ``configurable`` value from a runnable config."""
    if not config:
        return None
    configurable = config.get("configurable")
    if not configurable:
        return None
    value = configurable.get(key)
    return str(value) if value else None


def _config_arg(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> RunnableConfig | None:
    """Return the ``config`` argument of a ``put``/``aput`` call."""
    if "config" in kwargs:
        return kwargs["config"]
    return args[0] if args else None


def _report(
    reporter: _Reporter,
    config: RunnableConfig | None,
    returned_config: RunnableConfig | None,
) -> None:
    """Report the checkpoint a ``put``/``aput`` call persisted."""
    thread_id = _configurable_value(config, "thread_id")
    checkpoint_id = _configurable_value(returned_config, "checkpoint_id")
    if thread_id and checkpoint_id:
        reporter.checkpoint_written(thread_id, checkpoint_id)


def _wrap_checkpointer(checkpointer: Any, reporter: _Reporter) -> None:
    """Wrap one checkpointer instance's write methods.

    ``BaseCheckpointSaver.put`` is abstract, so wrapping the base class
    intercepts nothing: every saver overrides it. The instance is patched
    instead, which also keeps the patch scoped to savers an instrumented
    application actually compiled a graph with.
    """
    if getattr(checkpointer, _WRAPPED_MARKER, False):
        return

    def sync_put(
        wrapped: Callable[..., Any],
        _instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if _in_checkpoint_write.get():
            return wrapped(*args, **kwargs)
        token = _in_checkpoint_write.set(True)
        try:
            result = wrapped(*args, **kwargs)
        finally:
            _in_checkpoint_write.reset(token)
        _report(reporter, _config_arg(args, kwargs), result)
        return result

    async def async_put(
        wrapped: Callable[..., Any],
        _instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if _in_checkpoint_write.get():
            return await wrapped(*args, **kwargs)
        token = _in_checkpoint_write.set(True)
        try:
            result = await wrapped(*args, **kwargs)
        finally:
            _in_checkpoint_write.reset(token)
        _report(reporter, _config_arg(args, kwargs), result)
        return result

    patched = False
    for name, wrapper in (("put", sync_put), ("aput", async_put)):
        if getattr(checkpointer, name, None) is None:
            continue
        try:
            wrap_function_wrapper(checkpointer, name, wrapper)
        except (AttributeError, TypeError):  # pragma: no cover - exotic saver
            continue
        patched = True

    if patched:
        try:
            setattr(checkpointer, _WRAPPED_MARKER, True)
        except (AttributeError, TypeError):  # pragma: no cover - exotic saver
            pass
        _wrapped_checkpointers.add(checkpointer)


class _CompileWrapper:
    """Wrap ``StateGraph.compile`` to reach the checkpointer it was given."""

    def __init__(self, reporter: _Reporter) -> None:
        self._reporter = reporter

    def __call__(
        self,
        wrapped: Callable[..., Any],
        _instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        compiled = wrapped(*args, **kwargs)
        checkpointer = kwargs.get("checkpointer")
        if checkpointer is None and args:
            checkpointer = args[0]
        # ``checkpointer`` is also allowed to be ``None`` or a bool (a subgraph
        # inheriting the parent's saver); only real savers can be wrapped.
        if checkpointer is not None and not isinstance(checkpointer, bool):
            _wrap_checkpointer(checkpointer, self._reporter)
        return compiled


def instrument_checkpointers(
    callback_handler: OpenTelemetryLangChainCallbackHandler,
) -> bool:
    """Wrap ``StateGraph.compile``. Returns False when LangGraph is absent."""
    try:
        wrap_function_wrapper(
            "langgraph.graph.state",
            "StateGraph.compile",
            _CompileWrapper(callback_handler),
        )
    except (ImportError, AttributeError):
        return False
    return True


def uninstrument_checkpointers() -> None:
    """Undo ``instrument_checkpointers`` and restore patched savers."""
    try:
        unwrap("langgraph.graph.state.StateGraph", "compile")
    except (ImportError, AttributeError):  # pragma: no cover - langgraph absent
        pass

    for checkpointer in list(_wrapped_checkpointers):
        # The instance patch shadows the class method with an instance
        # attribute, so dropping the attribute restores the original.
        instance_dict: dict[str, Any] | None = getattr(
            checkpointer, "__dict__", None
        )
        if instance_dict is not None:
            for name in ("put", "aput", _WRAPPED_MARKER):
                instance_dict.pop(name, None)
    _wrapped_checkpointers.clear()
