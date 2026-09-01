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
* ``StateGraph.compile(checkpointer=...)``. The checkpointer the compiled graph
  retains is wrapped so that every ``put``/``aput`` (LangGraph writes one
  checkpoint per superstep) reports the id it persisted.
"""

from __future__ import annotations

import logging
import threading
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol
from weakref import WeakKeyDictionary

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

_logger = logging.getLogger(__name__)

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

# LangGraph's resume event supplies a checkpoint id and nothing else, so this is
# the only member of ``gen_ai.agent.resumed_from.type`` the library determines.
RESUMED_FROM_TYPE_CHECKPOINT = "checkpoint"

_WRAPPED_METHODS = ("put", "aput")
_MISSING = object()

# Upper bound on the LangGraph namespaces whose last checkpoint id is
# remembered for de-duplication.
_MAX_TRACKED_NAMESPACES = 1024

# Checkpointer instances patched by this instrumentation, mapped to the
# instance attributes they had before patching, so ``uninstrument`` restores
# rather than deletes.
_wrapped_checkpointers: WeakKeyDictionary[Any, dict[str, Any]] = (
    WeakKeyDictionary()
)
_wrapped_lock = threading.Lock()


class _Reporter(Protocol):
    def checkpoint_written(
        self, thread_id: str, checkpoint_id: str
    ) -> None: ...


def _configurable_value(
    config: RunnableConfig | None, key: str, default: str | None = None
) -> str | None:
    """Return one ``configurable`` value from a runnable config."""
    if not config:
        return default
    configurable = config.get("configurable")
    if not configurable:
        return default
    value = configurable.get(key)
    if value is None:
        return default
    return str(value)


def _config_arg(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> RunnableConfig | None:
    """Return the ``config`` argument of a ``put``/``aput`` call."""
    if "config" in kwargs:
        return kwargs["config"]
    return args[0] if args else None


class _CheckpointReporter:
    """Report each persisted checkpoint exactly once.

    A saver may implement ``aput`` by delegating to ``put`` (LangGraph's own
    ``InMemorySaver`` does), and the delegated call may even run on a worker
    thread, so nesting cannot be detected from the call context. Instead the
    last checkpoint id reported for a LangGraph namespace is remembered, and a
    repeat of that id is dropped. Checkpoint ids are unique per write, so a
    repeat only ever means the same write was seen twice.
    """

    def __init__(self, reporter: _Reporter) -> None:
        self._reporter = reporter
        self._last_reported: dict[tuple[str, str], str] = {}
        self._lock = threading.Lock()

    def report(
        self,
        config: RunnableConfig | None,
        returned_config: RunnableConfig | None,
    ) -> None:
        thread_id = _configurable_value(config, "thread_id")
        checkpoint_id = _configurable_value(returned_config, "checkpoint_id")
        if not thread_id or not checkpoint_id:
            return
        namespace = _configurable_value(config, "checkpoint_ns", "") or ""

        key = (thread_id, namespace)
        with self._lock:
            if self._last_reported.get(key) == checkpoint_id:
                return
            self._last_reported[key] = checkpoint_id
            overflow = len(self._last_reported) - _MAX_TRACKED_NAMESPACES
            for stale_key in list(self._last_reported)[:overflow]:
                del self._last_reported[stale_key]

        self._reporter.checkpoint_written(thread_id, checkpoint_id)


def _supports_tracking(checkpointer: Any) -> bool:
    """Return whether the saver can be tracked for later restoration.

    Checked before anything is patched: a saver that cannot be put in the
    registry must not be patched either, or uninstrument could never undo it.
    """
    try:
        hash(checkpointer)
        weakref.ref(checkpointer)
    except TypeError:
        return False
    return True


def _wrap_checkpointer(checkpointer: Any, reporter: _Reporter) -> None:
    """Wrap one checkpointer instance's write methods.

    ``BaseCheckpointSaver.put`` is abstract, so wrapping the base class
    intercepts nothing: every saver overrides it. The instance is patched
    instead, which also keeps the patch scoped to savers an instrumented
    application actually compiled a graph with.
    """
    # Checked before the registry is touched: an unhashable or
    # non-weak-referenceable saver cannot even be looked up there.
    if not _supports_tracking(checkpointer):
        _logger.debug(
            "Checkpointer %r cannot be tracked for uninstrument, "
            "skipping checkpoint events for it",
            type(checkpointer).__name__,
        )
        return

    with _wrapped_lock:
        if checkpointer in _wrapped_checkpointers:
            return

        checkpoint_reporter = _CheckpointReporter(reporter)

        def sync_put(
            wrapped: Callable[..., Any],
            _instance: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> Any:
            result = wrapped(*args, **kwargs)
            checkpoint_reporter.report(_config_arg(args, kwargs), result)
            return result

        async def async_put(
            wrapped: Callable[..., Any],
            _instance: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> Any:
            result = await wrapped(*args, **kwargs)
            checkpoint_reporter.report(_config_arg(args, kwargs), result)
            return result

        # Remember the pre-patch instance attributes so uninstrument restores
        # a saver that legitimately supplied ``put`` or ``aput`` per instance.
        instance_dict: dict[str, Any] = getattr(checkpointer, "__dict__", {})
        previous: dict[str, Any] = {
            name: instance_dict.get(name, _MISSING)
            for name in _WRAPPED_METHODS
        }

        patched: list[str] = []
        for name, wrapper in (("put", sync_put), ("aput", async_put)):
            if getattr(checkpointer, name, None) is None:
                continue
            try:
                wrap_function_wrapper(checkpointer, name, wrapper)
            except (AttributeError, TypeError):  # pragma: no cover
                continue
            patched.append(name)

        if not patched:
            return
        _wrapped_checkpointers[checkpointer] = {
            name: previous[name] for name in patched
        }


def _restore_checkpointer(checkpointer: Any, previous: dict[str, Any]) -> None:
    """Undo one instance patch, restoring any pre-existing attribute."""
    for name, original in previous.items():
        unwrap(checkpointer, name)
        instance_dict: dict[str, Any] | None = getattr(
            checkpointer, "__dict__", None
        )
        if instance_dict is None:
            continue
        if original is _MISSING:
            instance_dict.pop(name, None)
        else:
            instance_dict[name] = original


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
        # Prefer the saver the compiled graph actually retains, so the patch
        # follows what LangGraph will call rather than what was passed in.
        checkpointer = getattr(compiled, "checkpointer", None)
        if checkpointer is None:
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
    except (
        ImportError,
        AttributeError,
    ):  # pragma: no cover - langgraph absent
        pass

    with _wrapped_lock:
        tracked = list(_wrapped_checkpointers.items())
        _wrapped_checkpointers.clear()
    for checkpointer, previous in tracked:
        _restore_checkpointer(checkpointer, previous)
