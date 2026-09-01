# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import threading
from dataclasses import dataclass, field
from uuid import UUID

from opentelemetry.util.genai.types import GenAIInvocation

__all__ = ["_InvocationManager"]


@dataclass
class _InvocationState:
    invocation: GenAIInvocation | None
    children: list[UUID] = field(default_factory=lambda: list())
    parent_run_id: UUID | None = None
    ended: bool = False


class _InvocationManager:
    def __init__(
        self,
    ) -> None:
        # Map from run_id -> _InvocationState, to keep track of invocations and parent/child relationships
        # TODO: TTL cache to avoid memory leaks in long-running processes.
        self._invocations: dict[UUID, _InvocationState] = {}
        # Map from LangGraph thread id -> run_id of the live graph run on that
        # thread. A checkpointer call carries no callback run id, so the thread
        # id in its config is the only handle back to the running invocation.
        self._threads: dict[str, UUID] = {}
        self._threads_lock = threading.Lock()

    def bind_thread(self, thread_id: str, run_id: UUID) -> None:
        """Record which run currently owns a LangGraph thread id.

        LangGraph serializes runs on a thread id, so at most one run should own
        one at a time. If an application still starts two, the later run wins
        and the earlier one stops receiving checkpoint events.
        """
        with self._threads_lock:
            self._threads[thread_id] = run_id

    def unbind_thread(self, run_id: UUID) -> None:
        with self._threads_lock:
            for thread_id, owner in list(self._threads.items()):
                if owner == run_id:
                    del self._threads[thread_id]

    def get_thread_invocation(self, thread_id: str) -> GenAIInvocation | None:
        with self._threads_lock:
            run_id = self._threads.get(thread_id)
        return self.get_invocation(run_id) if run_id is not None else None

    def add_invocation_state(
        self,
        run_id: UUID,
        parent_run_id: UUID | None,
        invocation: GenAIInvocation | None,
    ) -> None:
        invocation_state = _InvocationState(invocation=invocation)

        if parent_run_id is not None and parent_run_id in self._invocations:
            invocation_state.parent_run_id = parent_run_id

            parent_invocation_state = self._invocations[parent_run_id]
            parent_invocation_state.children.append(run_id)

        self._invocations[run_id] = invocation_state

    def get_invocation(self, run_id: UUID) -> GenAIInvocation | None:
        invocation_state = self._invocations.get(run_id)
        return invocation_state.invocation if invocation_state else None

    def get_parent_run_id(self, run_id: UUID) -> UUID | None:
        invocation_state = self._invocations.get(run_id)
        return invocation_state.parent_run_id if invocation_state else None

    def delete_invocation_state(self, run_id: UUID) -> None:
        invocation_state = self._invocations.get(run_id)
        if not invocation_state:
            return

        invocation_state.ended = True

        # Defer removal if any children are still live, so upward traversal
        # (e.g. _find_nearest_agent) can still walk through this node.
        if any(c in self._invocations for c in invocation_state.children):
            return

        self._invocations.pop(run_id, None)

        # Propagate cleanup upward: if the parent has already ended and has no
        # more live children, it can now be removed too.
        if invocation_state.parent_run_id:
            parent_state = self._invocations.get(
                invocation_state.parent_run_id
            )
            if parent_state is not None and parent_state.ended:
                self.delete_invocation_state(invocation_state.parent_run_id)
