from __future__ import annotations

from abc import ABC
from collections.abc import Iterable
from typing import Any, Generic, Protocol, TypeVar

from ..messages import Message, RoutedMessage

InputT = TypeVar("InputT")

ModuleOutput = None | RoutedMessage[Any] | Iterable[RoutedMessage[Any]]


class ModuleContext(Protocol):
    async def put(self, queue_name: str, message: Message[Any]) -> None: ...

    async def submit(
        self,
        queue_name: str,
        payload: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...


class BaseModule(ABC, Generic[InputT]):
    run_in_thread = False

    def __init__(self, name: str, input_queue: str) -> None:
        if not name:
            raise ValueError("Module name cannot be empty.")
        if not input_queue:
            raise ValueError("Module input_queue cannot be empty.")

        self.name = name
        self.input_queue = input_queue

    async def process(
        self,
        message: Message[InputT],
        context: ModuleContext,
    ) -> ModuleOutput:
        raise NotImplementedError(
            f"{type(self).__name__} must implement process() or enable run_in_thread "
            "and implement process_blocking()."
        )

    def process_blocking(
        self,
        message: Message[InputT],
        context: ModuleContext,
    ) -> ModuleOutput:
        raise NotImplementedError(
            f"{type(self).__name__} must implement process_blocking() when "
            "run_in_thread is enabled."
        )
