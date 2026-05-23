from typing import Any, Protocol


class State:
    def serialize(self) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_serialized(cls, payload: dict[str, Any]):
        raise NotImplementedError


class StateIO(Protocol):
    def get_state(self) -> State:
        ...

    def set_state(self, state: State) -> None:
        ...
