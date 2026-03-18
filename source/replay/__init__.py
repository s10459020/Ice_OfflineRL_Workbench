from __future__ import annotations

from .state_types import AgentState

__all__ = [
    "AgentState",
    "StateCaptureWrapper",
    "StateDatasetReader",
    "StateReplayWrapper",
    "StateDatasetWriter",
    "ensure_state_capture",
]


def __getattr__(name: str):
    if name == "StateCaptureWrapper":
        from .state_capture_wrapper import StateCaptureWrapper

        return StateCaptureWrapper
    if name in ("StateDatasetWriter", "ensure_state_capture"):
        from .state_writer import StateDatasetWriter, ensure_state_capture

        return {"StateDatasetWriter": StateDatasetWriter, "ensure_state_capture": ensure_state_capture}[name]
    if name == "StateDatasetReader":
        from .state_reader import StateDatasetReader

        return StateDatasetReader
    if name == "StateReplayWrapper":
        from .state_replay_wrapper import StateReplayWrapper

        return StateReplayWrapper
    raise AttributeError(f"module 'replay' has no attribute '{name}'")
