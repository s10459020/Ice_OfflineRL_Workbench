from __future__ import annotations

from .state_types import AgentState

__all__ = [
    "AgentState",
    "convert_fullobs",
    "convert_fullobs_episode",
    "ObservationCollector",
    "StateCollector",
    "StateCaptureWrapper",
    "StateDatasetReader",
    "StateReplayWrapper",
    "StateDatasetWriter",
    "ensure_state_capture",
    "read_metadata",
    "resolve_env_id",
]


def __getattr__(name: str):
    if name in ("convert_fullobs", "convert_fullobs_episode"):
        from .convert_fullobs import convert_fullobs, convert_fullobs_episode

        return {"convert_fullobs": convert_fullobs, "convert_fullobs_episode": convert_fullobs_episode}[name]
    if name == "ObservationCollector":
        from .collect_observation import ObservationCollector

        return ObservationCollector
    if name == "StateCollector":
        from .collect_state import StateCollector

        return StateCollector
    if name == "StateCaptureWrapper":
        from .state_capture_wrapper import StateCaptureWrapper

        return StateCaptureWrapper
    if name in ("StateDatasetWriter", "ensure_state_capture"):
        from .write_state_dataset import StateDatasetWriter
        from .collect_state import ensure_state_capture

        return {"StateDatasetWriter": StateDatasetWriter, "ensure_state_capture": ensure_state_capture}[name]
    if name == "StateDatasetReader":
        from .read_state_dataset import StateDatasetReader

        return StateDatasetReader
    if name == "StateReplayWrapper":
        from .state_replay_wrapper import StateReplayWrapper

        return StateReplayWrapper
    if name in ("read_metadata", "resolve_env_id"):
        from .read_metadata import read_metadata, resolve_env_id

        return {"read_metadata": read_metadata, "resolve_env_id": resolve_env_id}[name]
    raise AttributeError(f"module 'replay' has no attribute '{name}'")
