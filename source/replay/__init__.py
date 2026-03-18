from __future__ import annotations

from .state_types import AgentState

__all__ = [
    "AgentState",
    "convert_observation",
    "ObservationCollector",
    "StateCollector",
    "serialize_state_tranjectory",
    "serialize_observation_tranjectory",
    "StateCaptureWrapper",
    "StateDatasetReader",
    "StateReplayWrapper",
    "StateDatasetWriter",
    "ensure_state_capture",
    "read_metadata",
    "resolve_env_id",
]


def __getattr__(name: str):
    if name == "convert_observation":
        from .convert_observation import convert_observation

        return convert_observation
    if name == "ObservationCollector":
        from .collect_observation_dataset import ObservationCollector

        return ObservationCollector
    if name == "StateCollector":
        from .collect_state_dataset import StateCollector

        return StateCollector
    if name == "serialize_state_tranjectory":
        from .serialize_state_tranjectory import serialize_state_tranjectory

        return serialize_state_tranjectory
    if name == "serialize_observation_tranjectory":
        from .serialize_observation_tranjectory import serialize_observation_tranjectory

        return serialize_observation_tranjectory
    if name == "StateCaptureWrapper":
        from .state_capture_wrapper import StateCaptureWrapper

        return StateCaptureWrapper
    if name in ("StateDatasetWriter", "ensure_state_capture"):
        from .write_state_dataset import StateDatasetWriter
        from .collect_state_dataset import ensure_state_capture

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
