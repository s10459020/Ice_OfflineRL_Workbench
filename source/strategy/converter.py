from __future__ import annotations

from typing import Any

from replay.convert_fullobs import convert_fullobs
from replay.state_types import AgentState


def convert_observation_tranjectory_to_state_tranjectory(observations: Any) -> list[AgentState]:
    """
    Convert one observation trajectory (fullobs) into one state trajectory.
    """
    return convert_fullobs(observations)
