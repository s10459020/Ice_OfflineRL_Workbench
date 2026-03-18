from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np

from replay import StateDatasetReader, StateDatasetWriter, StateReplayWrapper
from replay.state_types import AgentState

Policy = Callable[[Any], int]


def collect_state_dataset_with_signatures(
    env: gym.Env,
    output_path: str | Path,
    max_episodes: int = 3,
    max_episode_steps: int | None = None,
    seed: int | None = None,
    policy: Policy | None = None,
    write_interval: int = 0,
    fps: int = 6,
    print_flag: bool = True,
) -> dict[str, Any]:
    if max_episodes <= 0:
        return {"episodes": [], "path": str(Path(output_path))}
    if max_episode_steps is not None and max_episode_steps <= 0:
        return {"episodes": [], "path": str(Path(output_path))}

    writer = StateDatasetWriter(output_path=output_path, write_interval=write_interval)
    env = writer.wrap_env(env)
    if policy is None:
        policy = lambda _obs: env.action_space.sample()

    frame_delay = 1.0 / max(1, int(fps))
    episode_summaries: list[dict[str, Any]] = []

    try:
        for episode in range(1, max_episodes + 1):
            obs, info = env.reset(seed=None if seed is None else seed + episode)
            writer.on_reset(info)

            state_traj = [_state_from_info(info)]
            transition_traj: list[dict[str, Any]] = []
            episode_step = 0

            if print_flag:
                state = state_traj[0]
                print(
                    f"collector_reset episode={episode} state_index=0 "
                    f"pos={state.agent_pos} dir={state.agent_dir} mission='{state.mission}'"
                )
            env.render()
            time.sleep(frame_delay)

            while True:
                episode_step += 1
                action = int(policy(obs))
                next_obs, reward, terminated, truncated, info = env.step(action)
                writer.on_step(action, reward, terminated, truncated, info)
                next_state = _state_from_info(info)

                transition_traj.append(
                    {
                        "o": _digest_obj(obs),
                        "a": int(action),
                        "r": float(reward),
                        "o_next": _digest_obj(next_obs),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                    }
                )
                state_traj.append(next_state)

                if print_flag:
                    print(
                        f"collector_step episode={episode} step={episode_step} "
                        f"a={action} r={float(reward):.3f} "
                        f"terminated={terminated} truncated={truncated} "
                        f"pos={next_state.agent_pos} dir={next_state.agent_dir}"
                    )
                env.render()
                time.sleep(frame_delay)

                done = bool(terminated or truncated)
                forced_cutoff = max_episode_steps is not None and episode_step >= max_episode_steps
                if done:
                    break
                if forced_cutoff:
                    writer.end_episode()
                    break
                obs = next_obs

            transition_signature = _digest_obj(transition_traj)
            state_signature = _digest_obj([_state_payload(s) for s in state_traj])
            episode_summaries.append(
                {
                    "episode_index": episode - 1,
                    "num_transitions": len(transition_traj),
                    "transition_signature": transition_signature,
                    "state_signature": state_signature,
                }
            )
    finally:
        writer.close()

    return {"episodes": episode_summaries, "path": str(Path(output_path))}


def replay_state_dataset_with_signatures(
    env: gym.Env,
    reader: StateDatasetReader,
    episodes: int = 3,
    fps: int = 6,
    reset_pause_sec: float = 1.0,
    print_flag: bool = True,
) -> list[dict[str, Any]]:
    replay_env = StateReplayWrapper(env=env, reader=reader, random_episode=False)
    frame_delay = 1.0 / max(1, int(fps))
    episode_summaries: list[dict[str, Any]] = []

    try:
        for episode_index in range(min(int(episodes), reader.num_episodes)):
            obs, info = replay_env.reset(options={"episode_index": episode_index})
            _ = obs
            state = info["state"]
            states = [state]
            if print_flag:
                print(
                    f"replay_reset episode={episode_index} state_index={info['state_index']} "
                    f"trajectory_length={info['trajectory_length']} "
                    f"pos={state.agent_pos} dir={state.agent_dir}"
                )
            replay_env.render()
            time.sleep(max(0.0, float(reset_pause_sec)))

            terminated = False
            truncated = False
            while not (terminated or truncated):
                action = replay_env.action_space.sample()
                obs, reward, terminated, truncated, info = replay_env.step(action)
                _ = obs
                state = info["state"]
                states.append(state)
                if print_flag:
                    print(
                        f"replay_step episode={episode_index} state_index={info['state_index']} "
                        f"a={int(action)} r={float(reward):.3f} "
                        f"terminated={terminated} truncated={truncated} "
                        f"pos={state.agent_pos} dir={state.agent_dir}"
                    )
                replay_env.render()
                time.sleep(frame_delay)

            state_signature = _digest_obj([_state_payload(s) for s in states])
            episode_summaries.append(
                {
                    "episode_index": episode_index,
                    "num_states": len(states),
                    "state_signature": state_signature,
                }
            )
    finally:
        replay_env.close()

    return episode_summaries


def _digest_obj(value: Any) -> str:
    canonical = json.dumps(_to_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _state_from_info(info: dict[str, Any]) -> AgentState:
    state = info.get("state")
    if not isinstance(state, AgentState):
        raise KeyError("info['state'] missing. Wrap env with StateCaptureWrapper first.")
    return state


def _state_payload(state: AgentState) -> dict[str, Any]:
    grid = np.asarray(state.grid, dtype=np.uint8)
    return {
        "mission": str(state.mission),
        "agent_pos": [int(state.agent_pos[0]), int(state.agent_pos[1])],
        "agent_dir": int(state.agent_dir),
        "carrying": state.carrying,
        "grid_shape": list(grid.shape),
        "grid_digest": hashlib.sha256(grid.tobytes()).hexdigest()[:16],
    }


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value
