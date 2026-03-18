from __future__ import annotations

from typing import Any

import gymnasium as gym

from ice_offline.replay import StateDatasetReader, StateReplayWrapper
from ice_offline.tools import ensure_render_quiet


def replay(
    env: gym.Env,
    reader: StateDatasetReader,
    episodes: int = 3,
    render_flag: bool = False,
    print_flag: bool = True,
) -> list[dict[str, Any]]:
    env = ensure_render_quiet(env)
    replay_env = StateReplayWrapper(env=env, reader=reader, random_episode=False)
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
            if render_flag:
                replay_env.render()

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
                if render_flag:
                    replay_env.render()

            episode_summaries.append(
                {
                    "episode_index": episode_index,
                    "num_states": len(states),
                }
            )
    finally:
        replay_env.close()

    return episode_summaries
