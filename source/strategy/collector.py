from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol

import gymnasium as gym

from replay import ObservationCollector, StateCollector
from tools import ensure_render_quite

Policy = Callable[[Any], int]


class CollectorHook(Protocol):
    def prepare_env(self, env: gym.Env) -> gym.Env: ...

    def on_reset(self, info: dict[str, Any]) -> None: ...

    def on_step(
        self,
        action: int,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None: ...

    def on_episode_end(self, *, forced_cutoff: bool) -> None: ...

    def close(self) -> None: ...

    def result(self) -> dict[str, Any]: ...
def collect_dataset(
    env: gym.Env,
    *,
    collect_state: bool = False,
    collect_observation: bool = False,
    state_output_path: str | Path = "tmps/one_room_s8_info.hdf5",
    observation_output_path: str | Path = "tmps/one_room_s8_data.hdf5",
    dataset_id: str = "one-room-s8-data-v0",
    max_episodes: int = 3,
    max_episode_steps: int | None = 20,
    seed: int | None = 42,
    policy: Policy | None = None,
    flush_interval: int = 0,
    write_interval: int | None = None,
    record_infos: bool = True,
    overwrite_local_dataset: bool = True,
    normalize_mission_observation: bool = True,
    render_flag: bool = False,
    print_flag: bool = False,
) -> dict[str, Any]:
    empty_state = {"path": str(Path(state_output_path))} if collect_state else None
    empty_obs = (
        {
            "dataset_id": str(dataset_id),
            "minari_path": "",
            "path": str(Path(observation_output_path)),
        }
        if collect_observation
        else None
    )
    if max_episodes <= 0:
        return {"episodes": 0, "steps": 0, "collect_state": empty_state, "collect_observation": empty_obs}
    if max_episode_steps is not None and max_episode_steps <= 0:
        return {"episodes": 0, "steps": 0, "collect_state": empty_state, "collect_observation": empty_obs}
    if not collect_state and not collect_observation:
        raise ValueError("At least one of collect_state / collect_observation must be True.")

    hooks: list[CollectorHook] = []
    state_hook: StateCollector | None = None
    observation_hook: ObservationCollector | None = None
    if collect_observation:
        observation_hook = ObservationCollector(
            output_path=observation_output_path,
            dataset_id=dataset_id,
            record_infos=record_infos,
            overwrite_local_dataset=overwrite_local_dataset,
            normalize_mission_observation=normalize_mission_observation,
        )
        hooks.append(observation_hook)
    if collect_state:
        if write_interval is not None:
            flush_interval = int(write_interval)
        state_hook = StateCollector(output_path=state_output_path, flush_interval=flush_interval)
        hooks.append(state_hook)

    wrapped_env = env
    for hook in hooks:
        wrapped_env = hook.prepare_env(wrapped_env)

    wrapped_env = ensure_render_quite(wrapped_env)

    if policy is None:
        policy = lambda obs: wrapped_env.action_space.sample()

    total_steps = 0
    episodes = 0
    try:
        for episode in range(1, max_episodes + 1):
            obs, info = wrapped_env.reset(seed=None if seed is None else seed + episode)
            for hook in hooks:
                hook.on_reset(info)
            if render_flag:
                wrapped_env.render()

            episode_step = 0
            while True:
                episode_step += 1
                action = int(policy(obs))
                next_obs, reward, terminated, truncated, info = wrapped_env.step(action)
                for hook in hooks:
                    hook.on_step(action, reward, terminated, truncated, info)
                total_steps += 1
                if render_flag:
                    wrapped_env.render()

                done = bool(terminated or truncated)
                forced_cutoff = max_episode_steps is not None and episode_step >= max_episode_steps
                if print_flag:
                    print(
                        f"step={total_steps} episode={episode} episode_step={episode_step} "
                        f"action={action} reward={float(reward):.3f} "
                        f"terminated={terminated} truncated={truncated}"
                    )

                if done:
                    if print_flag:
                        print(
                            f"episode_end episode={episode} episode_steps={episode_step} "
                            "reason=terminated_or_truncated"
                        )
                    break
                if forced_cutoff:
                    for hook in hooks:
                        hook.on_episode_end(forced_cutoff=True)
                    if print_flag:
                        print(
                            f"episode_end episode={episode} episode_steps={episode_step} "
                            "reason=max_episode_steps"
                        )
                    break
                obs = next_obs

            episodes += 1
    finally:
        for hook in reversed(hooks):
            hook.close()

    return {
        "episodes": episodes,
        "steps": total_steps,
        "collect_state": state_hook.result() if state_hook is not None else None,
        "collect_observation": observation_hook.result() if observation_hook is not None else None,
    }
