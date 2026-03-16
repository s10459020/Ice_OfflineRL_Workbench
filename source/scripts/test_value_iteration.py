from typing import Any

import gymnasium as gym
import minigrid  # noqa: F401  # Ensure MiniGrid environments are registered.
import numpy as np
from minigrid.wrappers import FullyObsWrapper

from agent import QTableAgent
from tools import RenderQuiteWrapper, StepPenaltyWrapper
from visualization.minigrid import (
    DistributionWrapper,
    RenderDelayWrapper,
    RenderOverlayWrapper,
    TrailWrapper,
)


def minigrid_q_encoder(obs: Any) -> Any:
    image = np.asarray(obs["image"], dtype=np.uint8)
    direction = int(obs["direction"])
    return direction, image.tobytes()


def _get_observation_wrappers(env: gym.Env) -> list[gym.ObservationWrapper]:
    wrappers: list[gym.Wrapper] = []
    current: gym.Env = env
    while isinstance(current, gym.Wrapper):
        wrappers.append(current)
        current = current.env

    obs_wrappers: list[gym.ObservationWrapper] = []
    for wrapper in reversed(wrappers):
        if isinstance(wrapper, gym.ObservationWrapper):
            obs_wrappers.append(wrapper)
    return obs_wrappers


def _build_wrapped_observation(
    base_env: gym.Env,
    obs_wrappers: list[gym.ObservationWrapper],
    x: int,
    y: int,
    d: int,
) -> Any:
    old_pos = tuple(base_env.agent_pos)
    old_dir = base_env.agent_dir
    try:
        base_env.agent_pos = (x, y)
        base_env.agent_dir = d
        obs = base_env.gen_obs()
        for wrapper in obs_wrappers:
            obs = wrapper.observation(obs)
    finally:
        base_env.agent_pos = old_pos
        base_env.agent_dir = old_dir
    return obs


def run_value_iteration(
    env: gym.Env,
    agent: QTableAgent,
    sweeps: int = 20,
    seed: int | None = 42,
    print_interval: int | None = 1,
    render_flag: bool = True,
) -> tuple[int, int]:
    if sweeps <= 0:
        return 0, 0
    if print_interval is not None and print_interval <= 0:
        raise ValueError("print_interval must be positive when provided.")

    base_env = env.unwrapped
    width = int(base_env.width)
    height = int(base_env.height)
    n_actions = int(agent.n_actions)
    obs_wrappers = _get_observation_wrappers(env)

    steps = 0
    for sweep in range(1, sweeps + 1):
        # One reset per full sweep. The next reset happens only after full traversal.
        env.reset(seed=None if seed is None else seed + sweep)
        if render_flag:
            env.render()

        for x in range(width):
            for y in range(height):
                cell = base_env.grid.get(x, y)
                if cell is not None and not bool(cell.can_overlap()):
                    continue

                for d in range(4):
                    for action in range(n_actions):
                        obs = _build_wrapped_observation(base_env, obs_wrappers, x, y, d)

                        base_env.agent_pos = (x, y)
                        base_env.agent_dir = d
                        if hasattr(base_env, "step_count"):
                            base_env.step_count = 0

                        next_obs, reward, terminated, truncated, _ = env.step(action)
                        if render_flag:
                            env.render()

                        done = bool(terminated or truncated)
                        agent.update(obs, action, float(reward), next_obs, done)
                        steps += 1

        if print_interval is not None and sweep % print_interval == 0:
            print(f"sweep={sweep} steps={steps}")

    return steps, sweeps


agent = QTableAgent(
    n_actions=4,
    alpha=0.1,
    gamma=0.99,
    epsilon=0.0,
    seed=42,
)
agent.set_encoder(minigrid_q_encoder)

env = gym.make("BabyAI-OneRoomS8-v0", render_mode="human")
env = FullyObsWrapper(env)
env = RenderQuiteWrapper(env)
env = StepPenaltyWrapper(env, step_penalty=0.01)
env = RenderOverlayWrapper(env)
env = TrailWrapper(env, clear_on_render=False, max_trails=8)
env = DistributionWrapper(
    env,
    value_fn=lambda obs, action: agent.q(obs, action),
    style="rect12",
)
env = RenderDelayWrapper(env, fps=4, render_on_done=True)

print("start value_iteration | env=BabyAI-OneRoomS8-v0-fullobs | step_penalty=0.01 | trail=on | distribution=rect12 | delay=4fps")
try:
    steps, sweeps = run_value_iteration(
        env=env,
        agent=agent,
        sweeps=20,
        seed=42,
        print_interval=1,
        render_flag=True,
    )
finally:
    env.close()

print(f"train_done | steps={steps} | sweeps={sweeps} | q_states={len(agent.q_table)}")
