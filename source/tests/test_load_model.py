import random
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import minigrid  # noqa: F401  # Ensure MiniGrid environments are registered.

from agent import QTableAgent
from tests.test_Q_table import q_table_state_from_minigrid_observation


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_model(model_name: str = "q_table_agent.pkl") -> tuple[QTableAgent, Path]:
    model_path = REPO_ROOT / "model" / model_name
    agent = QTableAgent.load(model_path)
    return agent, model_path


def run_loaded_model_human(
    model_name: str = "q_table_agent.pkl",
    env_id: str = "MiniGrid-Empty-Random-5x5-v0",
    episodes: int = 50,
    max_steps_per_episode: int = 200,
    seed: int = 123,
    state_encoder: Callable[[Any], Any] = q_table_state_from_minigrid_observation,
    allowed_actions: tuple[int, ...] = (0, 1, 2),
    random_reset: bool = True,
) -> None:
    agent, model_path = load_model(model_name=model_name)
    env = gym.make(env_id, render_mode="human")

    if any((a < 0 or a >= int(env.action_space.n)) for a in allowed_actions):
        raise ValueError("allowed_actions contains invalid action ids.")
    if len(allowed_actions) != agent.n_actions:
        raise ValueError(
            f"allowed_actions length ({len(allowed_actions)}) must match "
            f"loaded agent.n_actions ({agent.n_actions})."
        )

    print(f"loaded model: {model_path}")
    print(f"q_states={len(agent.q_table)} | env={env_id}")

    rng = random.Random(seed)
    success = 0
    for episode in range(1, episodes + 1):
        reset_seed = rng.randrange(0, 2**31 - 1) if random_reset else (seed + episode)
        obs, _ = env.reset(seed=reset_seed)
        state: Any = state_encoder(obs)
        ep_reward = 0.0

        for step in range(1, max_steps_per_episode + 1):
            action_idx = agent.act(state, greedy=True)
            env_action = int(allowed_actions[action_idx])
            next_obs, reward, terminated, truncated, _ = env.step(env_action)
            state = state_encoder(next_obs)
            ep_reward += float(reward)

            if terminated or truncated:
                if terminated:
                    success += 1
                print(
                    f"episode={episode} step={step} reward={ep_reward:.3f} "
                    f"terminated={terminated} truncated={truncated}"
                )
                break
        else:
            print(
                f"episode={episode} step={max_steps_per_episode} reward={ep_reward:.3f} "
                f"terminated=False truncated=False"
            )

    env.close()
    print(f"success_rate={success / max(1, episodes):.3f}")


if __name__ == "__main__":
    run_loaded_model_human()
