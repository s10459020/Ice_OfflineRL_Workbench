from pathlib import Path

import gymnasium as gym

from joint_learning.lib.agent import model_path
from joint_learning.lib.eval import evaluate
from joint_learning.lib.plot import plot_path
from joint_learning.lib.plot import save_training_plot


STEPS = 500_000
BATCH_SIZE = 256
PRINT_INTERVAL = 2_000
RETURN_AVG_WINDOW = 10


def train(agent, dataset, experiment_id: str) -> Path:
    env = gym.make(dataset.env_id)
    history: list[tuple[int, float]] = []

    print(f"train agent={agent.id} dataset={dataset.id} steps={STEPS} device={dataset.device}")
    try:
        for step in range(1, STEPS + 1):
            batch = dataset.sample_batch(BATCH_SIZE)
            agent.update(batch)

            if step % PRINT_INTERVAL == 0 or step == STEPS:
                episode_return = evaluate(agent, env, dataset)
                history.append((step, episode_return))
                recent_returns = [value for _, value in history[-RETURN_AVG_WINDOW:]]
                moving_avg_return = sum(recent_returns) / len(recent_returns)
                print(
                    f"step={step} "
                    f"return={episode_return:.6g} "
                    f"moving_avg_return={moving_avg_return:.6g}"
                )
    finally:
        env.close()

    path = model_path(agent, dataset)
    agent.save(path)
    save_training_plot(
        history,
        plot_path(experiment_id, agent, dataset),
        f"{experiment_id} {agent.id} {dataset.id}",
    )
    print(f"saved model: {path}")
    return path
