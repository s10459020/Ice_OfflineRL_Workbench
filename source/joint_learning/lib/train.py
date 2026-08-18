from pathlib import Path

import gymnasium as gym

from joint_learning.agents.dynamics import SCASDynamics
from joint_learning.lib.eval import evaluate
from joint_learning.lib.metrics import metrics_path
from joint_learning.lib.metrics import save_metric
from joint_learning.lib.plot import plot_path
from joint_learning.lib.plot import save_training_plot


STEPS = 200_000
BATCH_SIZE = 256
PRINT_INTERVAL = 2_000
RETURN_AVG_WINDOW = 10

MODEL_STEPS = 500_000
MODEL_BATCH_SIZE = 256
MODEL_PRINT_INTERVAL = 10_000


def agent_path(agent, dataset, train_index: int) -> Path:
    return Path(__file__).resolve().parent.parent / "model" / f"{agent.id}-{dataset.id}-{train_index}.pt"


def model_path(dataset) -> Path:
    return Path(__file__).resolve().parent.parent / "model" / f"dynamics-{dataset.id}.pt"


def train_model(dataset, device: str) -> SCASDynamics:
    model = SCASDynamics(dataset.obs_size, dataset.act_size, device=device)
    path = model_path(dataset)
    if path.exists():
        model.load(path)
        return model.eval()

    print(f"train model: {dataset.id}")
    for step in range(1, MODEL_STEPS + 1):
        model.update(dataset.sample_batch(MODEL_BATCH_SIZE))
        if step % MODEL_PRINT_INTERVAL == 0:
            print(f"model step {step}/{MODEL_STEPS}")

    model.save(path)
    return model.eval()


def train(agent, dataset, experiment_id: str, train_index: int) -> float:
    env = gym.make(dataset.env_id)
    history: list[tuple[int, float]] = []
    moving_avg_return = 0.0

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

    path = agent_path(agent, dataset, train_index)
    agent.save(path)
    save_metric(metrics_path(agent.id, dataset.id), moving_avg_return)
    save_training_plot(
        history,
        plot_path(experiment_id, agent, dataset),
        f"{experiment_id} {agent.id} {dataset.id}",
    )
    print(f"saved model: {path}")
    return moving_avg_return
