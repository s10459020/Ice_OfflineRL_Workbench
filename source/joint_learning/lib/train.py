from pathlib import Path

import gymnasium as gym

from joint_learning.agents.dynamics import Dynamic
from joint_learning.lib.eval import evaluate_mean
from joint_learning.lib.metrics import metrics_path
from joint_learning.lib.metrics import save_metric
from joint_learning.lib.plot import plot_path
from joint_learning.lib.plot import save_training_plot


def agent_path(agent, dataset, train_index: int) -> Path:
    return Path(__file__).resolve().parent.parent / "model" / f"{agent.id}-{dataset.id}-{train_index}.pt"


def dynamic_path(dataset) -> Path:
    return Path(__file__).resolve().parent.parent / "model" / f"dynamics-{dataset.id}.pt"


def train_dynamic(dataset, model_steps: int, model_batch_size: int, model_print_interval: int) -> Dynamic:
    dynamic = Dynamic(dataset.obs_size, dataset.act_size, device=dataset.device)
    path = dynamic_path(dataset)
    if path.exists():
        dynamic.load(path)
        return dynamic.eval().freeze()

    print(f"train dynamic: {dataset.id}")
    for step in range(1, model_steps + 1):
        dynamic.update(dataset.sample_batch(model_batch_size))
        if step % model_print_interval == 0:
            print(f"model step {step}/{model_steps}")

    dynamic.save(path)
    return dynamic.eval().freeze()


def train(
    agent,
    dataset,
    experiment_id: str,
    train_index: int,
    train_count: int,
    eval_count: int,
    batch_size: int,
    print_interval: int,
    return_avg_window: int,
) -> float:
    env = gym.make(dataset.env_id)
    history: list[tuple[int, float]] = []
    moving_avg_return = 0.0

    print(
        f"train agent={agent.id} dataset={dataset.id} "
        f"steps={train_count} device={dataset.device}"
    )
    try:
        for step in range(1, train_count + 1):
            batch = dataset.sample_batch(batch_size)
            agent.update(batch)

            if step % print_interval == 0 or step == train_count:
                episode_return = evaluate_mean(agent, env, dataset, eval_count)
                history.append((step, episode_return))
                recent_returns = [value for _, value in history[-return_avg_window:]]
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
