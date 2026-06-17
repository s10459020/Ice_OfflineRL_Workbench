import gymnasium as gym

from ice_offline.config.paths import data_path
from ice_offline.dataset.base import Dataset
from ice_offline.store.probe.op_collector import ProbeCollectWrapper
from ice_offline.store.probe.op_collector import ProbeEvalFn
from ice_offline.store.probe.op_collector import ProbeInterface
from ice_offline.store.state._lookup import STATE_OPS
from ice_offline.store.state.op_replayer import make_replayer
from ice_offline.tools.printer import print_stage


def replay(
    dataset: Dataset,
    env: gym.Env,
    *,
    episodes: int,
    seed: int | None = None,
    print_interval: int = 1,
) -> None:
    for i in range(episodes):
        _, info = env.reset(seed=None if seed is None else seed + i)
        episode_index = info["episode_index"]
        trajectory = dataset.episodes[episode_index]
        for action in trajectory.actions:
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        if (i + 1) % print_interval == 0:
            print(f"replay episode={i + 1}/{episodes}")


def probe(
    task_id: str,
    dataset: Dataset,
    probe: ProbeInterface,
    eval_fn: ProbeEvalFn,
    *,
    episodes: int = 1,
    seed: int | None = None,
    env_kwargs: dict | None = None,
) -> object:
    state_cls, state_io_cls, _ = STATE_OPS[dataset.env_id]
    env = make_replayer(
        dataset=dataset,
        state_cls=state_cls,
        state_io_cls=state_io_cls,
        eval_env=dataset.make_env(**(env_kwargs or {})),
        render_mode=None,
    )
    probe_col = ProbeCollectWrapper(env, probe, eval_fn)

    print_stage(f"Probe Replay {task_id}")
    replay(dataset, probe_col, episodes=episodes, seed=seed, print_interval=1)
    path = data_path("probe", task_id)
    probe_data = probe_col.save(path)
    return probe_data


if __name__ == "__main__":
    import numpy as np

    from ice_offline.config.paths import _task_id
    from ice_offline.dataset._lookup import make_dataset
    from ice_offline.store.probe.action_axis_probe import ActionAxisProbe

    device = "cuda:0"
    dataset = make_dataset("hopper_simple", device=device)

    def eval_fn(observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return np.zeros(actions.shape[0], dtype=np.float32)

    probe_data = probe(
        _task_id(dataset.id, "probe"),
        dataset,
        ActionAxisProbe(100),
        eval_fn,
    )
    print(probe_data.path)
