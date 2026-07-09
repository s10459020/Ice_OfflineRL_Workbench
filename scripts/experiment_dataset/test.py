from pathlib import Path

import gymnasium as gym

from ice_offline.agent._lookup import make_agent
from ice_offline.config.paths import _task_id
from ice_offline.config.paths import eval_data_path
from ice_offline.dataset._lookup import make_dataset
from ice_offline.store.eval.collector import EvalCollector
from ice_offline.store.state._lookup import STATE_OPS
from ice_offline.store.state.op_converter import StateConverter
from ice_offline.store.state.op_replayer import make_replayer
from plot import eval
from plot import plot
from view import save_boxplots
from view import save_table_boxplot
from view import save_tables
from view import TABLES


DATASETS = [
    "walker2d_d4rl_medium",
    "walker2d_d4rl_hybrid",
    "walker2d_d4rl_expert",
    "walker2d_replay_medium",
    "walker2d_replay_expert",
]

AGENTS = [
    ("bc", None, 50_000),
    ("td3bc_n", None, 100_000),
    ("iql", None, 200_000),
    ("cql", None, 500_000),
    ("scas_gp", 100_000, 500_000),
    ("scaspl_gp", 100_000, 500_000),
    ("aspl_gp_punish_050", None, 500_000),
]

COUNT = 10
EVALS = 100
INTERVAL = 1_000
EXPERIMENT = "in_dataset"


def _steps(start_step: int) -> list[int]:
    return [start_step + INTERVAL * index for index in range(COUNT + 1)]


class DatasetReplayResetCallback:
    def __init__(self, dataset) -> None:
        self._dataset = dataset
        self._wrappers: dict[int, object] = {}

    def __call__(self, env, seed: int) -> object:
        wrapper = self._wrapper(env)
        wrapper.reset(seed=seed)
        return wrapper.env.unwrapped._get_obs()

    def close(self) -> None:
        for wrapper in self._wrappers.values():
            wrapper._state_dataset.close()
        self._wrappers.clear()

    def _wrapper(self, env):
        key = id(env)
        if key not in self._wrappers:
            state_cls, state_io_cls, _ = STATE_OPS[self._dataset.env_id]
            self._wrappers[key] = make_replayer(
                dataset=self._dataset,
                state_cls=state_cls,
                state_io_cls=state_io_cls,
                eval_env=env,
                render_mode=None,
            )
        return self._wrappers[key]


def _make_reset_callback(dataset) -> DatasetReplayResetCallback:
    _, _, converter_cls = STATE_OPS[dataset.env_id]
    state_path = dataset.path.with_name("state_data.hdf5")
    _ensure_state_dataset(state_path, dataset, converter_cls)
    return DatasetReplayResetCallback(dataset)


def _ensure_state_dataset(
    state_path: Path,
    dataset,
    converter_cls,
) -> None:
    if state_path.exists():
        return
    state_dataset = StateConverter(dataset=dataset, converter_cls=converter_cls).convert()
    state_dataset.close()


def run_in_dataset(
    agent,
    env: gym.Env,
    reset_callback,
    seed: int = 42,
) -> float:
    agent.set_seed(seed)
    o = reset_callback(env, seed)
    result = 0.0
    trun = term = False
    while not (trun or term):
        a = agent.act(o)
        o, r, trun, term, _ = env.step(a)
        result += float(r)
    return result


def test(
    task_id: str,
    dataset_id: str,
    agent_id: str,
    model_step: int | None,
    agent_steps: list[int],
) -> Path:
    dataset = make_dataset(dataset_id, device="cuda")
    eval_path = eval_data_path(EXPERIMENT, task_id)
    eval_col = EvalCollector(dataset.make_env())
    reset_callback = _make_reset_callback(dataset)
    try:
        for agent_step in agent_steps:
            agent = make_agent(agent_id, dataset, device="cuda", model_step=model_step)
            agent.load(task_id, agent_step)
            print(f"experiment={EXPERIMENT}, task={task_id}, agent_step={agent_step}")
            for index in range(EVALS):
                result = run_in_dataset(agent, eval_col, reset_callback, 42 + index)
                print(f"test step={agent_step} episode={index + 1}/{EVALS} return={result:.6g}")
            eval_col.flush(agent_step)
        eval_col.save(eval_path)
    finally:
        reset_callback.close()
        eval_col.close()

    print(f"saved: {eval_path}")
    return eval_path


if __name__ == "__main__":
    for agent_id, model_step, agent_step in AGENTS:
        for dataset_id in DATASETS:
            task_id = _task_id(dataset_id, agent_id)
            agent_steps = _steps(agent_step)
            path = test(task_id, dataset_id, agent_id, model_step, agent_steps)
            returns_rows = eval(task_id, path)
            plot(task_id, returns_rows)

    agent_ids = [agent_id for agent_id, _, _ in AGENTS]
    save_tables(DATASETS, agent_ids)
    save_table_boxplot(TABLES)
    save_boxplots(DATASETS, agent_ids)
