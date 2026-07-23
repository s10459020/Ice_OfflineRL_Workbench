import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


SCRIPT_ROOT = Path(__file__).resolve().parent

TRAIN_SCRIPTS = {
    "base_train": {
        "path": SCRIPT_ROOT / "experiment_base" / "train_agent.py",
        "kind": "train",
    },
    "base_train_min": {
        "path": SCRIPT_ROOT / "experiment_base" / "train_min.py",
        "kind": "train_min_task_start",
    },
    "stability_train": {
        "path": SCRIPT_ROOT / "experiment_stability" / "train_agent.py",
        "kind": "train",
    },
    "stability_train_min": {
        "path": SCRIPT_ROOT / "experiment_stability" / "train_min.py",
        "kind": "train_min_flat",
    },
    "hybrid_random_train": {
        "path": SCRIPT_ROOT / "experiment_hybrid_random" / "train.py",
        "kind": "train",
    },
    "hybrid_random_train_min": {
        "path": SCRIPT_ROOT / "experiment_hybrid_random" / "train_min.py",
        "kind": "train_min_task_start",
    },
}

TASKS = [
    # format: (run_name, task_steps, dataset_id, agent_id, agent_kwargs)
    # train task_steps: [model_start, agent_start, train_steps]
    # train_min task_steps: [model_start, agent_start]
    # ("stability_train", [100_000, 0, 200_000], "walker2d_replay_medium", "scaspl_param", {}),
    # ("stability_train_min", [100_000, 500_000], "walker2d_d4rl_medium", "scc", {}),
    # ("stability_train_min", [100_000, 500_000], "walker2d_d4rl_hybrid", "scc", {}),
    # ("stability_train_min", [100_000, 500_000], "walker2d_d4rl_expert", "scc", {}),
    # ("stability_train_min", [100_000, 500_000], "walker2d_replay_medium", "scc", {}),
    # ("stability_train_min", [100_000, 500_000], "walker2d_replay_expert", "scc", {}),
    # ("base_train_min", [100_000, 500_000], "hopper_d4rl_medium", "scaspl_n", {}),
    # ("base_train_min", [100_000, 500_000], "hopper_d4rl_expert", "scaspl_n", {}),
    # ("base_train_min", [100_000, 500_000], "hopper_d4rl_hybrid", "scaspl_n", {}),
    # ("base_train_min", [100_000, 500_000], "hopper_replay_medium", "scaspl_n", {}),
    # ("base_train_min", [100_000, 500_000], "hopper_replay_expert", "scaspl_n", {}),
    # ("base_train_min", [100_000, 500_000], "halfcheetah_d4rl_medium", "scaspl_n", {}),
    # ("base_train_min", [100_000, 500_000], "halfcheetah_d4rl_expert", "scaspl_n", {}),
    # ("base_train_min", [100_000, 500_000], "halfcheetah_d4rl_hybrid", "scaspl_n", {}),
    # ("base_train_min", [100_000, 500_000], "halfcheetah_replay_medium", "scaspl_n", {}),
    # ("base_train_min", [100_000, 500_000], "halfcheetah_replay_expert", "scaspl_n", {}),
    # ("base_train_min", [100_000, 500_000], "hopper_d4rl_medium", "scc_n", {}),
    # ("base_train_min", [100_000, 500_000], "hopper_d4rl_expert", "scc_n", {}),
    # ("base_train_min", [100_000, 500_000], "hopper_d4rl_hybrid", "scc_n", {}),
    # ("base_train_min", [100_000, 500_000], "hopper_replay_medium", "scc_n", {}),
    # ("base_train_min", [100_000, 500_000], "hopper_replay_expert", "scc_n", {}),
    # ("base_train_min", [100_000, 500_000], "halfcheetah_d4rl_medium", "scc_n", {}),
    # ("base_train_min", [100_000, 500_000], "halfcheetah_d4rl_expert", "scc_n", {}),
    # ("base_train_min", [100_000, 500_000], "halfcheetah_d4rl_hybrid", "scc_n", {}),
    # ("base_train_min", [100_000, 500_000], "halfcheetah_replay_medium", "scc_n", {}),
    # ("base_train_min", [100_000, 500_000], "halfcheetah_replay_expert", "scc_n", {}),
    ("hybrid_random_train", [None, 0, 50_000], "walker2d_random_expert_1", "bc", {}),
    ("hybrid_random_train", [None, 0, 100_000], "walker2d_random_expert_1", "td3bc_n", {}),
    ("hybrid_random_train", [None, 0, 200_000], "walker2d_random_expert_1", "iql", {}),
    ("hybrid_random_train", [None, 0, 500_000], "walker2d_random_expert_1", "cql", {}),
    ("hybrid_random_train", [None, 0, 500_000], "walker2d_random_expert_1", "aspl_c", {}),
    ("hybrid_random_train", [500_000, 0, 500_000], "walker2d_random_expert_1", "scc_n", {}),
    ("hybrid_random_train", [500_000, 0, 500_000], "walker2d_random_expert_3", "scc_n", {}),
    ("hybrid_random_train", [500_000, 0, 500_000], "walker2d_random_expert_5", "scc_n", {}),
    ("hybrid_random_train", [500_000, 0, 500_000], "walker2d_random_expert_7", "scc_n", {}),
    ("hybrid_random_train", [500_000, 0, 500_000], "walker2d_random_expert_9", "scc_n", {}),
    ("hybrid_random_train_min", [None, 50_000], "walker2d_random_expert_1", "bc", {}),
    ("hybrid_random_train_min", [None, 50_000], "walker2d_random_expert_3", "bc", {}),
    ("hybrid_random_train_min", [None, 50_000], "walker2d_random_expert_5", "bc", {}),
    ("hybrid_random_train_min", [None, 50_000], "walker2d_random_expert_7", "bc", {}),
    ("hybrid_random_train_min", [None, 50_000], "walker2d_random_expert_9", "bc", {}),
    ("hybrid_random_train_min", [None, 100_000], "walker2d_random_expert_1", "td3bc_n", {}),
    ("hybrid_random_train_min", [None, 100_000], "walker2d_random_expert_3", "td3bc_n", {}),
    ("hybrid_random_train_min", [None, 100_000], "walker2d_random_expert_5", "td3bc_n", {}),
    ("hybrid_random_train_min", [None, 100_000], "walker2d_random_expert_7", "td3bc_n", {}),
    ("hybrid_random_train_min", [None, 100_000], "walker2d_random_expert_9", "td3bc_n", {}),
    ("hybrid_random_train_min", [None, 200_000], "walker2d_random_expert_1", "iql", {}),
    ("hybrid_random_train_min", [None, 200_000], "walker2d_random_expert_3", "iql", {}),
    ("hybrid_random_train_min", [None, 200_000], "walker2d_random_expert_5", "iql", {}),
    ("hybrid_random_train_min", [None, 200_000], "walker2d_random_expert_7", "iql", {}),
    ("hybrid_random_train_min", [None, 200_000], "walker2d_random_expert_9", "iql", {}),
    ("hybrid_random_train_min", [None, 500_000], "walker2d_random_expert_1", "cql", {}),
    ("hybrid_random_train_min", [None, 500_000], "walker2d_random_expert_3", "cql", {}),
    ("hybrid_random_train_min", [None, 500_000], "walker2d_random_expert_5", "cql", {}),
    ("hybrid_random_train_min", [None, 500_000], "walker2d_random_expert_7", "cql", {}),
    ("hybrid_random_train_min", [None, 500_000], "walker2d_random_expert_9", "cql", {}),
    ("hybrid_random_train_min", [None, 500_000], "walker2d_random_expert_1", "aspl_c", {}),
    ("hybrid_random_train_min", [None, 500_000], "walker2d_random_expert_3", "aspl_c", {}),
    ("hybrid_random_train_min", [None, 500_000], "walker2d_random_expert_5", "aspl_c", {}),
    ("hybrid_random_train_min", [None, 500_000], "walker2d_random_expert_7", "aspl_c", {}),
    ("hybrid_random_train_min", [None, 500_000], "walker2d_random_expert_9", "aspl_c", {}),
]

DEFAULT_RUNS = tuple(TRAIN_SCRIPTS.keys())
LOCAL_MODULE_NAMES = ("plot",)
LOADED_MODULES = {}


def _load_train_module(run_name: str, script_path: Path) -> ModuleType:
    module_name = f"_trains_{run_name}"
    previous_path = list(sys.path)
    previous_modules = {local_module_name: sys.modules.get(local_module_name) for local_module_name in LOCAL_MODULE_NAMES}

    for local_module_name in LOCAL_MODULE_NAMES:
        sys.modules.pop(local_module_name, None)

    sys.modules.pop(module_name, None)
    sys.path.insert(0, str(script_path.parent))

    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    sys.path[:] = previous_path
    for local_module_name, previous_module in previous_modules.items():
        sys.modules.pop(local_module_name, None)
        if previous_module is not None:
            sys.modules[local_module_name] = previous_module

    return module


def _train_module(run_name: str) -> ModuleType:
    if run_name not in LOADED_MODULES:
        script_spec = TRAIN_SCRIPTS[run_name]
        LOADED_MODULES[run_name] = _load_train_module(run_name, script_spec["path"])
    return LOADED_MODULES[run_name]


def _selected_tasks(run_names: tuple[str, ...]) -> list[tuple[str, list[int | None], str, str, dict]]:
    return [task for task in TASKS if task[0] in run_names]


def _print_tasks(run_names: tuple[str, ...]) -> None:
    for run_name in run_names:
        tasks = _selected_tasks((run_name,))
        print(f"{run_name}: {len(tasks)} task(s)")
        for task in tasks:
            print(f"  {task}")


def _run_train(train_module: ModuleType, task_steps: list[int | None], dataset_id: str, agent_id: str, agent_kwargs: dict) -> None:
    task_id = train_module.train(task_steps, dataset_id, agent_id, agent_kwargs)
    train_module.analyze(task_id, train_module.eval_path(task_id))
    train_module.plot_train(
        task_id,
        train_module.metric_path(task_id),
        [train_module.returns_path(task_id), train_module.steps_path(task_id)],
        dataset_id,
        agent_id,
    )


def _run_train_min(train_module: ModuleType, kind: str, task_steps: list[int | None], dataset_id: str, agent_id: str, agent_kwargs: dict) -> None:
    model_start, agent_start = task_steps
    if kind == "train_min_task_start":
        train_module.train_min_agent(task_steps, dataset_id, agent_id, agent_kwargs)
    else:
        train_module.train_min_agent(dataset_id, agent_id, model_start, agent_start)


def _run_task(task: tuple[str, list[int | None], str, str, dict]) -> None:
    run_name, task_steps, dataset_id, agent_id, agent_kwargs = task
    train_module = _train_module(run_name)
    kind = TRAIN_SCRIPTS[run_name]["kind"]

    print(f"start {run_name}: {task_steps}, {dataset_id}, {agent_id}")
    if kind == "train":
        _run_train(train_module, task_steps, dataset_id, agent_id, agent_kwargs)
    else:
        _run_train_min(train_module, kind, task_steps, dataset_id, agent_id, agent_kwargs)
    print(f"done {run_name}: {dataset_id}, {agent_id}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        choices=TRAIN_SCRIPTS.keys(),
        default=[],
    )
    parser.add_argument(
        "--list",
        action="store_true",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_names = tuple(args.run) or DEFAULT_RUNS
    tasks = _selected_tasks(run_names)

    if args.list:
        _print_tasks(run_names)
    else:
        print(f"start trains: {len(tasks)} task(s)")
        for task in tasks:
            _run_task(task)
        print("done trains")
