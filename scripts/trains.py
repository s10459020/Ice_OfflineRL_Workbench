import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


SCRIPT_ROOT = Path(__file__).resolve().parent
AGENT_ROOT = SCRIPT_ROOT.parent / "source" / "ice_offline" / "agent"
INVALIDATED_AGENTS = {
    "aspl",
    "aspl_gp",
    "aspl_c",
    "scas_n",
    "scas_gp",
    "scas_gpn",
    "scaspl_p",
    "scaspl_pn",
    "scaspl_pgp",
    "scaspl_pc",
    "scaspl_pnc",
}

AGENT_SOURCE_DEPENDENCIES = {
    "aspl_gp": ("aspl.py", "aspl_gp.py"),
    "aspl_c": ("aspl.py", "aspl_c.py"),
    "scas_n": ("scas.py", "scas_n.py"),
    "scas_gp": ("scas.py", "scas_gp.py"),
    "scas_gpn": ("scas.py", "scas_n.py", "scas_gp.py", "scas_gpn.py"),
    "scaspl_p": ("scaspl.py", "scaspl_p.py", "scas.py", "aspl.py"),
    "scaspl_pn": ("scaspl.py", "scaspl_n.py", "scaspl_pn.py", "scas.py", "aspl.py"),
    "scaspl_pgp": ("scaspl.py", "scaspl_p.py", "scaspl_pgp.py", "scaspl_gp.py", "scas.py", "aspl.py"),
    "scaspl_pc": ("scaspl.py", "scaspl_p.py", "scaspl_pc.py", "aspl_c.py", "scas.py", "aspl.py"),
    "scaspl_pnc": ("scaspl.py", "scaspl_n.py", "scaspl_pn.py", "scaspl_pnc.py", "aspl_c.py", "scas.py", "aspl.py"),
}

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
    # ("base_train_min", [500_000, 200_000], "walker2d_d4rl_medium", "scas_n", {}),
    # ("base_train_min", [500_000, 200_000], "walker2d_d4rl_hybrid", "scas_n", {}),
    # ("base_train_min", [500_000, 200_000], "walker2d_replay_medium", "scas_n", {}),
    # ("base_train", [500_000, 0, 200_000], "walker2d_replay_expert", "scas_n", {}),
    # ("base_train", [500_000, 0, 200_000], "halfcheetah_d4rl_medium", "scas_n", {}),
    # ("base_train", [500_000, 0, 200_000], "halfcheetah_d4rl_expert", "scas_n", {}),
    # ("base_train", [500_000, 0, 200_000], "halfcheetah_d4rl_hybrid", "scas_n", {}),
    # ("base_train", [500_000, 0, 200_000], "halfcheetah_replay_medium", "scas_n", {}),
    # ("base_train", [500_000, 0, 200_000], "halfcheetah_replay_expert", "scas_n", {}),
    # ("hybrid_random_train", [500_000, 0, 200_000], "walker2d_random_expert_1", "scas_n", {}),
    # ("hybrid_random_train", [500_000, 0, 200_000], "walker2d_random_expert_3", "scas_n", {}),
    # ("hybrid_random_train", [500_000, 0, 200_000], "walker2d_random_expert_5", "scas_n", {}),
    # ("hybrid_random_train", [500_000, 0, 200_000], "walker2d_random_expert_7", "scas_n", {}),
    # ("hybrid_random_train", [500_000, 0, 200_000], "walker2d_random_expert_9", "scas_n", {}),
    # ("base_train", [100_000, 0, 500_000], "walker2d_d4rl_medium", "scaspl_p", {}),
    # ("base_train", [100_000, 200_000, 500_000], "walker2d_d4rl_expert", "scaspl_p", {}),
    # ("base_train", [100_000, 0, 500_000], "walker2d_d4rl_hybrid", "scaspl_p", {}),
    # ("base_train", [100_000, 200_000, 500_000], "walker2d_replay_medium", "scaspl_p", {}),
    # ("base_train", [100_000, 0, 500_000], "walker2d_replay_expert", "scaspl_p", {}),
    # ("base_train", [500_000, 210_000, 500_000], "walker2d_d4rl_medium", "scaspl_pn", {}),
    # ("base_train", [500_000, 210_000, 500_000], "walker2d_d4rl_hybrid", "scaspl_pn", {}),
    # ("base_train", [500_000, 210_000, 500_000], "walker2d_replay_expert", "scaspl_pn", {}),
    # ("base_train", [500_000, 0, 200_000], "walker2d_d4rl_medium", "scaspl_pgp", {}),
    ("base_train", [500_000, 0, 200_000], "walker2d_d4rl_expert", "scaspl_pgp", {}),
    ("base_train", [500_000, 0, 200_000], "walker2d_d4rl_hybrid", "scaspl_pgp", {}),
    ("base_train", [500_000, 0, 200_000], "walker2d_replay_medium", "scaspl_pgp", {}),
    ("base_train", [500_000, 0, 200_000], "walker2d_replay_expert", "scaspl_pgp", {}),
    # ("base_train", [500_000, 0, 200_000], "walker2d_d4rl_medium", "scaspl_pc", {}),
    # ("base_train", [500_000, 0, 200_000], "walker2d_d4rl_expert", "scaspl_pc", {}),
    # ("base_train", [500_000, 0, 200_000], "walker2d_d4rl_hybrid", "scaspl_pc", {}),
    # ("base_train", [500_000, 0, 200_000], "walker2d_replay_medium", "scaspl_pc", {}),
    # ("base_train", [500_000, 0, 200_000], "walker2d_replay_expert", "scaspl_pc", {}),
    # ("base_train", [500_000, 0, 200_000], "walker2d_d4rl_medium", "scaspl_pnc", {}),
    # ("base_train", [500_000, 0, 200_000], "walker2d_d4rl_expert", "scaspl_pnc", {}),
    # ("base_train", [500_000, 0, 200_000], "walker2d_d4rl_hybrid", "scaspl_pnc", {}),
    # ("base_train", [500_000, 0, 200_000], "walker2d_replay_medium", "scaspl_pnc", {}),
    # ("base_train", [500_000, 0, 200_000], "walker2d_replay_expert", "scaspl_pnc", {}),
    # ("base_train", [500_000, 0, 200_000], "walker2d_d4rl_medium", "scas_gpn", {}),
    # ("base_train", [500_000, 0, 200_000], "walker2d_d4rl_expert", "scas_gpn", {}),
    # ("base_train", [500_000, 0, 200_000], "walker2d_d4rl_hybrid", "scas_gpn", {}),
    # ("base_train", [500_000, 0, 200_000], "walker2d_replay_medium", "scas_gpn", {}),
    # ("base_train", [500_000, 0, 200_000], "walker2d_replay_expert", "scas_gpn", {}),
    # ("base_train", [None, 0, 200_000], "hopper_d4rl_medium", "aspl_c", {}),
    # ("base_train", [None, 0, 200_000], "hopper_d4rl_expert", "aspl_c", {}),
    # ("base_train", [None, 0, 200_000], "hopper_d4rl_hybrid", "aspl_c", {}),
    # ("base_train", [None, 0, 200_000], "hopper_replay_medium", "aspl_c", {}),
    # ("base_train", [None, 0, 200_000], "hopper_replay_expert", "aspl_c", {}),
    # ("base_train", [None, 0, 200_000], "halfcheetah_d4rl_medium", "aspl_c", {}),
    # ("base_train", [None, 0, 200_000], "halfcheetah_d4rl_expert", "aspl_c", {}),
    # ("base_train", [None, 0, 200_000], "halfcheetah_d4rl_hybrid", "aspl_c", {}),
    # ("base_train", [None, 0, 200_000], "halfcheetah_replay_medium", "aspl_c", {}),
    # ("base_train", [None, 0, 200_000], "halfcheetah_replay_expert", "aspl_c", {}),
    # ("base_train", [500_000, 0, 200_000], "hopper_d4rl_medium", "scaspl_pn", {}),
    # ("base_train", [500_000, 0, 200_000], "hopper_d4rl_expert", "scaspl_pn", {}),
    # ("base_train", [500_000, 0, 200_000], "hopper_d4rl_hybrid", "scaspl_pn", {}),
    # ("base_train", [500_000, 0, 200_000], "hopper_replay_medium", "scaspl_pn", {}),
    # ("base_train", [500_000, 0, 200_000], "hopper_replay_expert", "scaspl_pn", {}),
    # ("base_train", [500_000, 0, 200_000], "halfcheetah_d4rl_medium", "scaspl_pn", {}),
    # ("base_train", [500_000, 0, 200_000], "halfcheetah_d4rl_expert", "scaspl_pn", {}),
    # ("base_train", [500_000, 0, 200_000], "halfcheetah_d4rl_hybrid", "scaspl_pn", {}),
    # ("base_train", [500_000, 0, 200_000], "halfcheetah_replay_medium", "scaspl_pn", {}),
    # ("base_train", [500_000, 0, 200_000], "halfcheetah_replay_expert", "scaspl_pn", {}),
]

DEFAULT_RUNS = tuple(dict.fromkeys(task[0] for task in TASKS))
LOCAL_MODULE_NAMES = ("plot",)
LOADED_MODULES = {}
DEVICE = "cuda"
TRAIN_MIN_INTERVAL = 1_000
TRAIN_MIN_COUNT = 10


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
        train_module = _load_train_module(run_name, script_spec["path"])
        train_module.DEVICE = DEVICE
        if hasattr(train_module, "INTERVAL"):
            train_module.INTERVAL = TRAIN_MIN_INTERVAL
        if hasattr(train_module, "COUNT"):
            train_module.COUNT = TRAIN_MIN_COUNT
        LOADED_MODULES[run_name] = train_module
    return LOADED_MODULES[run_name]


def _selected_tasks(run_names: tuple[str, ...]) -> list[tuple[str, list[int | None], str, str, dict]]:
    return [task for task in TASKS if task[0] in run_names]


def _task_id(train_module: ModuleType, agent_id: str, dataset_id: str) -> str:
    return train_module.experiment_task_id(train_module.EXPERIMENT_TRAIN, agent_id, dataset_id)


def _agent_source_paths(agent_id: str) -> tuple[Path, ...]:
    source_names = AGENT_SOURCE_DEPENDENCIES.get(agent_id, (f"{agent_id}.py",))
    return tuple(AGENT_ROOT / source_name for source_name in source_names)


def _is_stale(agent_id: str, path: Path) -> bool:
    if agent_id not in INVALIDATED_AGENTS:
        return False
    source_paths = _agent_source_paths(agent_id)
    if not path.exists() or not all(source_path.exists() for source_path in source_paths):
        return False
    source_mtime = max(source_path.stat().st_mtime for source_path in source_paths)
    return path.stat().st_mtime < source_mtime


def _model_ready(agent_id: str, path: Path) -> bool:
    return path.exists() and not _is_stale(agent_id, path)


def _model_id(train_module: ModuleType, dataset_id: str) -> str:
    return train_module.experiment_task_id(train_module.EXPERIMENT_TRAIN, "scas_model", dataset_id)


def _checkpoint_steps(train_module: ModuleType, task_id: str) -> list[int]:
    model_dir = train_module.model_path(task_id, 0).parent
    if not model_dir.exists():
        return []
    steps = []
    for path in model_dir.glob("*.pt"):
        try:
            steps.append(int(path.stem))
        except ValueError:
            pass
    return sorted(steps)


def _train_min_required_steps(agent_start: int) -> list[int]:
    return [
        agent_start + TRAIN_MIN_INTERVAL * index
        for index in range(1, TRAIN_MIN_COUNT + 1)
    ]


def _task_status(task: tuple[str, list[int | None], str, str, dict]) -> str:
    run_name, task_steps, dataset_id, agent_id, _ = task
    train_module = _train_module(run_name)
    kind = TRAIN_SCRIPTS[run_name]["kind"]
    task_id = _task_id(train_module, agent_id, dataset_id)
    model_start = task_steps[0]
    agent_start = task_steps[1]

    if model_start is not None:
        model_id = _model_id(train_module, dataset_id)
        if not train_module.model_path(model_id, model_start).exists():
            return "blocked:model"

    if kind == "train":
        train_steps = task_steps[2]
        if agent_start > 0 and not _model_ready(agent_id, train_module.model_path(task_id, agent_start)):
            return "blocked:agent"
        final_path = train_module.model_path(task_id, train_steps)
        if final_path.exists() and _is_stale(agent_id, final_path):
            return "stale"
        if final_path.exists():
            return "ok"
        steps = [
            step
            for step in _checkpoint_steps(train_module, task_id)
            if step <= train_steps and _model_ready(agent_id, train_module.model_path(task_id, step))
        ]
        if steps:
            return f"partial:{steps[-1]}/{train_steps}"
        return "missing"

    if agent_start > 0 and not _model_ready(agent_id, train_module.model_path(task_id, agent_start)):
        return "blocked:agent"
    required_steps = _train_min_required_steps(agent_start)
    present_steps = [
        step
        for step in required_steps
        if _model_ready(agent_id, train_module.model_path(task_id, step))
    ]
    if len(present_steps) == len(required_steps):
        return "ok"
    if present_steps:
        return f"partial:{len(present_steps)}/{len(required_steps)}"
    return "missing"


def _print_tasks(run_names: tuple[str, ...]) -> None:
    for run_name in run_names:
        tasks = _selected_tasks((run_name,))
        statuses = [_task_status(task) for task in tasks]
        status_counts = {}
        for status in statuses:
            status_key = status.split(":", 1)[0]
            status_counts[status_key] = status_counts.get(status_key, 0) + 1
        status_text = ", ".join(
            f"{status_key}={count}"
            for status_key, count in sorted(status_counts.items())
        )
        if status_text:
            status_text = f" ({status_text})"
        print(f"{run_name}: {len(tasks)} task(s){status_text}")
        for task, status in zip(tasks, statuses):
            print(f"  [{status}] {task}")


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
    parser.add_argument(
        "--device",
        default=DEVICE,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    DEVICE = args.device
    run_names = tuple(args.run) or DEFAULT_RUNS
    tasks = _selected_tasks(run_names)

    if args.list:
        _print_tasks(run_names)
    else:
        print(f"start trains: {len(tasks)} task(s), device={DEVICE}")
        for task in tasks:
            _run_task(task)
        print("done trains")
