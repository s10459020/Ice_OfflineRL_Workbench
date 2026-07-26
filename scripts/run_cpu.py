import argparse
import csv
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from ice_offline.config.paths import model_path


SCRIPT_ROOT = Path(__file__).resolve().parent
AGENT_ROOT = SCRIPT_ROOT.parent / "source" / "ice_offline" / "agent"
DEVICE = "cpu"

INVALIDATED_AGENTS = {
    "scas",
    "scas_gp",
    "scas_gpn",
}

AGENT_SOURCE_DEPENDENCIES = {
    "scas": ("scas.py",),
    "scas_gp": ("scas.py", "scas_gp.py"),
    "scas_gpn": ("scas.py", "scas_n.py", "scas_gp.py", "scas_gpn.py"),
}

RUN_SCRIPTS = {
    "stability_train": {
        "path": SCRIPT_ROOT / "experiment_stability" / "train_agent.py",
        "kind": "train",
    },
    "stability_train_min": {
        "path": SCRIPT_ROOT / "experiment_stability" / "train_min.py",
        "kind": "train_min",
    },
    "stability_test": {
        "path": SCRIPT_ROOT / "experiment_stability" / "test.py",
        "kind": "test",
    },
}

TASKS = [
    # format: (run_name, task_steps, dataset_id, agent_id, agent_kwargs)
    # train task_steps: [model_start, agent_start, train_steps]
    ("stability_train", [100_000, 0, 200_000], "walker2d_replay_medium", "scas_gp", {}),
]

DEFAULT_RUNS = tuple(dict.fromkeys(task[0] for task in TASKS))
LOCAL_MODULE_NAMES = ("plot", "view", "train_min")
LOADED_MODULES = {}
TRAIN_MIN_INTERVAL = 1_000
TRAIN_MIN_COUNT = 20
TEST_EVALS = 100
TRAIN_STEPS_PER_SECOND = 180.0
TEST_EPISODES_PER_SECOND = 5.0


def _load_module(run_name: str, script_path: Path) -> ModuleType:
    module_name = f"_run_cpu_{run_name}"
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


def _run_module(run_name: str) -> ModuleType:
    if run_name not in LOADED_MODULES:
        script_spec = RUN_SCRIPTS[run_name]
        module = _load_module(run_name, script_spec["path"])
        module.DEVICE = DEVICE
        LOADED_MODULES[run_name] = module
    return LOADED_MODULES[run_name]


def _selected_tasks(run_names: tuple[str, ...]) -> list[tuple[str, list[int | None], str, str, dict]]:
    return [task for task in TASKS if task[0] in run_names]


def _task_id(module: ModuleType, agent_id: str, dataset_id: str) -> str:
    return module.experiment_task_id(module.EXPERIMENT_TRAIN, agent_id, dataset_id)


def _test_id(module: ModuleType, agent_id: str, dataset_id: str) -> str:
    return module.experiment_task_id(module.EXPERIMENT, agent_id, dataset_id)


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


def _model_id(module: ModuleType, dataset_id: str) -> str:
    return module.experiment_task_id(module.EXPERIMENT_TRAIN, "scas_model", dataset_id)


def _model_path(module: ModuleType, task_id: str, step: int) -> Path:
    if hasattr(module, "model_path"):
        return module.model_path(task_id, step)
    return model_path(task_id, step)


def _checkpoint_steps(module: ModuleType, task_id: str) -> list[int]:
    model_dir = _model_path(module, task_id, 0).parent
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


def _test_required_steps(agent_start: int) -> list[int]:
    return [
        agent_start + TRAIN_MIN_INTERVAL * index
        for index in range(TRAIN_MIN_COUNT + 1)
    ]


def _model_status(module: ModuleType, dataset_id: str, model_start: int | None) -> str | None:
    if model_start is None:
        return None
    model_id = _model_id(module, dataset_id)
    if not _model_path(module, model_id, model_start).exists():
        return "blocked:model"
    return None


def _train_status(module: ModuleType, task_steps: list[int | None], dataset_id: str, agent_id: str) -> str:
    model_start, agent_start, train_steps = task_steps
    blocked_status = _model_status(module, dataset_id, model_start)
    if blocked_status is not None:
        return blocked_status

    task_id = _task_id(module, agent_id, dataset_id)
    if agent_start > 0 and not _model_ready(agent_id, _model_path(module, task_id, agent_start)):
        return "blocked:agent"

    final_path = _model_path(module, task_id, train_steps)
    steps = [
        step
        for step in _checkpoint_steps(module, task_id)
        if step <= train_steps and _model_ready(agent_id, _model_path(module, task_id, step))
    ]
    if _model_ready(agent_id, final_path):
        return "ok"
    if steps:
        return f"partial:{steps[-1]}/{train_steps}"
    if final_path.exists():
        return "stale"
    return "missing"


def _train_min_status(module: ModuleType, task_steps: list[int | None], dataset_id: str, agent_id: str) -> str:
    model_start, agent_start = task_steps
    blocked_status = _model_status(module, dataset_id, model_start)
    if blocked_status is not None:
        return blocked_status

    task_id = _task_id(module, agent_id, dataset_id)
    if agent_start > 0 and not _model_ready(agent_id, _model_path(module, task_id, agent_start)):
        return "blocked:agent"

    required_steps = _train_min_required_steps(agent_start)
    present_steps = [
        step
        for step in required_steps
        if _model_ready(agent_id, _model_path(module, task_id, step))
    ]
    if len(present_steps) == len(required_steps):
        return "ok"
    if present_steps:
        return f"partial:{len(present_steps)}/{len(required_steps)}"
    return "missing"


def _return_status(module: ModuleType, dataset_id: str, agent_id: str, agent_start: int) -> str:
    test_id = _test_id(module, agent_id, dataset_id)
    path_returns = module.returns_path(test_id)
    path_eval = module.eval_path(test_id)
    expected_steps = _test_required_steps(agent_start)

    if not path_returns.exists() or not path_eval.exists():
        return "missing"

    actual_steps = []
    first_step = None
    last_step = None
    with path_returns.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            step = int(float(row[0]))
            actual_steps.append(step)
            if first_step is None:
                first_step = step
            last_step = step

    if all(step in actual_steps for step in expected_steps):
        return "ok"
    return f"partial:{first_step}..{last_step}"


def _test_status(module: ModuleType, task_steps: list[int | None], dataset_id: str, agent_id: str) -> str:
    model_start, agent_start = task_steps
    blocked_status = _model_status(module, dataset_id, model_start)
    if blocked_status is not None:
        return blocked_status

    task_id = _task_id(module, agent_id, dataset_id)
    required_steps = _test_required_steps(agent_start)
    if not all(_model_ready(agent_id, _model_path(module, task_id, step)) for step in required_steps):
        return "blocked:train_min"
    return _return_status(module, dataset_id, agent_id, agent_start)


def _task_status(task: tuple[str, list[int | None], str, str, dict]) -> str:
    run_name, task_steps, dataset_id, agent_id, _ = task
    module = _run_module(run_name)
    kind = RUN_SCRIPTS[run_name]["kind"]
    if kind == "train":
        return _train_status(module, task_steps, dataset_id, agent_id)
    if kind == "train_min":
        return _train_min_status(module, task_steps, dataset_id, agent_id)
    return _test_status(module, task_steps, dataset_id, agent_id)


def _remaining_steps(kind: str, task_steps: list[int | None], status: str) -> int:
    if status == "ok":
        return 0
    if kind == "train":
        train_steps = int(task_steps[2])
        if status.startswith("partial:"):
            done_steps = int(status.split(":", 1)[1].split("/", 1)[0])
            return max(train_steps - done_steps, 0)
        return train_steps - int(task_steps[1])
    if kind == "train_min":
        if status.startswith("partial:"):
            done_count = int(status.split(":", 1)[1].split("/", 1)[0])
            return max(TRAIN_MIN_COUNT - done_count, 0) * TRAIN_MIN_INTERVAL
        return TRAIN_MIN_COUNT * TRAIN_MIN_INTERVAL
    return 0


def _task_eta_seconds(task: tuple[str, list[int | None], str, str, dict], status: str) -> float:
    run_name, task_steps, _, _, _ = task
    kind = RUN_SCRIPTS[run_name]["kind"]
    if kind == "test":
        if status == "ok":
            return 0.0
        episodes = (TRAIN_MIN_COUNT + 1) * TEST_EVALS
        return episodes / TEST_EPISODES_PER_SECOND
    return _remaining_steps(kind, task_steps, status) / TRAIN_STEPS_PER_SECOND


def _format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _print_tasks(run_names: tuple[str, ...]) -> None:
    tasks = _selected_tasks(run_names)
    statuses = [_task_status(task) for task in tasks]
    total_eta = sum(_task_eta_seconds(task, status) for task, status in zip(tasks, statuses))

    status_counts = {}
    for status in statuses:
        status_key = status.split(":", 1)[0]
        status_counts[status_key] = status_counts.get(status_key, 0) + 1
    status_text = ", ".join(
        f"{status_key}={count}"
        for status_key, count in sorted(status_counts.items())
    )
    print(f"run_cpu: {len(tasks)} task(s), eta={_format_duration(total_eta)} ({status_text})")
    print(f"{'run':20s} {'status':18s} {'eta':8s} {'dataset':28s} {'agent':10s} steps")
    for task, status in zip(tasks, statuses):
        run_name, task_steps, dataset_id, agent_id, _ = task
        eta = _format_duration(_task_eta_seconds(task, status))
        print(f"{run_name:20s} {status:18s} {eta:8s} {dataset_id:28s} {agent_id:10s} {task_steps}")


def _run_train(module: ModuleType, task_steps: list[int | None], dataset_id: str, agent_id: str, agent_kwargs: dict) -> None:
    task_id = module.train(task_steps, dataset_id, agent_id, agent_kwargs)
    module.analyze(task_id, module.eval_path(task_id))
    module.plot_train(
        task_id,
        module.metric_path(task_id),
        [module.returns_path(task_id), module.steps_path(task_id)],
        dataset_id,
        agent_id,
    )


def _run_train_min(module: ModuleType, task_steps: list[int | None], dataset_id: str, agent_id: str) -> None:
    model_start, agent_start = task_steps
    module.train_min_agent(dataset_id, agent_id, model_start, agent_start)


def _run_test(module: ModuleType, task_steps: list[int | None], dataset_id: str, agent_id: str) -> str:
    model_start, agent_start = task_steps
    task_id = module.test(dataset_id, agent_id, model_start, agent_start)
    module.analyze(task_id, module.eval_path(task_id))
    module.plot_test(task_id, module.returns_path(task_id), dataset_id, agent_id)
    return task_id


def _run_task(task: tuple[str, list[int | None], str, str, dict]) -> tuple[str, str, str] | None:
    run_name, task_steps, dataset_id, agent_id, agent_kwargs = task
    module = _run_module(run_name)
    kind = RUN_SCRIPTS[run_name]["kind"]

    print(f"start {run_name}: {task_steps}, {dataset_id}, {agent_id}, device={DEVICE}")
    if kind == "train":
        _run_train(module, task_steps, dataset_id, agent_id, agent_kwargs)
        result = None
    elif kind == "train_min":
        _run_train_min(module, task_steps, dataset_id, agent_id)
        result = None
    else:
        _run_test(module, task_steps, dataset_id, agent_id)
        result = run_name, dataset_id, agent_id
    print(f"done {run_name}: {dataset_id}, {agent_id}")
    return result


def _save_views(executed: list[tuple[str, str, str]]) -> None:
    run_records = [record for record in executed if record[0] == "stability_test"]
    if not run_records:
        return

    module = _run_module("stability_test")
    dataset_ids = [
        "walker2d_d4rl_medium",
        "walker2d_d4rl_expert",
        "walker2d_d4rl_hybrid",
        "walker2d_replay_medium",
        "walker2d_replay_expert",
    ]
    agent_ids = []
    for _, _, agent_id in run_records:
        if agent_id not in agent_ids:
            agent_ids.append(agent_id)
    module.save_tables(dataset_ids, agent_ids)
    module.save_boxplots(dataset_ids, agent_ids)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        choices=RUN_SCRIPTS.keys(),
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
        executed = []
        print(f"start run_cpu: {len(tasks)} task(s), device={DEVICE}")
        for task in tasks:
            result = _run_task(task)
            if result is not None:
                executed.append(result)
        _save_views(executed)
        print("done run_cpu")
