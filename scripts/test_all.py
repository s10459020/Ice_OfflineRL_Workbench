import argparse
import csv
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


SCRIPT_ROOT = Path(__file__).resolve().parent

TEST_SCRIPTS = {
    "base": SCRIPT_ROOT / "experiment_base" / "test.py",
    "stability": SCRIPT_ROOT / "experiment_stability" / "test.py",
    "noise_init": SCRIPT_ROOT / "experiment_noise_init" / "test.py",
    "noise_action": SCRIPT_ROOT / "experiment_noise_action" / "test.py",
    "noise_state": SCRIPT_ROOT / "experiment_noise_state" / "test.py",
    "hybrid_random": SCRIPT_ROOT / "experiment_hybrid_random" / "test.py",
}

TASKS = [
    # format: (run_name, dataset_id, agent_id, model_step, agent_step)
    # aspl/scaspl-series agents are intentionally excluded.
    ("noise_init", "noise_init_5e-2@walker2d_replay_medium", "scas_n", 100_000, 500_000),
    ("noise_init", "noise_init_1e-1@walker2d_replay_medium", "scas_n", 100_000, 500_000),
    ("noise_init", "noise_init_5e-1@walker2d_replay_medium", "scas_n", 100_000, 500_000),
    ("noise_init", "noise_init_1e0@walker2d_replay_medium", "scas_n", 100_000, 500_000),
    ("noise_action", "noise_action_5e-2@walker2d_replay_medium", "scas_n", 100_000, 500_000),
    ("noise_action", "noise_action_1e-1@walker2d_replay_medium", "scas_n", 100_000, 500_000),
    ("noise_action", "noise_action_5e-1@walker2d_replay_medium", "scas_n", 100_000, 500_000),
    ("noise_action", "noise_action_1e0@walker2d_replay_medium", "scas_n", 100_000, 500_000),
    ("noise_state", "noise_state_5e-4@walker2d_replay_medium", "scas_n", 100_000, 500_000),
    ("noise_state", "noise_state_1e-3@walker2d_replay_medium", "scas_n", 100_000, 500_000),
    ("noise_state", "noise_state_5e-3@walker2d_replay_medium", "scas_n", 100_000, 500_000),
    ("noise_state", "noise_state_1e-2@walker2d_replay_medium", "scas_n", 100_000, 500_000),
]

DEFAULT_RUNS = tuple(TEST_SCRIPTS.keys())
LOCAL_MODULE_NAMES = ("plot", "view", "train_min")
LOADED_MODULES = {}


def _load_test_module(run_name: str, script_path: Path) -> ModuleType:
    module_name = f"_tests_{run_name}"
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


def _test_module(run_name: str) -> ModuleType:
    if run_name not in LOADED_MODULES:
        LOADED_MODULES[run_name] = _load_test_module(run_name, TEST_SCRIPTS[run_name])
    return LOADED_MODULES[run_name]


def _selected_tasks(run_names: tuple[str, ...]) -> list[tuple[str, str, str, int | None, int]]:
    selected = []
    seen = set()
    for task in TASKS:
        if task[0] not in run_names:
            continue
        if task in seen:
            continue
        selected.append(task)
        seen.add(task)
    return selected


def _dataset_spec(test_module: ModuleType, dataset_id: str) -> tuple:
    for dataset_spec in test_module.DATASETS:
        if dataset_spec == dataset_id:
            return (dataset_spec,)
        if isinstance(dataset_spec, tuple) and dataset_spec[0] == dataset_id:
            return dataset_spec
    return (dataset_id,)


def _return_status(test_module: ModuleType, dataset_id: str, agent_id: str, start_step: int) -> str:
    test_id = test_module.experiment_task_id(test_module.EXPERIMENT, agent_id, dataset_id)
    path_returns = test_module.returns_path(test_id)
    path_eval = test_module.eval_path(test_id)
    expected_last = start_step + test_module.INTERVAL * test_module.COUNT

    if not path_returns.exists() or not path_eval.exists():
        return "missing"

    rows = 0
    first_step = None
    last_step = None
    with path_returns.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            rows += 1
            step = int(float(row[0]))
            if first_step is None:
                first_step = step
            last_step = step

    if rows >= test_module.COUNT + 1 and first_step == start_step and last_step == expected_last:
        return "refresh"
    return f"partial({first_step}..{last_step})"


def _print_tasks(run_names: tuple[str, ...]) -> None:
    for run_name in run_names:
        test_module = _test_module(run_name)
        tasks = _selected_tasks((run_name,))
        task_statuses = []
        for task in tasks:
            _, dataset_id, agent_id, model_step, agent_step = task
            status = _return_status(test_module, dataset_id, agent_id, agent_step)
            task_statuses.append((task, status))

        missing_count = sum(status == "missing" for _, status in task_statuses)
        refresh_count = sum(status == "refresh" for _, status in task_statuses)
        partial_count = len(task_statuses) - missing_count - refresh_count
        print(f"{run_name}: {len(tasks)} task(s), missing={missing_count}, refresh={refresh_count}, partial={partial_count}")
        for task, status in task_statuses:
            print(f"  [{status}] {task}")


def _plot_task(test_module: ModuleType, task_id: str, dataset_id: str, agent_id: str) -> None:
    path_returns = test_module.returns_path(task_id)
    if hasattr(test_module, "plot_test"):
        test_module.plot_test(task_id, path_returns, dataset_id, agent_id)
    else:
        test_module.plot(task_id, path_returns, dataset_id, agent_id)


def _run_task(task: tuple[str, str, str, int | None, int]) -> tuple[str, str, str]:
    run_name, dataset_id, agent_id, model_step, agent_step = task
    test_module = _test_module(run_name)
    dataset_spec = _dataset_spec(test_module, dataset_id)

    print(f"start {run_name}: {dataset_id}, {agent_id}, model_step={model_step}, agent_step={agent_step}")
    task_id = test_module.test(*dataset_spec, agent_id, model_step, agent_step)
    test_module.analyze(task_id, test_module.eval_path(task_id))
    _plot_task(test_module, task_id, dataset_id, agent_id)
    print(f"done {run_name}: {dataset_id}, {agent_id}")
    return run_name, dataset_id, agent_id


def _default_dataset_ids(test_module: ModuleType, executed_dataset_ids: list[str]) -> list[str]:
    dataset_ids = []
    for dataset_spec in test_module.DATASETS:
        if isinstance(dataset_spec, tuple):
            dataset_id = dataset_spec[0]
        else:
            dataset_id = dataset_spec
        if dataset_id not in dataset_ids:
            dataset_ids.append(dataset_id)

    if dataset_ids:
        return dataset_ids
    return executed_dataset_ids


def _default_agent_ids(test_module: ModuleType, executed_agent_ids: list[str]) -> list[str]:
    agent_ids = []
    for agent_id, _, _ in test_module.AGENTS:
        if agent_id not in agent_ids:
            agent_ids.append(agent_id)

    if agent_ids:
        return agent_ids
    return executed_agent_ids


def _save_views(executed: list[tuple[str, str, str]]) -> None:
    for run_name in DEFAULT_RUNS:
        run_records = [record for record in executed if record[0] == run_name]
        if not run_records:
            continue

        test_module = _test_module(run_name)
        executed_dataset_ids = []
        executed_agent_ids = []
        for _, dataset_id, agent_id in run_records:
            if dataset_id not in executed_dataset_ids:
                executed_dataset_ids.append(dataset_id)
            if agent_id not in executed_agent_ids:
                executed_agent_ids.append(agent_id)

        dataset_ids = _default_dataset_ids(test_module, executed_dataset_ids)
        agent_ids = _default_agent_ids(test_module, executed_agent_ids)
        test_module.save_tables(dataset_ids, agent_ids)
        test_module.save_boxplots(dataset_ids, agent_ids)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        choices=TEST_SCRIPTS.keys(),
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
        print(f"start tests: {len(tasks)} task(s)")
        for task in tasks:
            executed.append(_run_task(task))
        _save_views(executed)
        print("done tests")
