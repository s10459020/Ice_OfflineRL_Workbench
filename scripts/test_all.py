import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


SCRIPT_ROOT = Path(__file__).resolve().parent

TEST_SCRIPTS = {
    "noise_init": SCRIPT_ROOT / "experiment_noise_init" / "test.py",
    "noise_action": SCRIPT_ROOT / "experiment_noise_action" / "test.py",
    "noise_state": SCRIPT_ROOT / "experiment_noise_state" / "test.py",
}

DEFAULT_EXPERIMENTS = ("noise_init", "noise_action", "noise_state")
DEFAULT_AGENTS = ("bc", "iql", "cql", "aspl_c", "scc_n")
LOCAL_MODULE_NAMES = ("plot", "view")


def _load_test_module(experiment_name: str, script_path: Path) -> ModuleType:
    module_name = f"_test_all_{experiment_name}"
    previous_path = list(sys.path)
    previous_modules = {module_name: sys.modules.get(module_name) for module_name in LOCAL_MODULE_NAMES}

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


def _agent_specs(test_module: ModuleType, agent_ids: tuple[str, ...]) -> list[tuple[str, int | None, int]]:
    specs = []
    for agent_id in agent_ids:
        specs.extend(agent_spec for agent_spec in test_module.AGENTS if agent_spec[0] == agent_id)
    return specs


def _run_experiment(experiment_name: str, agent_ids: tuple[str, ...]) -> None:
    test_module = _load_test_module(experiment_name, TEST_SCRIPTS[experiment_name])
    agent_specs = _agent_specs(test_module, agent_ids)
    dataset_ids = [dataset_spec[0] for dataset_spec in test_module.DATASETS]
    table_agent_ids = [agent_spec[0] for agent_spec in agent_specs]

    print(f"start experiment={experiment_name}, agents={','.join(table_agent_ids)}")
    for dataset_spec in test_module.DATASETS:
        test_dataset_id = dataset_spec[0]
        for agent_id, model_step, agent_step in agent_specs:
            test_id = test_module.test(*dataset_spec, agent_id, model_step, agent_step)
            test_module.analyze(test_id, test_module.eval_path(test_id))
            test_module.plot(test_id, test_module.returns_path(test_id), test_dataset_id, agent_id)

    test_module.save_tables(dataset_ids, table_agent_ids)
    test_module.save_boxplots(dataset_ids, table_agent_ids)
    print(f"done experiment={experiment_name}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        action="append",
        choices=TEST_SCRIPTS.keys(),
        default=[],
    )
    parser.add_argument(
        "--agent",
        action="append",
        default=[],
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    experiment_names = tuple(args.experiment) or DEFAULT_EXPERIMENTS
    agent_ids = tuple(args.agent) or DEFAULT_AGENTS

    for experiment_name in experiment_names:
        _run_experiment(experiment_name, agent_ids)
