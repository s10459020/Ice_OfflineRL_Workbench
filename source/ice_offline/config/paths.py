from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASETS_ROOT = PROJECT_ROOT / "tmps" / "datasets"
CUSTOM_DATASETS_ROOT = DATASETS_ROOT / "custom"
RUNS_ROOT = PROJECT_ROOT / "tmps" / "runs"
MODELS_ROOT = PROJECT_ROOT / "tmps" / "models"
RETURNS_ROOT = PROJECT_ROOT / "tmps" / "returns"
STEPS_ROOT = PROJECT_ROOT / "tmps" / "steps"
EVALS_ROOT = PROJECT_ROOT / "tmps" / "evals"
METRICS_ROOT = PROJECT_ROOT / "tmps" / "metrics"
VIEW_ROOT = PROJECT_ROOT / "tmps" / "view"


def _task_id(
    dataset_id: str,
    agent_id: str | None = None,
    postfix: object | None = None,
) -> str:
    if agent_id is None:
        return f"{dataset_id}-v0"
    if postfix is None:
        return f"{dataset_id}-{agent_id}-v0"
    return f"{dataset_id}-{agent_id}-{postfix}-v0"


def eval_data_path(mode: str, task_id: str) -> Path:
    return RUNS_ROOT / mode / task_id / "data" / "eval_data.hdf5"


def custom_dataset_path(dataset_id: str) -> Path:
    return CUSTOM_DATASETS_ROOT / dataset_id / "data" / "main_data.hdf5"


def minari_dataset_path(dataset_id: str) -> Path:
    return DATASETS_ROOT / dataset_id / "data" / "main_data.hdf5"


def d4rl_dataset_path(dataset_id: str) -> Path:
    return DATASETS_ROOT / "d4rl" / dataset_id / "d4rl_data.hdf5"


def main_data_path(mode: str, task_id: str) -> Path:
    return RUNS_ROOT / mode / task_id / "data" / "main_data.hdf5"


def model_path(task_id: str, step: int) -> Path:
    return MODELS_ROOT / task_id / str(step)


def eval_path(task_id: str) -> Path:
    return EVALS_ROOT / f"{task_id}.csv"


def returns_path(mode: str, task_id: str) -> Path:
    return RETURNS_ROOT / mode / f"{task_id}.csv"


def steps_path(mode: str, task_id: str) -> Path:
    return STEPS_ROOT / mode / f"{task_id}.csv"


def metric_path(task_id: str) -> Path:
    return METRICS_ROOT / f"{task_id}.csv"


def plot_path(task_group: str, dataset_id: str, agent_id: str) -> Path:
    return VIEW_ROOT / "plot" / task_group / agent_id / f"{dataset_id}.png"


def table_path(group: str, name: str) -> Path:
    return VIEW_ROOT / "table" / group / name
