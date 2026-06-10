from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASETS_ROOT = PROJECT_ROOT / "tmps" / "datasets"
CUSTOM_DATASETS_ROOT = DATASETS_ROOT / "custom"
RUNS_ROOT = PROJECT_ROOT / "tmps" / "runs"
MODELS_ROOT = PROJECT_ROOT / "tmps" / "models"
RETURNS_ROOT = PROJECT_ROOT / "tmps" / "returns"
EVALS_ROOT = PROJECT_ROOT / "tmps" / "evals"
METRICS_ROOT = PROJECT_ROOT / "tmps" / "metrics"
VIEW_ROOT = PROJECT_ROOT / "tmps" / "view"


def _task_id(dataset_id: str, agent_id: str | None = None) -> str:
    if agent_id is None:
        return f"{dataset_id}-v0"
    return f"{dataset_id}-{agent_id}-v0"


def data_path_train(dataset_id: str, agent_id: str) -> Path:
    return RUNS_ROOT / "train" / _task_id(dataset_id, agent_id) / "data" / "eval_data.hdf5"


def custom_dataset_path(dataset_id: str) -> Path:
    return CUSTOM_DATASETS_ROOT / dataset_id / "data" / "main_data.hdf5"


def data_path_test(dataset_id: str, agent_id: str) -> Path:
    return RUNS_ROOT / "test" / _task_id(dataset_id, agent_id) / "data" / "main_data.hdf5"


def data_path_collect(dataset_id: str, agent_id: str) -> Path:
    return RUNS_ROOT / "collect" / _task_id(dataset_id, agent_id) / "data" / "main_data.hdf5"


def data_path_probe(dataset_id: str, agent_id: str) -> Path:
    return RUNS_ROOT / "probe" / _task_id(dataset_id, agent_id) / "data" / "main_data.hdf5"


def model_path(dataset_id: str, agent_id: str, step: int) -> Path:
    return MODELS_ROOT / _task_id(dataset_id, agent_id) / str(step)


def eval_path(dataset_id: str, agent_id: str) -> Path:
    return EVALS_ROOT / f"{_task_id(dataset_id, agent_id)}.csv"


def metric_path(dataset_id: str, agent_id: str) -> Path:
    return METRICS_ROOT / f"{_task_id(dataset_id, agent_id)}.csv"


def returns_path(dataset_id: str, agent_id: str | None = None) -> Path:
    return RETURNS_ROOT / f"{_task_id(dataset_id, agent_id)}.json"
