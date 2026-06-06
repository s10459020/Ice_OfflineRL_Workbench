from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASETS_ROOT = PROJECT_ROOT / "tmps" / "datasets"
RUNS_ROOT = PROJECT_ROOT / "tmps" / "runs"
MODELS_ROOT = PROJECT_ROOT / "tmps" / "models"
RETURNS_ROOT = PROJECT_ROOT / "tmps" / "returns"
EVALS_ROOT = PROJECT_ROOT / "tmps" / "evals"
VIEW_ROOT = PROJECT_ROOT / "tmps" / "view"


def _task_id(dataset_id: str, agent_id: str | None = None) -> str:
    if agent_id is None:
        return f"{dataset_id}-v0"
    return f"{dataset_id}-{agent_id}-v0"


def data_path_train(dataset_id: str, agent_id: str) -> Path:
    return RUNS_ROOT / "train" / _task_id(dataset_id, agent_id) / "data" / "main_data.hdf5"


def data_path_test(dataset_id: str, agent_id: str) -> Path:
    return RUNS_ROOT / "test" / _task_id(dataset_id, agent_id) / "data" / "main_data.hdf5"


def model_path(dataset_id: str, agent_id: str, step: int) -> Path:
    return MODELS_ROOT / _task_id(dataset_id, agent_id) / str(step)


def eval_dir(dataset_id: str, agent_id: str) -> Path:
    return EVALS_ROOT / _task_id(dataset_id, agent_id)


def returns_path(dataset_id: str, agent_id: str | None = None) -> Path:
    return RETURNS_ROOT / f"{_task_id(dataset_id, agent_id)}.json"
