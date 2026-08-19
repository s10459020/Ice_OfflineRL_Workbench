from pathlib import Path


METRICS_ROOT = Path(__file__).resolve().parent.parent / "metrics"


def metrics_path(agent_id: str, dataset_id: str) -> Path:
    return METRICS_ROOT / f"{agent_id}-{dataset_id}.txt"


def clear_metric(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def save_metric(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(f"{value:.6f}\n")


def load_metric(path: Path) -> float:
    values = [float(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return sum(values) / len(values)
