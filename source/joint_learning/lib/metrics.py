from pathlib import Path


METRICS_ROOT = Path(__file__).resolve().parent.parent / "metrics"


def metrics_path(agent_id: str, dataset_id: str) -> Path:
    return METRICS_ROOT / f"{agent_id}-{dataset_id}.txt"


def write_metrics(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    content = "".join(f"{value:.6f}\n" for value in values)
    temporary_path.write_text(content, encoding="utf-8", newline="\n")
    temporary_path.replace(path)


def load_metric(path: Path) -> float:
    values = [float(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return sum(values) / len(values)
