import csv
from pathlib import Path


def table_path(experiment_id: str) -> Path:
    return Path(__file__).resolve().parent.parent / "table" / f"{experiment_id}.csv"


def write_table(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)

