import csv
from pathlib import Path

from view_result.returns import returns
from view_result.returns import returns_path
from view_result.skip import data_exists
from view_result.task import bottom_dataset_path
from view_result.task import dataset_group_name
from view_result.task import dataset_env_name
from view_result.task import table_output_path
from view_result.task import test_dataset_path
from view_result.task import top_dataset_path


def source_exists(dataset_path: str) -> bool:
    if not dataset_path:
        return False
    if returns_path(dataset_path).exists():
        return True
    return data_exists(dataset_path)


def return_values(dataset_path: str) -> list[float] | None:
    if not source_exists(dataset_path):
        return None
    return returns(dataset_path)


def mean_return(dataset_path: str) -> float | None:
    values = return_values(dataset_path)
    if values is None:
        return None
    return sum(values) / len(values)


def max_return(dataset_path: str) -> float | None:
    values = return_values(dataset_path)
    if values is None:
        return None
    return max(values)


def cell(value: float | None) -> str:
    if value is None:
        return ""
    return str(float(value))


def scale(value: float | None, bottom: float | None, top: float | None) -> float | None:
    if value is None or bottom is None or top is None:
        return None
    if top == bottom:
        return None
    return (value - bottom) / (top - bottom) * 100.0


def actual_row(dataset_cls, agent_list: list[str]) -> list[str]:
    group_name = dataset_group_name(dataset_cls)
    bottom_path = bottom_dataset_path(dataset_cls)
    top_path = top_dataset_path(dataset_cls)
    values = [mean_return(bottom_path)]
    values.extend(mean_return(test_dataset_path(dataset_cls, agent_id)) for agent_id in agent_list)
    values.append(mean_return(top_path))
    return [group_name, *[cell(value) for value in values]]


def normalized_row(
    dataset_cls,
    agent_list: list[str],
    *,
    use_max_top: bool,
) -> list[str]:
    group_name = dataset_group_name(dataset_cls)
    bottom_path = bottom_dataset_path(dataset_cls)
    top_path = top_dataset_path(dataset_cls)
    bottom = mean_return(bottom_path)
    top = max_return(top_path) if use_max_top else mean_return(top_path)
    values = [
        scale(mean_return(test_dataset_path(dataset_cls, agent_id)), bottom, top)
        for agent_id in agent_list
    ]
    return [group_name, *[cell(value) for value in values]]


def save_csv(dataset_cls, name: str, header: list[str], rows: list[list[str]]) -> Path:
    out_path = table_output_path(dataset_cls, name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"saved: {out_path}")
    return out_path


def actual_rows(dataset_cls_list: list, agent_list: list[str]) -> list[list[str]]:
    rows = []
    for dataset_cls in dataset_cls_list:
        rows.append(actual_row(dataset_cls, agent_list))
    return rows


def normalized_rows(dataset_cls_list: list, agent_list: list[str], *, use_max_top: bool) -> list[list[str]]:
    rows = []
    for dataset_cls in dataset_cls_list:
        rows.append(normalized_row(dataset_cls, agent_list, use_max_top=use_max_top))
    return rows


def dataset_classes_by_env(dataset_class_list: list) -> dict[str, list]:
    grouped = {}
    for dataset_cls in dataset_class_list:
        grouped.setdefault(dataset_env_name(dataset_cls), []).append(dataset_cls)
    return grouped


def main(dataset_class_list: list, agent_id_list: list[str]) -> None:
    agent_list = [agent_id for agent_id in agent_id_list if agent_id != "random"]
    for dataset_cls_list in dataset_classes_by_env(dataset_class_list).values():
        table_dataset = dataset_cls_list[0]
        save_csv(table_dataset, "actual_returns.csv", ["task", "random", *agent_list, "dataset"], actual_rows(dataset_cls_list, agent_list))
        save_csv(table_dataset, "mean_normalized_returns.csv", ["task", *agent_list], normalized_rows(dataset_cls_list, agent_list, use_max_top=False))
        save_csv(table_dataset, "max_normalized_returns.csv", ["task", *agent_list], normalized_rows(dataset_cls_list, agent_list, use_max_top=True))


if __name__ == "__main__":
    main([], [])
