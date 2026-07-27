import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "source"
RESULT_ROOT = PROJECT_ROOT / "documents" / "result"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(RESULT_ROOT) not in sys.path:
    sys.path.insert(0, str(RESULT_ROOT))

from generate_result_quick_tables import EXPERIMENTS
from generate_result_quick_tables import STALE_SCORE_AGENTS
from generate_result_quick_tables import AgentSpec
from generate_result_quick_tables import Candidate
from generate_result_quick_tables import DatasetSpec
from generate_result_quick_tables import ExperimentSpec
from generate_result_quick_tables import dataset_returns
from generate_result_quick_tables import invalidated_candidate
from generate_result_quick_tables import selected_candidate
from generate_result_quick_tables import selected_legacy_candidate
from ice_offline.config.paths import boxplot_path
from ice_offline.run.agent_display import agent_display_name
from ice_offline.run.boxplot import boxplot_data


BOXPLOT_GROUP = "boxplot_result/base"


def base_spec() -> ExperimentSpec:
    return next(spec for spec in EXPERIMENTS if spec.output_name == "base")


def result_candidate(spec: ExperimentSpec, dataset: DatasetSpec, agent: AgentSpec) -> Candidate | None:
    candidate = selected_candidate(spec, dataset, agent)
    stale_selected = False
    if candidate is None:
        candidate = selected_legacy_candidate(spec, dataset, agent)
    elif invalidated_candidate(spec, dataset, agent.agent_id, candidate):
        legacy_candidate = selected_legacy_candidate(spec, dataset, agent)
        if legacy_candidate is not None:
            candidate = legacy_candidate
        else:
            stale_selected = agent.agent_id in STALE_SCORE_AGENTS

    if candidate is None:
        return None
    if invalidated_candidate(spec, dataset, agent.agent_id, candidate) and not (candidate.legacy or stale_selected):
        return None
    if not candidate.values:
        return None
    return candidate


def result_label(spec: ExperimentSpec, dataset: DatasetSpec, agent: AgentSpec, candidate: Candidate) -> str:
    return agent_display_name(agent.agent_id)


def save_result_boxplot(spec: ExperimentSpec, dataset: DatasetSpec) -> Path:
    labels = ["lower"]
    values: list[list[float] | None] = [list(dataset_returns(dataset.lower_id))]

    for agent in spec.agents:
        candidate = result_candidate(spec, dataset, agent)
        if candidate is None:
            continue
        labels.append(result_label(spec, dataset, agent, candidate))
        values.append(list(candidate.values))

    labels.append("upper")
    values.append(list(dataset_returns(dataset.upper_id)))

    path = boxplot_path(BOXPLOT_GROUP, f"{dataset.dataset_id}.png")
    boxplot_data(dataset.dataset_id, labels, values, path)
    return path


def generate(dataset_ids: list[str]) -> list[Path]:
    spec = base_spec()
    paths = []
    for dataset in spec.datasets:
        if dataset_ids and dataset.dataset_id not in dataset_ids:
            continue
        paths.append(save_result_boxplot(spec, dataset))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(args.dataset)
