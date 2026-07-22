import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from ice_offline.config.paths import eval_path
from ice_offline.config.paths import experiment_task_id
from ice_offline.config.paths import returns_path
from ice_offline.dataset._lookup import make_dataset


RESULT_ROOT = PROJECT_ROOT / "documents" / "result"
AGENT_ROOT = PROJECT_ROOT / "source" / "ice_offline" / "agent"
TEST_COUNT = 20
TEST_INTERVAL = 1_000
TRAIN_FALLBACK_COUNT = 10
SCORE_DIGITS = 3


@dataclass(frozen=True)
class AgentSpec:
    agent_id: str
    model_step: int | None
    agent_step: int


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    lower_id: str
    upper_id: str
    train_dataset_id: str


@dataclass(frozen=True)
class ExperimentSpec:
    output_name: str
    test_experiment: str
    train_experiment: str
    train_min_experiment: str | None
    datasets: tuple[DatasetSpec, ...]
    agents: tuple[AgentSpec, ...]


@dataclass(frozen=True)
class EvalRows:
    path: Path
    mtime: float
    rows: tuple[tuple[int, tuple[float, ...]], ...]


@dataclass(frozen=True)
class Candidate:
    stage: str
    suffix: str
    path: Path
    mtime: float
    complete: bool
    reason: str
    values: tuple[float, ...]
    max_step: int | None
    expected_step: int


@dataclass(frozen=True)
class SelectedCell:
    experiment: str
    dataset_id: str
    train_dataset_id: str
    agent_id: str
    stage: str
    suffix: str
    cell: str
    score: float | None
    raw_mean: float | None
    lower_mean: float
    upper_mean: float
    eval_path: str
    eval_mtime: str
    agent_path: str
    agent_mtime: str
    agent_newer_than_eval: bool
    complete: bool
    reason: str
    max_step: int | None
    expected_step: int | None


WALKER_TABLES = (
    DatasetSpec("walker2d_d4rl_medium", "walker2d_random", "walker2d_d4rl_medium", "walker2d_d4rl_medium"),
    DatasetSpec("walker2d_d4rl_expert", "walker2d_random", "walker2d_d4rl_expert", "walker2d_d4rl_expert"),
    DatasetSpec("walker2d_d4rl_hybrid", "walker2d_random", "walker2d_d4rl_hybrid", "walker2d_d4rl_hybrid"),
    DatasetSpec("walker2d_replay_medium", "walker2d_random", "walker2d_d4rl_medium", "walker2d_replay_medium"),
    DatasetSpec("walker2d_replay_expert", "walker2d_random", "walker2d_d4rl_expert", "walker2d_replay_expert"),
)


BASE_TABLES = (
    DatasetSpec("hopper_d4rl_medium", "hopper_random", "hopper_d4rl_medium", "hopper_d4rl_medium"),
    DatasetSpec("hopper_d4rl_expert", "hopper_random", "hopper_d4rl_expert", "hopper_d4rl_expert"),
    DatasetSpec("hopper_d4rl_hybrid", "hopper_random", "hopper_d4rl_hybrid", "hopper_d4rl_hybrid"),
    DatasetSpec("hopper_replay_medium", "hopper_random", "hopper_d4rl_medium", "hopper_replay_medium"),
    DatasetSpec("hopper_replay_expert", "hopper_random", "hopper_d4rl_expert", "hopper_replay_expert"),
    *WALKER_TABLES,
    DatasetSpec("halfcheetah_d4rl_medium", "halfcheetah_random", "halfcheetah_d4rl_medium", "halfcheetah_d4rl_medium"),
    DatasetSpec("halfcheetah_d4rl_expert", "halfcheetah_random", "halfcheetah_d4rl_expert", "halfcheetah_d4rl_expert"),
    DatasetSpec("halfcheetah_d4rl_hybrid", "halfcheetah_random", "halfcheetah_d4rl_hybrid", "halfcheetah_d4rl_hybrid"),
    DatasetSpec("halfcheetah_replay_medium", "halfcheetah_random", "halfcheetah_d4rl_medium", "halfcheetah_replay_medium"),
    DatasetSpec("halfcheetah_replay_expert", "halfcheetah_random", "halfcheetah_d4rl_expert", "halfcheetah_replay_expert"),
)


REPRESENTATIVE_AGENTS = (
    AgentSpec("bc", None, 50_000),
    AgentSpec("td3bc_n", None, 100_000),
    AgentSpec("iql", None, 200_000),
    AgentSpec("cql", None, 500_000),
    AgentSpec("aspl_c", None, 500_000),
    AgentSpec("scas_n", 100_000, 500_000),
    AgentSpec("scaspl_n", 100_000, 500_000),
    AgentSpec("scc_n", 100_000, 500_000),
)


STABILITY_TD3BC_AGENTS = (
    AgentSpec("td3bc", None, 100_000),
    AgentSpec("td3bc_plus", None, 100_000),
    AgentSpec("td3bc_gp", None, 100_000),
    AgentSpec("td3bc_gp_plus", None, 100_000),
    AgentSpec("td3bc_n", None, 100_000),
    AgentSpec("td3bc_gpn", None, 100_000),
)


STABILITY_ASPL_AGENTS = (
    AgentSpec("aspl", None, 200_000),
    AgentSpec("aspl_gp", None, 500_000),
    AgentSpec("aspl_c", None, 500_000),
)


STABILITY_SCAS_AGENTS = (
    AgentSpec("scas", 100_000, 500_000),
    AgentSpec("scas_n", 100_000, 500_000),
    AgentSpec("scas_gp", 100_000, 500_000),
    AgentSpec("scas_gpn", 100_000, 500_000),
)


STABILITY_SCASPL_AGENTS = (
    AgentSpec("scaspl", 100_000, 500_000),
    AgentSpec("scaspl_n", 100_000, 500_000),
    AgentSpec("scaspl_gp", 100_000, 500_000),
    AgentSpec("scaspl_c", 100_000, 500_000),
    AgentSpec("scaspl_nc", 100_000, 500_000),
    AgentSpec("scaspl_gpc", 100_000, 500_000),
)


STABILITY_SCC_AGENTS = (
    AgentSpec("scc", 100_000, 500_000),
    AgentSpec("scc_n", 100_000, 500_000),
    AgentSpec("scc_gp", 100_000, 500_000),
)


def noise_tables(prefix: str, values: tuple[str, ...]) -> tuple[DatasetSpec, ...]:
    base_specs = (
        ("walker2d_d4rl_medium", "walker2d_d4rl_medium"),
        ("walker2d_d4rl_hybrid", "walker2d_d4rl_hybrid"),
        ("walker2d_replay_medium", "walker2d_replay_medium"),
    )
    return tuple(
        DatasetSpec(
            f"{prefix}_{value}@{dataset_id}",
            "walker2d_random",
            "walker2d_d4rl_medium",
            train_dataset_id,
        )
        for dataset_id, train_dataset_id in base_specs
        for value in values
    )


HYBRID_TABLES = (
    DatasetSpec("walker2d_random_expert_1", "walker2d_random", "walker2d_d4rl_expert", "walker2d_random_expert_1"),
    DatasetSpec("walker2d_random_expert_3", "walker2d_random", "walker2d_d4rl_expert", "walker2d_random_expert_3"),
    DatasetSpec("walker2d_random_expert_5", "walker2d_random", "walker2d_d4rl_expert", "walker2d_random_expert_5"),
    DatasetSpec("walker2d_random_expert_7", "walker2d_random", "walker2d_d4rl_expert", "walker2d_random_expert_7"),
    DatasetSpec("walker2d_random_expert_9", "walker2d_random", "walker2d_d4rl_expert", "walker2d_random_expert_9"),
)


EXPERIMENTS = (
    ExperimentSpec("stability_td3bc", "base", "base_train", "base", WALKER_TABLES, STABILITY_TD3BC_AGENTS),
    ExperimentSpec("stability_aspl", "base", "base_train", "base", WALKER_TABLES, STABILITY_ASPL_AGENTS),
    ExperimentSpec("stability_scas", "base", "base_train", "base", WALKER_TABLES, STABILITY_SCAS_AGENTS),
    ExperimentSpec("stability_scaspl", "base", "base_train", "base", WALKER_TABLES, STABILITY_SCASPL_AGENTS),
    ExperimentSpec("stability_scc", "base", "base_train", "base", WALKER_TABLES, STABILITY_SCC_AGENTS),
    ExperimentSpec("base", "base", "base_train", None, BASE_TABLES, REPRESENTATIVE_AGENTS),
    ExperimentSpec("noise_init", "noise_init", "base_train", "base", noise_tables("noise_init", ("5e-2", "1e-1", "5e-1", "1e0")), REPRESENTATIVE_AGENTS),
    ExperimentSpec("noise_action", "noise_action", "base_train", "base", noise_tables("noise_action", ("5e-2", "1e-1", "5e-1", "1e0")), REPRESENTATIVE_AGENTS),
    ExperimentSpec("noise_state", "noise_state", "base_train", "base", noise_tables("noise_state", ("5e-4", "1e-3", "5e-3", "1e-2")), REPRESENTATIVE_AGENTS),
    ExperimentSpec("hybrid_random", "experience_hybrid_random", "experience_hybrid_random_train", None, HYBRID_TABLES, REPRESENTATIVE_AGENTS),
)


DATASET_RETURN_CACHE: dict[str, tuple[float, ...]] = {}
EVAL_CACHE: dict[Path, EvalRows | None] = {}


def task_eval_path(experiment_id: str, agent_id: str, dataset_id: str) -> Path:
    return eval_path(experiment_task_id(experiment_id, agent_id, dataset_id))


def agent_source_path(agent_id: str) -> Path:
    return AGENT_ROOT / f"{agent_id}.py"


def format_time(timestamp: float | None) -> str:
    if timestamp is None:
        return ""
    return str(timestamp)


def read_eval_rows(path: Path) -> EvalRows | None:
    if path in EVAL_CACHE:
        return EVAL_CACHE[path]
    if not path.exists():
        EVAL_CACHE[path] = None
        return None

    task_id = str(path.relative_to(PROJECT_ROOT / "tmps" / "evals").parents[1])
    path_returns = returns_path(task_id)
    if path_returns.exists():
        rows = read_return_csv_rows(path_returns)
        result = EvalRows(path=path, mtime=path.stat().st_mtime, rows=rows)
        EVAL_CACHE[path] = result
        return result

    rows_by_step: dict[int, list[float]] = {}
    with h5py.File(path, "r") as file:
        keys = sorted(
            [key for key in file.keys() if key.startswith("episode_")],
            key=lambda key: tuple(int(part) for part in key.split("_")[1:]),
        )
        for key in keys:
            step = int(key.split("_")[1])
            rewards = file[key]["rewards"][()]
            rows_by_step.setdefault(step, []).append(float(rewards.sum()))

    rows = tuple(
        (step, tuple(values))
        for step, values in sorted(rows_by_step.items())
    )
    result = EvalRows(path=path, mtime=path.stat().st_mtime, rows=rows)
    EVAL_CACHE[path] = result
    return result


def read_return_csv_rows(path: Path) -> tuple[tuple[int, tuple[float, ...]], ...]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        next(reader)
        return tuple(
            (
                int(float(row[0])),
                tuple(
                    float(value)
                    for value in row[1:]
                    if value != "" and value != "nan"
                ),
            )
            for row in reader
        )


def flatten_rows(rows: tuple[tuple[int, tuple[float, ...]], ...]) -> tuple[float, ...]:
    return tuple(value for _, values in rows for value in values)


def latest_rows(rows: tuple[tuple[int, tuple[float, ...]], ...], count: int) -> tuple[tuple[int, tuple[float, ...]], ...]:
    return rows[-count:]


def max_step(rows: tuple[tuple[int, tuple[float, ...]], ...]) -> int | None:
    if not rows:
        return None
    return max(step for step, _ in rows)


def test_expected_end(agent_step: int) -> int:
    return agent_step + TEST_COUNT * TEST_INTERVAL


def test_candidate(experiment_id: str, agent_id: str, dataset_id: str, agent_step: int, stage: str, suffix: str) -> Candidate | None:
    path = task_eval_path(experiment_id, agent_id, dataset_id)
    eval_rows = read_eval_rows(path)
    if eval_rows is None:
        return None

    highest_step = max_step(eval_rows.rows)
    expected_step = test_expected_end(agent_step)
    complete = highest_step is not None and highest_step >= expected_step
    if complete:
        values = flatten_rows(eval_rows.rows) if stage == "test" else flatten_rows(latest_rows(eval_rows.rows, TRAIN_FALLBACK_COUNT))
        reason = "ok"
    elif highest_step is not None and highest_step >= agent_step:
        values = flatten_rows(latest_rows(eval_rows.rows, TRAIN_FALLBACK_COUNT))
        complete = stage != "test"
        reason = "partial_test_eval"
    else:
        values = ()
        reason = "missing_required_step"

    return Candidate(
        stage=stage,
        suffix=suffix,
        path=path,
        mtime=eval_rows.mtime,
        complete=complete,
        reason=reason,
        values=values,
        max_step=highest_step,
        expected_step=expected_step,
    )


def train_candidate(experiment_id: str, agent_id: str, dataset_id: str, agent_step: int) -> Candidate | None:
    path = task_eval_path(experiment_id, agent_id, dataset_id)
    eval_rows = read_eval_rows(path)
    if eval_rows is None:
        return None

    highest_step = max_step(eval_rows.rows)
    complete = highest_step is not None and highest_step >= agent_step
    values = flatten_rows(latest_rows(eval_rows.rows, TRAIN_FALLBACK_COUNT)) if complete else ()
    reason = "ok" if complete else "missing_required_step"
    return Candidate(
        stage="train",
        suffix="t",
        path=path,
        mtime=eval_rows.mtime,
        complete=complete,
        reason=reason,
        values=values,
        max_step=highest_step,
        expected_step=agent_step,
    )


def candidates_for(spec: ExperimentSpec, dataset: DatasetSpec, agent: AgentSpec) -> list[Candidate]:
    candidates: list[Candidate] = []
    test = test_candidate(spec.test_experiment, agent.agent_id, dataset.dataset_id, agent.agent_step, "test", "")
    if test is not None:
        candidates.append(test)

    if spec.train_min_experiment is not None:
        train_min_path = task_eval_path(spec.train_min_experiment, agent.agent_id, dataset.train_dataset_id)
        test_path = task_eval_path(spec.test_experiment, agent.agent_id, dataset.dataset_id)
        if train_min_path != test_path:
            train_min = test_candidate(spec.train_min_experiment, agent.agent_id, dataset.train_dataset_id, agent.agent_step, "train_min", "tm")
            if train_min is not None:
                candidates.append(train_min)

    train = train_candidate(spec.train_experiment, agent.agent_id, dataset.train_dataset_id, agent.agent_step)
    if train is not None:
        candidates.append(train)
    return candidates


def selected_candidate(spec: ExperimentSpec, dataset: DatasetSpec, agent: AgentSpec) -> Candidate | None:
    candidates = candidates_for(spec, dataset, agent)
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate.mtime)


def dataset_returns(dataset_id: str) -> tuple[float, ...]:
    if dataset_id in DATASET_RETURN_CACHE:
        return DATASET_RETURN_CACHE[dataset_id]

    dataset = make_dataset(dataset_id, device="cpu")
    values = tuple(float(episode.rewards.sum()) for episode in dataset.episodes)
    DATASET_RETURN_CACHE[dataset_id] = values
    return values


def mean(values: tuple[float, ...]) -> float:
    return sum(values) / len(values)


def scaled_score(value: float, lower: float, upper: float) -> float:
    return (value - lower) / (upper - lower) * 100.0


def formatted_cell(score: float | None, suffix: str, stale: bool) -> str:
    if score is None:
        return ""
    prefix = "!" if stale else ""
    stage_suffix = f"({suffix})" if suffix else ""
    return f"{prefix}{score:.{SCORE_DIGITS}f}{stage_suffix}"


def select_cell(spec: ExperimentSpec, dataset: DatasetSpec, agent: AgentSpec) -> SelectedCell:
    lower_mean = mean(dataset_returns(dataset.lower_id))
    upper_mean = mean(dataset_returns(dataset.upper_id))
    candidate = selected_candidate(spec, dataset, agent)
    source_path = agent_source_path(agent.agent_id)
    agent_mtime = source_path.stat().st_mtime if source_path.exists() else None
    agent_newer = False

    if candidate is None:
        return SelectedCell(
            experiment=spec.output_name,
            dataset_id=dataset.dataset_id,
            train_dataset_id=dataset.train_dataset_id,
            agent_id=agent.agent_id,
            stage="missing",
            suffix="",
            cell="",
            score=None,
            raw_mean=None,
            lower_mean=lower_mean,
            upper_mean=upper_mean,
            eval_path="",
            eval_mtime="",
            agent_path=str(source_path.relative_to(PROJECT_ROOT)) if source_path.exists() else "",
            agent_mtime=format_time(agent_mtime),
            agent_newer_than_eval=False,
            complete=False,
            reason="missing_eval",
            max_step=None,
            expected_step=None,
        )

    if agent_mtime is not None:
        agent_newer = agent_mtime > candidate.mtime

    raw_mean = mean(candidate.values) if candidate.complete and candidate.values else None
    score = scaled_score(raw_mean, lower_mean, upper_mean) if raw_mean is not None else None
    return SelectedCell(
        experiment=spec.output_name,
        dataset_id=dataset.dataset_id,
        train_dataset_id=dataset.train_dataset_id,
        agent_id=agent.agent_id,
        stage=candidate.stage,
        suffix=candidate.suffix if candidate.complete else "",
        cell=formatted_cell(score, candidate.suffix if candidate.complete else "", agent_newer and score is not None),
        score=score,
        raw_mean=raw_mean,
        lower_mean=lower_mean,
        upper_mean=upper_mean,
        eval_path=str(candidate.path.relative_to(PROJECT_ROOT)),
        eval_mtime=format_time(candidate.mtime),
        agent_path=str(source_path.relative_to(PROJECT_ROOT)) if source_path.exists() else "",
        agent_mtime=format_time(agent_mtime),
        agent_newer_than_eval=agent_newer,
        complete=candidate.complete,
        reason=candidate.reason,
        max_step=candidate.max_step,
        expected_step=candidate.expected_step,
    )


def write_experiment_table(spec: ExperimentSpec, cells: list[SelectedCell]) -> Path:
    path = RESULT_ROOT / f"{spec.output_name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    by_key = {
        (cell.dataset_id, cell.agent_id): cell
        for cell in cells
        if cell.experiment == spec.output_name
    }
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["task", *[agent.agent_id for agent in spec.agents]])
        for dataset in spec.datasets:
            writer.writerow([
                dataset.dataset_id,
                *[
                    by_key[(dataset.dataset_id, agent.agent_id)].cell
                    for agent in spec.agents
                ],
            ])
    return path


def write_version_table(cells: list[SelectedCell]) -> Path:
    path = RESULT_ROOT / "agent_dataset_versions.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "experiment",
            "dataset",
            "train_dataset",
            "agent",
            "stage",
            "cell",
            "score",
            "raw_mean",
            "lower_mean",
            "upper_mean",
            "complete",
            "reason",
            "max_step",
            "expected_step",
            "eval_path",
            "eval_mtime",
            "agent_path",
            "agent_mtime",
            "agent_newer_than_eval",
        ])
        for cell in cells:
            writer.writerow([
                cell.experiment,
                cell.dataset_id,
                cell.train_dataset_id,
                cell.agent_id,
                cell.stage,
                cell.cell,
                "" if cell.score is None else f"{cell.score:.6f}",
                "" if cell.raw_mean is None else f"{cell.raw_mean:.6f}",
                f"{cell.lower_mean:.6f}",
                f"{cell.upper_mean:.6f}",
                str(cell.complete),
                cell.reason,
                "" if cell.max_step is None else str(cell.max_step),
                "" if cell.expected_step is None else str(cell.expected_step),
                cell.eval_path,
                cell.eval_mtime,
                cell.agent_path,
                cell.agent_mtime,
                str(cell.agent_newer_than_eval),
            ])
    return path


def write_agent_file_table() -> Path:
    path = RESULT_ROOT / "agent_file_versions.csv"
    agent_ids = sorted({agent.agent_id for spec in EXPERIMENTS for agent in spec.agents})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["agent", "agent_path", "agent_mtime"])
        for agent_id in agent_ids:
            path_agent = agent_source_path(agent_id)
            writer.writerow([
                agent_id,
                str(path_agent.relative_to(PROJECT_ROOT)) if path_agent.exists() else "",
                format_time(path_agent.stat().st_mtime if path_agent.exists() else None),
            ])
    return path


def generate() -> tuple[list[Path], list[SelectedCell]]:
    cells: list[SelectedCell] = []
    output_paths: list[Path] = []
    for spec in EXPERIMENTS:
        for dataset in spec.datasets:
            for agent in spec.agents:
                cells.append(select_cell(spec, dataset, agent))
        output_paths.append(write_experiment_table(spec, cells))
    output_paths.append(write_version_table(cells))
    output_paths.append(write_agent_file_table())
    return output_paths, cells


if __name__ == "__main__":
    paths, _ = generate()
    for path in paths:
        print(f"saved: {path}")
