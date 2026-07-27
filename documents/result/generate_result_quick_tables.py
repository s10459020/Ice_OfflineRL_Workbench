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
from ice_offline.config.paths import model_path
from ice_offline.config.paths import returns_path
from ice_offline.dataset._lookup import make_dataset


RESULT_ROOT = PROJECT_ROOT / "documents" / "result"
AGENT_ROOT = PROJECT_ROOT / "source" / "ice_offline" / "agent"
TEST_COUNT = 20
TEST_INTERVAL = 1_000
TRAIN_FALLBACK_COUNT = 10
SCORE_DIGITS = 2
TOP_SCORE_RATIO = 0.95
FORCE_UPDATED_AGENTS = {
    "scaspl",
    "scaspl_n",
    "scaspl_p",
    "scaspl_pn",
    "scaspl_pgp",
    "scaspl_gp",
    "scaspl_pc",
    "scaspl_pnc",
    "scaspl_c",
    "scaspl_nc",
    "scaspl_gpc",
}
STALE_SCORE_AGENTS = {
    "scas_gpn",
    "scaspl_p",
    "scaspl_pn",
    "scaspl_pgp",
    "scaspl_pc",
    "scaspl_pnc",
}
LEGACY_RESULT_SOURCE_AGENTS = {
    "scaspl_p": ("scaspl",),
    "scaspl_pn": ("scaspl_n",),
    "scaspl_pgp": ("scaspl_gp",),
    "scaspl_pc": ("scaspl_c",),
    "scaspl_pnc": ("scaspl_nc",),
}
INVALIDATED_AGENTS = {
    "aspl",
    "aspl_gp",
    "aspl_c",
    "scas",
    "scas_n",
    "scas_gp",
    "scas_gpn",
    "scaspl",
    "scaspl_n",
    "scaspl_p",
    "scaspl_pn",
    "scaspl_pgp",
    "scaspl_gp",
    "scaspl_pc",
    "scaspl_pnc",
    "scaspl_c",
    "scaspl_nc",
    "scaspl_gpc",
}

AGENT_SOURCE_DEPENDENCIES = {
    "aspl_gp": ("aspl.py", "aspl_gp.py"),
    "aspl_c": ("aspl.py", "aspl_c.py"),
    "scas_n": ("scas.py", "scas_n.py"),
    "scas_gp": ("scas.py", "scas_gp.py"),
    "scas_gpn": ("scas.py", "scas_n.py", "scas_gp.py", "scas_gpn.py"),
    "scaspl": ("scaspl.py", "scas.py", "aspl.py"),
    "scaspl_n": ("scaspl.py", "scaspl_n.py", "scas.py", "aspl.py"),
    "scaspl_p": ("scaspl.py", "scaspl_p.py", "scas.py", "aspl.py"),
    "scaspl_pn": ("scaspl.py", "scaspl_n.py", "scaspl_pn.py", "scas.py", "aspl.py"),
    "scaspl_pgp": ("scaspl.py", "scaspl_p.py", "scaspl_pgp.py", "scaspl_gp.py", "scas.py", "aspl.py"),
    "scaspl_gp": ("scaspl.py", "scaspl_gp.py", "scas.py", "aspl.py"),
    "scaspl_pc": ("scaspl.py", "scaspl_p.py", "scaspl_pc.py", "aspl_c.py", "scas.py", "aspl.py"),
    "scaspl_pnc": ("scaspl.py", "scaspl_n.py", "scaspl_pn.py", "scaspl_pnc.py", "aspl_c.py", "scas.py", "aspl.py"),
    "scaspl_c": ("scaspl.py", "scaspl_c.py", "aspl_c.py", "scas.py", "aspl.py"),
    "scaspl_nc": ("scaspl.py", "scaspl_n.py", "scaspl_nc.py", "scas.py", "aspl.py"),
    "scaspl_gpc": ("scaspl.py", "scaspl_gp.py", "scaspl_gpc.py", "aspl_c.py", "scas.py", "aspl.py"),
}

AGENT_RESULT_DEPENDENCIES = {
    "aspl_gp": ("aspl",),
    "aspl_c": ("aspl",),
}


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
    fallback_scores: bool = True


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
    source_agent_id: str
    legacy: bool = False


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


DATASET_KIND_ORDER = (
    ("d4rl_medium", "d4rl_medium"),
    ("d4rl_expert", "d4rl_expert"),
    ("d4rl_hybrid", "d4rl_hybrid"),
    ("replay_medium", "d4rl_medium"),
    ("replay_expert", "d4rl_expert"),
)


def standard_tables(environment: str, dataset_kind_order: tuple[tuple[str, str], ...] = DATASET_KIND_ORDER) -> tuple[DatasetSpec, ...]:
    return tuple(
        DatasetSpec(
            f"{environment}_{dataset_kind}",
            f"{environment}_random",
            f"{environment}_{upper_kind}",
            f"{environment}_{dataset_kind}",
        )
        for dataset_kind, upper_kind in dataset_kind_order
    )


WALKER_TABLES = standard_tables("walker2d")


BASE_TABLES = (
    *standard_tables("hopper"),
    *WALKER_TABLES,
    *standard_tables("halfcheetah"),
)


REPRESENTATIVE_AGENTS = (
    AgentSpec("bc", None, 50_000),
    AgentSpec("td3bc_n", None, 100_000),
    AgentSpec("iql", None, 200_000),
    AgentSpec("cql", None, 500_000),
    AgentSpec("aspl_c", None, 500_000),
    AgentSpec("scas_n", 100_000, 500_000),
    AgentSpec("scaspl_pn", 500_000, 500_000),
    AgentSpec("scc_n", 100_000, 500_000),
)


STABILITY_TD3BC_AGENTS = (
    AgentSpec("td3bc_n", None, 100_000),
    AgentSpec("td3bc_plus", None, 100_000),
    AgentSpec("td3bc", None, 100_000),
    AgentSpec("td3bc_gp_plus", None, 100_000),
    AgentSpec("td3bc_gp", None, 100_000),
    AgentSpec("td3bc_gpn", None, 100_000),
)


STABILITY_ASPL_AGENTS = (
    AgentSpec("aspl", None, 500_000),
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
    AgentSpec("scaspl_p", 100_000, 500_000),
    AgentSpec("scaspl_pn", 500_000, 500_000),
    AgentSpec("scaspl_pgp", 100_000, 500_000),
    AgentSpec("scaspl_pc", 100_000, 500_000),
    AgentSpec("scaspl_pnc", 500_000, 500_000),
)


STABILITY_SCC_AGENTS = (
    AgentSpec("scc", 100_000, 500_000),
    AgentSpec("scc_n", 100_000, 500_000),
    AgentSpec("scc_gp", 100_000, 500_000),
    AgentSpec("scc_gpn", 100_000, 500_000),
)


def noise_tables(prefix: str, values: tuple[str, ...]) -> tuple[DatasetSpec, ...]:
    base_specs = (
        ("walker2d_d4rl_medium", "walker2d_d4rl_medium", "walker2d_d4rl_medium"),
        ("walker2d_d4rl_hybrid", "walker2d_d4rl_hybrid", "walker2d_d4rl_medium"),
        ("walker2d_replay_medium", "walker2d_replay_medium", "walker2d_d4rl_medium"),
    )
    return tuple(
        DatasetSpec(
            f"{prefix}_{value}@{dataset_id}",
            "walker2d_random",
            upper_id,
            train_dataset_id,
        )
        for dataset_id, train_dataset_id, upper_id in base_specs
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
    ExperimentSpec("noise_init", "noise_init", "base_train", "base", noise_tables("noise_init", ("5e-2", "1e-1", "5e-1", "1e0")), REPRESENTATIVE_AGENTS, False),
    ExperimentSpec("noise_action", "noise_action", "base_train", "base", noise_tables("noise_action", ("5e-2", "1e-1", "5e-1", "1e0")), REPRESENTATIVE_AGENTS, False),
    ExperimentSpec("noise_state", "noise_state", "base_train", "base", noise_tables("noise_state", ("5e-4", "1e-3", "5e-3", "1e-2")), REPRESENTATIVE_AGENTS, False),
    ExperimentSpec("hybrid_random", "experience_hybrid_random", "experience_hybrid_random_train", None, HYBRID_TABLES, REPRESENTATIVE_AGENTS),
)


DATASET_RETURN_CACHE: dict[str, tuple[float, ...]] = {}
EVAL_CACHE: dict[Path, EvalRows | None] = {}


def task_eval_path(experiment_id: str, agent_id: str, dataset_id: str) -> Path:
    return eval_path(experiment_task_id(experiment_id, agent_id, dataset_id))


def agent_source_paths(agent_id: str) -> tuple[Path, ...]:
    source_names = AGENT_SOURCE_DEPENDENCIES.get(agent_id, (f"{agent_id}.py",))
    return tuple(AGENT_ROOT / source_name for source_name in source_names)


def agent_source_path(agent_id: str) -> Path:
    return agent_source_paths(agent_id)[0]


def agent_source_mtime(agent_id: str) -> float | None:
    source_paths = agent_source_paths(agent_id)
    existing_paths = [path for path in source_paths if path.exists()]
    if not existing_paths:
        return None
    return max(path.stat().st_mtime for path in existing_paths)


def format_agent_source_paths(agent_id: str) -> str:
    return ";".join(
        str(path.relative_to(PROJECT_ROOT))
        for path in agent_source_paths(agent_id)
        if path.exists()
    )


def invalidated_path(agent_id: str, path: Path) -> bool:
    if agent_id not in INVALIDATED_AGENTS:
        return False
    source_mtime = agent_source_mtime(agent_id)
    if source_mtime is None or not path.exists():
        return False
    return path.stat().st_mtime < source_mtime


def candidate_model_path(spec: ExperimentSpec, dataset: DatasetSpec, agent_id: str, candidate: Candidate) -> Path | None:
    if candidate.max_step is None:
        return None
    train_id = experiment_task_id(spec.train_experiment, candidate.source_agent_id, dataset.train_dataset_id)
    path = model_path(train_id, candidate.max_step)
    if not path.exists():
        return None
    return path


def candidate_version_mtime(spec: ExperimentSpec, dataset: DatasetSpec, agent_id: str, candidate: Candidate) -> float:
    path = candidate_model_path(spec, dataset, agent_id, candidate)
    if path is None:
        return candidate.mtime
    return path.stat().st_mtime


def result_dependency_mtime(spec: ExperimentSpec, dataset: DatasetSpec, agent_id: str) -> float | None:
    dependency_mtimes = []
    for dependency_id in AGENT_RESULT_DEPENDENCIES.get(agent_id, ()):
        dependency_path = task_eval_path(spec.train_experiment, dependency_id, dataset.train_dataset_id)
        dependency_eval = read_eval_rows(dependency_path)
        if dependency_eval is not None:
            step = max_step(dependency_eval.rows)
            train_id = experiment_task_id(spec.train_experiment, dependency_id, dataset.train_dataset_id)
            path = model_path(train_id, step) if step is not None else None
            if path is not None and path.exists():
                dependency_mtimes.append(path.stat().st_mtime)
            else:
                dependency_mtimes.append(dependency_eval.mtime)
    if not dependency_mtimes:
        return None
    return max(dependency_mtimes)


def invalidated_by_result_dependency(spec: ExperimentSpec, dataset: DatasetSpec, agent_id: str, candidate: Candidate) -> bool:
    dependency_mtime = result_dependency_mtime(spec, dataset, agent_id)
    if dependency_mtime is None:
        return False
    return candidate_version_mtime(spec, dataset, agent_id, candidate) < dependency_mtime


def invalidated_candidate(spec: ExperimentSpec, dataset: DatasetSpec, agent_id: str, candidate: Candidate) -> bool:
    source_mtime = agent_source_mtime(agent_id)
    if agent_id in INVALIDATED_AGENTS and source_mtime is not None:
        if candidate_version_mtime(spec, dataset, agent_id, candidate) < source_mtime:
            return True
    return (
        invalidated_path(agent_id, candidate.path)
        or invalidated_by_result_dependency(spec, dataset, agent_id, candidate)
    )


def invalidation_reason(spec: ExperimentSpec, dataset: DatasetSpec, agent_id: str, candidate: Candidate, default_reason: str) -> str:
    source_mtime = agent_source_mtime(agent_id)
    if agent_id in INVALIDATED_AGENTS and source_mtime is not None:
        if candidate_version_mtime(spec, dataset, agent_id, candidate) < source_mtime:
            return "stale_agent"
    if invalidated_by_result_dependency(spec, dataset, agent_id, candidate):
        return "stale_dependency"
    path = candidate.path
    if invalidated_path(agent_id, path):
        return "stale_agent"
    return default_reason


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


def test_steps(agent_step: int) -> tuple[int, ...]:
    return tuple(
        agent_step + TEST_INTERVAL * index
        for index in range(TEST_COUNT + 1)
    )


def test_candidate(
    experiment_id: str,
    agent_id: str,
    dataset_id: str,
    agent_step: int,
    stage: str,
    suffix: str,
    legacy: bool = False,
) -> Candidate | None:
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
        values = flatten_rows(latest_rows(eval_rows.rows, TRAIN_FALLBACK_COUNT)) if highest_step is not None else ()
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
        source_agent_id=agent_id,
        legacy=legacy,
    )


def train_candidate(
    experiment_id: str,
    agent_id: str,
    dataset_id: str,
    agent_step: int,
    legacy: bool = False,
) -> Candidate | None:
    path = task_eval_path(experiment_id, agent_id, dataset_id)
    eval_rows = read_eval_rows(path)
    if eval_rows is None:
        return None

    highest_step = max_step(eval_rows.rows)
    complete = highest_step is not None and highest_step >= agent_step
    values = flatten_rows(latest_rows(eval_rows.rows, TRAIN_FALLBACK_COUNT)) if highest_step is not None else ()
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
        source_agent_id=agent_id,
        legacy=legacy,
    )


def candidates_for(spec: ExperimentSpec, dataset: DatasetSpec, agent: AgentSpec) -> list[Candidate]:
    candidates: list[Candidate] = []
    test = test_candidate(spec.test_experiment, agent.agent_id, dataset.dataset_id, agent.agent_step, "test", "")
    if test is not None:
        candidates.append(test)

    if not spec.fallback_scores:
        return candidates

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


def legacy_candidates_for(spec: ExperimentSpec, dataset: DatasetSpec, agent: AgentSpec) -> list[Candidate]:
    candidates: list[Candidate] = []
    for legacy_agent_id in LEGACY_RESULT_SOURCE_AGENTS.get(agent.agent_id, ()):
        test = test_candidate(spec.test_experiment, legacy_agent_id, dataset.dataset_id, agent.agent_step, "test", "", True)
        if test is not None:
            candidates.append(test)

        if not spec.fallback_scores:
            continue

        if spec.train_min_experiment is not None:
            train_min_path = task_eval_path(spec.train_min_experiment, legacy_agent_id, dataset.train_dataset_id)
            test_path = task_eval_path(spec.test_experiment, legacy_agent_id, dataset.dataset_id)
            if train_min_path != test_path:
                train_min = test_candidate(spec.train_min_experiment, legacy_agent_id, dataset.train_dataset_id, agent.agent_step, "train_min", "tm", True)
                if train_min is not None:
                    candidates.append(train_min)

        train = train_candidate(spec.train_experiment, legacy_agent_id, dataset.train_dataset_id, agent.agent_step, True)
        if train is not None:
            candidates.append(train)
    return candidates


def selected_candidate(spec: ExperimentSpec, dataset: DatasetSpec, agent: AgentSpec) -> Candidate | None:
    candidates = candidates_for(spec, dataset, agent)
    if not candidates:
        return None
    valid_candidates = [
        candidate
        for candidate in candidates
        if not invalidated_candidate(spec, dataset, agent.agent_id, candidate)
    ]
    if valid_candidates:
        return max(valid_candidates, key=lambda candidate: candidate.mtime)
    return max(candidates, key=lambda candidate: candidate.mtime)


def selected_legacy_candidate(spec: ExperimentSpec, dataset: DatasetSpec, agent: AgentSpec) -> Candidate | None:
    candidates = legacy_candidates_for(spec, dataset, agent)
    if not candidates:
        return None
    complete_candidates = [candidate for candidate in candidates if candidate.complete]
    if complete_candidates:
        return max(complete_candidates, key=lambda candidate: candidate.mtime)
    valued_candidates = [candidate for candidate in candidates if candidate.values]
    if valued_candidates:
        return max(valued_candidates, key=lambda candidate: candidate.mtime)
    return max(candidates, key=lambda candidate: candidate.mtime)


def model_status(spec: ExperimentSpec, dataset: DatasetSpec, agent: AgentSpec) -> tuple[str, str, Path | None, int | None, int]:
    if agent.model_step is not None:
        model_id = experiment_task_id(spec.train_experiment, "scas_model", dataset.train_dataset_id)
        if not model_path(model_id, agent.model_step).exists():
            return "missing_model", "", None, None, agent.model_step

    task_id = experiment_task_id(spec.train_experiment, agent.agent_id, dataset.train_dataset_id)
    steps = test_steps(agent.agent_step)
    existing_steps = [
        step
        for step in steps
        if model_path(task_id, step).exists()
    ]
    if len(existing_steps) == len(steps):
        return "train_min", "tm", model_path(task_id, steps[-1]), existing_steps[-1], steps[-1]
    if agent.agent_step in existing_steps:
        return "train", "t", model_path(task_id, agent.agent_step), agent.agent_step, steps[-1]
    if existing_steps:
        return "partial_model", "L", model_path(task_id, existing_steps[-1]), existing_steps[-1], steps[-1]
    return "missing", "", None, None, steps[-1]


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


def formatted_marker(suffix: str, stale: bool) -> str:
    if not suffix:
        return ""
    prefix = "!" if stale else ""
    return f"{prefix}({suffix})"


def top_score_threshold(best_score: float) -> float:
    if best_score > 0.0:
        return best_score * TOP_SCORE_RATIO
    return best_score - abs(best_score) * (1.0 - TOP_SCORE_RATIO)


def marked_cell(cell: SelectedCell, threshold: float | None) -> str:
    if threshold is None or cell.score is None:
        return cell.cell
    if cell.score >= threshold:
        return f"{cell.cell}*"
    return cell.cell


def marked_cell_map(cells: list[SelectedCell]) -> dict[tuple[str, str, str], str]:
    groups: dict[tuple[str, str], list[SelectedCell]] = {}
    for cell in cells:
        groups.setdefault((cell.experiment, cell.dataset_id), []).append(cell)

    result: dict[tuple[str, str, str], str] = {}
    for row_cells in groups.values():
        scores = [cell.score for cell in row_cells if cell.score is not None]
        threshold = top_score_threshold(max(scores)) if scores else None
        for cell in row_cells:
            result[(cell.experiment, cell.dataset_id, cell.agent_id)] = marked_cell(cell, threshold)
    return result


def select_cell(spec: ExperimentSpec, dataset: DatasetSpec, agent: AgentSpec) -> SelectedCell:
    lower_mean = mean(dataset_returns(dataset.lower_id))
    upper_mean = mean(dataset_returns(dataset.upper_id))
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
    source_path_text = format_agent_source_paths(agent.agent_id)
    agent_mtime = agent_source_mtime(agent.agent_id)
    agent_newer = False

    if candidate is None:
        if not spec.fallback_scores:
            stage, suffix, status_path, max_status_step, expected_status_step = model_status(spec, dataset, agent)
            status_mtime = status_path.stat().st_mtime if status_path is not None and status_path.exists() else None
            force_updated = agent.agent_id in FORCE_UPDATED_AGENTS
            if agent_mtime is not None and status_mtime is not None:
                agent_newer = agent_mtime > status_mtime
            agent_newer = agent_newer or force_updated
            reason = "missing_noise_test"
            dependency_mtime = result_dependency_mtime(spec, dataset, agent.agent_id)
            dependency_stale = (
                dependency_mtime is not None
                and status_path is not None
                and status_path.exists()
                and status_path.stat().st_mtime < dependency_mtime
            )
            if status_path is not None and (
                invalidated_path(agent.agent_id, status_path)
                or dependency_stale
            ):
                stage = "stale"
                suffix = ""
                reason = "stale_dependency" if dependency_stale else "stale_agent"
            return SelectedCell(
                experiment=spec.output_name,
                dataset_id=dataset.dataset_id,
                train_dataset_id=dataset.train_dataset_id,
                agent_id=agent.agent_id,
                stage=stage,
                suffix=suffix,
                cell=formatted_marker(suffix, agent_newer),
                score=None,
                raw_mean=None,
                lower_mean=lower_mean,
                upper_mean=upper_mean,
                eval_path=str(status_path.relative_to(PROJECT_ROOT)) if status_path is not None and status_path.exists() else "",
                eval_mtime=format_time(status_mtime),
                agent_path=source_path_text,
                agent_mtime=format_time(agent_mtime),
                agent_newer_than_eval=agent_newer,
                complete=False,
                reason=reason,
                max_step=max_status_step,
                expected_step=expected_status_step,
            )
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
            agent_path=source_path_text,
            agent_mtime=format_time(agent_mtime),
            agent_newer_than_eval=False,
            complete=False,
            reason="missing_eval",
            max_step=None,
            expected_step=None,
        )

    if agent_mtime is not None:
        agent_newer = agent_mtime > candidate_version_mtime(spec, dataset, agent.agent_id, candidate)
    agent_newer = agent_newer or agent.agent_id in FORCE_UPDATED_AGENTS or candidate.legacy or stale_selected

    if invalidated_candidate(spec, dataset, agent.agent_id, candidate) and not (candidate.legacy or stale_selected):
        reason = invalidation_reason(spec, dataset, agent.agent_id, candidate, candidate.reason)
        return SelectedCell(
            experiment=spec.output_name,
            dataset_id=dataset.dataset_id,
            train_dataset_id=dataset.train_dataset_id,
            agent_id=agent.agent_id,
            stage=candidate.stage,
            suffix="",
            cell="",
            score=None,
            raw_mean=None,
            lower_mean=lower_mean,
            upper_mean=upper_mean,
            eval_path=str(candidate.path.relative_to(PROJECT_ROOT)),
            eval_mtime=format_time(candidate.mtime),
            agent_path=source_path_text,
            agent_mtime=format_time(agent_mtime),
            agent_newer_than_eval=True,
            complete=False,
            reason=reason,
            max_step=candidate.max_step,
            expected_step=candidate.expected_step,
        )

    raw_mean = mean(candidate.values) if candidate.values else None
    score = scaled_score(raw_mean, lower_mean, upper_mean) if raw_mean is not None else None
    suffix = candidate.suffix if candidate.complete else "L" if candidate.values else ""
    reason = "legacy_source" if candidate.legacy else invalidation_reason(spec, dataset, agent.agent_id, candidate, candidate.reason)
    return SelectedCell(
        experiment=spec.output_name,
        dataset_id=dataset.dataset_id,
        train_dataset_id=dataset.train_dataset_id,
        agent_id=agent.agent_id,
        stage=candidate.stage,
        suffix=suffix,
        cell=formatted_cell(score, suffix, agent_newer and score is not None),
        score=score,
        raw_mean=raw_mean,
        lower_mean=lower_mean,
        upper_mean=upper_mean,
        eval_path=str(candidate.path.relative_to(PROJECT_ROOT)),
        eval_mtime=format_time(candidate.mtime),
        agent_path=source_path_text,
        agent_mtime=format_time(agent_mtime),
        agent_newer_than_eval=agent_newer,
        complete=candidate.complete,
        reason=reason,
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
            row_cells = [
                by_key[(dataset.dataset_id, agent.agent_id)]
                for agent in spec.agents
            ]
            scores = [cell.score for cell in row_cells if cell.score is not None]
            threshold = top_score_threshold(max(scores)) if scores else None
            writer.writerow([
                dataset.dataset_id,
                *[
                    marked_cell(cell, threshold)
                    for cell in row_cells
                ],
            ])
    return path


def write_version_table(cells: list[SelectedCell]) -> Path:
    path = RESULT_ROOT / "agent_dataset_versions.csv"
    by_marked_cell = marked_cell_map(cells)
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
                by_marked_cell[(cell.experiment, cell.dataset_id, cell.agent_id)],
                "" if cell.score is None else f"{cell.score:.{SCORE_DIGITS}f}",
                "" if cell.raw_mean is None else f"{cell.raw_mean:.{SCORE_DIGITS}f}",
                f"{cell.lower_mean:.{SCORE_DIGITS}f}",
                f"{cell.upper_mean:.{SCORE_DIGITS}f}",
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
            writer.writerow([
                agent_id,
                format_agent_source_paths(agent_id),
                format_time(agent_source_mtime(agent_id)),
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
