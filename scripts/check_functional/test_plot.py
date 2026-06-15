import matplotlib

matplotlib.use("Agg")

from ice_offline.config.paths import _task_id
from ice_offline.config.paths import VIEW_ROOT
from ice_offline.config.paths import eval_returns_path
from ice_offline.config.paths import metric_path
from ice_offline.run.plot import plot


DATASET_ID = "hopper_simple"
AGENT_ID = "bc_deterministic"


def main() -> None:
    task_id = _task_id(DATASET_ID, AGENT_ID)
    metrics_path = metric_path(task_id)
    returns_path = eval_returns_path(task_id)
    output_path = VIEW_ROOT / "plot" / "check_functional" / "plot.png"

    plot(
        metric_paths=[metrics_path],
        eval_paths=[returns_path],
        output_path=output_path,
    )

    if not output_path.exists():
        raise RuntimeError(f"plot output missing: {output_path}")
    if output_path.stat().st_size <= 0:
        raise RuntimeError(f"plot output is empty: {output_path}")

    print(f"metrics_path={metrics_path}")
    print(f"eval_path={returns_path}")
    print(f"output_path={output_path}")
    print(f"output_size={output_path.stat().st_size}")


if __name__ == "__main__":
    main()
