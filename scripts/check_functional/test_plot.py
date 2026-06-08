import matplotlib

matplotlib.use("Agg")

from ice_offline.config.paths import VIEW_ROOT
from ice_offline.config.paths import eval_path
from ice_offline.config.paths import metric_path
from ice_offline.tools.plotter import plot


DATASET_ID = "hopper_simple"
AGENT_ID = "bc_deterministic"


def main() -> None:
    metrics_path = metric_path(DATASET_ID, AGENT_ID)
    returns_path = eval_path(DATASET_ID, AGENT_ID)
    output_path = VIEW_ROOT / "plot" / "check_functional" / "plot.png"

    plot(
        metrics_path=str(metrics_path),
        eval_paths=str(returns_path),
        output_path=str(output_path),
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
