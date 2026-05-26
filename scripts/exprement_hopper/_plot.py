from pathlib import Path

from ice_offline.plot.plotter import plot_csv


EVAL_ROOT = Path("tmps/eval")
PLOT_ROOT = Path("tmps/plot")


def plot(runner_key: str) -> None:
    eval_dir = EVAL_ROOT / f"{runner_key}-v0"
    csv_paths = [str(path) for path in sorted(eval_dir.glob("*.csv"))]
    if len(csv_paths) == 0:
        raise FileNotFoundError(f"no csv found: {eval_dir}")
    output_path = PLOT_ROOT / f"{runner_key}.png"
    plot_csv(csv_paths=csv_paths, plot_name=runner_key, show=True, output_path=str(output_path))
    print(f"saved: {output_path}")


if __name__ == "__main__":
    plot("hopper_expert_aspl")
    plot("hopper_expert_bc")
    plot("hopper_expert_cql")
    plot("hopper_expert_iql")
    plot("hopper_expert_scas")
    plot("hopper_medium_aspl")
    plot("hopper_medium_bc")
    plot("hopper_medium_cql")
    plot("hopper_medium_iql")
    plot("hopper_medium_scas")
    plot("hopper_simple_aspl")
    plot("hopper_simple_bc")
    plot("hopper_simple_cql")
    plot("hopper_simple_iql")
    plot("hopper_simple_scas")
