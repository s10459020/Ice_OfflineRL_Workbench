from pathlib import Path

from ice_offline.plot.plotter import plot_csv


EVAL_ROOT = Path("tmps/eval")
PLOT_ROOT = Path("tmps/plot")


def plot(dataset_path: str) -> None:
    eval_dir = EVAL_ROOT / dataset_path
    csv_paths = [str(path) for path in sorted(eval_dir.glob("*.csv"))]
    if len(csv_paths) == 0:
        raise FileNotFoundError(f"no csv found: {eval_dir}")
    output_path = PLOT_ROOT / f"{dataset_path}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_csv(csv_paths=csv_paths, plot_name=dataset_path, show=False, output_path=str(output_path))
    print(f"saved: {output_path}")


if __name__ == "__main__":
    # plot("train/hopper_simple_bc-v0")
    # plot("test/hopper_simple_bc-v0")
    # plot("hopper_simple_aspl-v0")
    # plot("hopper_simple_bc-v0")
    # plot("hopper_simple_cql-v0")
    # plot("hopper_simple_iql-v0")
    # plot("hopper_simple_scas-v0")
    # plot("hopper_medium_aspl-v0")
    # plot("hopper_medium_bc-v0")
    # plot("hopper_medium_cql-v0")
    # plot("hopper_medium_iql-v0")
    plot("hopper_medium_scas-v0")
    # plot("hopper_expert_aspl-v0")
    # plot("hopper_expert_bc-v0")
    # plot("hopper_expert_cql-v0")
    # plot("hopper_expert_iql-v0")
    # plot("hopper_expert_scas-v0")
    # plot("hopper_medium_replay_d4rl_aspl-v0")
    # plot("hopper_medium_replay_d4rl_bc-v0")
    # plot("hopper_medium_replay_d4rl_cql-v0")
    # plot("hopper_medium_replay_d4rl_iql-v0")
    # plot("hopper_medium_replay_d4rl_scas-v0")
    # plot("hopper_medium_expert_d4rl_aspl-v0")
    # plot("hopper_medium_expert_d4rl_bc-v0")
    # plot("hopper_medium_expert_d4rl_cql-v0")
    # plot("hopper_medium_expert_d4rl_iql-v0")
    # plot("hopper_medium_expert_d4rl_scas-v0")
