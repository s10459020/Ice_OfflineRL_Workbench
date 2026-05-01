from pathlib import Path

from ice_offline.plot import plot_eval_csv


CSV_DIR = Path("tmps/eval")
OUTPUT_DIR = Path("tmps/eval/plots")
ONEROOMS8_REWARD_FILE = Path("tmps/eval/eval_reward_onerooms8.txt")
INVERTEDPENDULUM_REWARD_FILE = Path("tmps/eval/eval_reward_invertedpendulum.txt")

SHOW_MEAN_LINE = False
QUANTILE_INTERVAL = None
QUANTILE_LINES = None


def load_reward_refs(path: Path) -> dict[str, float]:
    refs: dict[str, float] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key == "reward_mean":
            refs[key] = float(value)
    return refs


def env_reward_refs(csv_name: str, onerooms8_refs: dict[str, float], inverted_refs: dict[str, float]) -> dict[str, float]:
    if "onerooms8" in csv_name:
        return onerooms8_refs
    if "invertedpendulum" in csv_name:
        return inverted_refs
    return {}


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    onerooms8_refs = load_reward_refs(ONEROOMS8_REWARD_FILE)
    inverted_refs = load_reward_refs(INVERTEDPENDULUM_REWARD_FILE)

    csv_paths = sorted(CSV_DIR.glob("*.csv"))
    for csv_path in csv_paths:
        refs = env_reward_refs(csv_path.name, onerooms8_refs=onerooms8_refs, inverted_refs=inverted_refs)
        output_path = OUTPUT_DIR / f"{csv_path.stem}.png"
        saved = plot_eval_csv(
            csv_path=csv_path,
            output_path=output_path,
            show_mean_line=SHOW_MEAN_LINE,
            quantile_interval=QUANTILE_INTERVAL,
            quantile_lines=QUANTILE_LINES,
            reward_reference_lines=refs,
        )
        print(f"saved: {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
