from pathlib import Path

from ice_offline.plot import plot_eval_csv

# Hyperparameters
CSV_PATH = "tmps/eval/loss_reward/cql_discrete_onerooms8.csv"
OUTPUT_PATH = None
SHOW_MEAN_LINE = False
QUANTILE_INTERVAL = None
QUANTILE_LINES = None

def main() -> int:
    csv_path = Path(CSV_PATH)
    output_path = Path(OUTPUT_PATH) if OUTPUT_PATH else None
    saved_path = plot_eval_csv(
        csv_path=csv_path,
        output_path=output_path,
        show_mean_line=SHOW_MEAN_LINE,
        quantile_interval=QUANTILE_INTERVAL,
        quantile_lines=QUANTILE_LINES,
    )
    print(f"saved: {saved_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
