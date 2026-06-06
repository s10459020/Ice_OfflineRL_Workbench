from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASETS_ROOT = PROJECT_ROOT / "tmps" / "datasets"
RUNS_ROOT = PROJECT_ROOT / "tmps" / "runs"
MODELS_ROOT = PROJECT_ROOT / "tmps" / "models"
RETURNS_ROOT = PROJECT_ROOT / "tmps" / "returns"
EVALS_ROOT = PROJECT_ROOT / "tmps" / "evals"
