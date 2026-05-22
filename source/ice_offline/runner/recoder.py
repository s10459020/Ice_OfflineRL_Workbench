from pathlib import Path

from ice_offline.tools.paths import eval_root


class Recoder:
    def __init__(self, runner_id: str, clear: bool = False) -> None:
        self.runner_id = runner_id
        self.runner_dir = Path(eval_root()) / runner_id
        self.runner_dir.mkdir(parents=True, exist_ok=True)
        if clear:
            for p in self.runner_dir.glob("*"):
                if p.is_file():
                    p.unlink()

    def save(self, eval_id: str, step: int, evals: list[float]) -> None:
        path = self.runner_dir / f"{eval_id}.csv"
        is_new = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as f:
            if is_new:
                header = "step," + ",".join(str(i) for i in range(1, len(evals) + 1))
                f.write(f"{header}\n")
            values = ",".join(str(float(v)) for v in evals)
            f.write(f"{step},{values}\n")
