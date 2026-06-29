import csv

from ice_offline.config.paths import metric_path


class MetricRecorder:
    def __init__(self, task_id: str, keys: list[str] | None = None, resume: bool = False) -> None:
        self.path = metric_path(task_id)
        self.keys = keys
        self.last = {}
        if not resume:
            self.new()

    def new(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", *(self.keys or [])])

    def flush(self, step, metrics: dict[str, float | None]) -> None:
        keys = self.keys or list(metrics.keys())
        values = [metrics.get(key) for key in keys]
        
        with self.path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([step, *values])

        self.last = dict(zip(keys, values))
