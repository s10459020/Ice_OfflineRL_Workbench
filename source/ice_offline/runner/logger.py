class Logger:
    def __init__(self, interval: int = 0) -> None:
        self.interval = interval

    def _should_log(self, step: int) -> bool:
        return self.interval > 0 and step % self.interval == 0

    def print(self, event: str, step: int, total: int, **data: float) -> None:
        if not self._should_log(step):
            return
        print(event, f"step={step}/{total}", *[f"{k}={v:.6f}" for k, v in data.items()])
