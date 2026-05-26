from typing import Protocol


class EarlyStopEvent(Protocol):
    def should_stop(self, metrics: dict[str, list[float]]) -> bool: ...


class RunnerStopper:
    def __init__(self, early_stop_events: list[EarlyStopEvent] | None) -> None:
        self.early_stop_events = early_stop_events

    def should_stop(self, metrics: dict[str, list[float]]) -> bool:
        if self.early_stop_events is None:
            return False
        for early_stop_event in self.early_stop_events:
            if early_stop_event.should_stop(metrics):
                return True
        return False

    def precheck_evals(self) -> int:
        if self.early_stop_events is None:
            return 0
        rounds = 0
        for early_stop_event in self.early_stop_events:
            patience = getattr(early_stop_event, "patience", 0)
            if patience > rounds:
                rounds = int(patience)
        return rounds
