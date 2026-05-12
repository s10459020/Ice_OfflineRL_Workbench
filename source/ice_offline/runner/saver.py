from pathlib import Path

from ice_offline.agent._spec import model_ref
from ice_offline.tools.paths import model_root


class RunnerSaver:
    def __init__(self, runner_id: str, steps_begin: int, steps_begin_auto: bool, save_interval: int) -> None:
        self.runner_id = runner_id
        self.steps_begin = steps_begin
        self.steps_begin_auto = steps_begin_auto
        self.save_interval = save_interval

    def _find_latest_checkpoint_step(self) -> int:
        runner_dir = Path(model_root()) / self.runner_id
        if not runner_dir.exists():
            return 0

        latest_step = 0
        for checkpoint_file in runner_dir.glob("*.pt"):
            step_text = checkpoint_file.stem
            if step_text.isdigit():
                step = int(step_text)
                if step > latest_step:
                    latest_step = step
        return latest_step

    def resolve_steps_begin(self) -> int:
        if not self.steps_begin_auto:
            return self.steps_begin
        auto_step = self._find_latest_checkpoint_step()
        if auto_step > 0:
            return auto_step
        return self.steps_begin

    def load_if_needed(self, agent, steps_begin: int) -> None:
        if steps_begin > 0:
            agent.load(model_ref(self.runner_id, steps_begin))

    def save_if_needed(self, agent, step: int) -> None:
        if self.save_interval > 0 and step % self.save_interval == 0:
            agent.save(model_ref(self.runner_id, step))
