from dataclasses import dataclass
from typing import Any, Protocol
from pathlib import Path

from ice_offline.agent._spec import EnvSpec
from ice_offline.dataset._spec import BaseDataset
from ice_offline.pipeline.batch_loader import MinariLoader
from ice_offline.run.evaluator import OfflineEvalFn
from ice_offline.run.evaluator import OnlineEvalFn
from ice_offline.run.evaluator import Evaluator
from ice_offline.run.saver import RunnerSaver
from ice_offline.run.stopper import EarlyStopEvent
from ice_offline.run.stopper import RunnerStopper
from ice_offline.tools.paths import eval_root


BatchType = dict[str, Any]


class RunnerAgent(Protocol):
    device: str

    def configure(self, env_spec: EnvSpec) -> None: ...
    def act_best(self, observation: Any) -> Any: ...
    def update(self, batch: BatchType) -> None: ...
    def save(self, model_name: str | Path) -> Path: ...
    def load(self, model_name: str | Path) -> None: ...


@dataclass
class TorchBatchOfflineRunner:
    runner_id: str
    batch_size: int
    steps: int
    steps_begin: int = 0
    steps_begin_auto: bool = False
    save_interval: int = 0
    eval_interval: int = 0
    eval_batches: int = 8
    eval_episodes: int = 5
    eval_dir: str = str(eval_root())

    def train(
        self,
        dataset: BaseDataset,
        agent: RunnerAgent,
        eval_offline_fns: list[OfflineEvalFn],
        eval_online_fns: list[OnlineEvalFn],
        early_stop_events: list[EarlyStopEvent] | None = None,
    ) -> None:
        saver = RunnerSaver(self.runner_id, self.steps_begin, self.steps_begin_auto, self.save_interval)
        evaluator = Evaluator(
            self.runner_id,
            self.eval_dir,
            self.eval_batches,
            self.eval_episodes,
            self.eval_interval,
        )
        stopper = RunnerStopper(early_stop_events)

        steps_begin = saver.resolve_steps_begin()
        minari_loader = MinariLoader(dataset)
        env_spec = EnvSpec(
            observation_shape=minari_loader.obs_shape,
            observation_cardinality=dataset.observation_cardinality(
                minari_loader.obs_shape,
                minari_loader.minari_dataset,
            ),
            action_shape=minari_loader.act_shape,
            action_cardinality=dataset.action_cardinality(
                minari_loader.act_shape,
                minari_loader.minari_dataset,
            ),
        )
        agent.configure(env_spec)
        saver.load_if_needed(agent, steps_begin)

        if steps_begin > 0:
            precheck_rounds = stopper.precheck_evals()
            for _ in range(precheck_rounds):
                offline_metrics = evaluator.evaluate_offline(agent, minari_loader, eval_offline_fns, self.batch_size)
                online_metrics = evaluator.evaluate_online(agent, dataset, eval_online_fns)
                metrics = {**offline_metrics, **online_metrics}
                evaluator.append_eval_step(steps_begin, metrics)
                print(evaluator.summary_text(steps_begin, self.steps, metrics))
                if stopper.should_stop(metrics):
                    print(f"early_stop=True step={steps_begin}")
                    return

        for step in range(steps_begin + 1, self.steps + 1):
            batch: BatchType = minari_loader.sample_batch(self.batch_size)
            agent.update(batch)
            saver.save_if_needed(agent, step)

            if evaluator.should_eval(step):
                offline_metrics = evaluator.evaluate_offline(agent, minari_loader, eval_offline_fns, self.batch_size)
                online_metrics = evaluator.evaluate_online(agent, dataset, eval_online_fns)
                metrics = {**offline_metrics, **online_metrics}
                evaluator.append_eval_step(step, metrics)
                print(evaluator.summary_text(step, self.steps, metrics))
                if stopper.should_stop(metrics):
                    print(f"early_stop=True step={step}")
                    break
