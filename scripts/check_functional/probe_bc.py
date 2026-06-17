import numpy as np
import torch
from torch.distributions import Normal

from ice_offline.agent.bc_stochastic import BCStochasticAgent
from ice_offline.config.paths import _task_id
from ice_offline.dataset._lookup import make_dataset
from ice_offline.dataset.base import Dataset
from ice_offline.run.probe import probe_replay
from ice_offline.store.probe.action_axis_probe import ActionAxisProbe


MODEL_STEP = 10_000
EPISODES = 3
SEED = None
AGENT_ID = "bc_stochastic"
PROBE_SAMPLE_COUNT = 100
DATASET_ID = "hopper_simple"


def probe(
    dataset: Dataset,
    *,
    episodes: int = EPISODES,
    model_step: int = MODEL_STEP,
    seed: int | None = SEED,
) -> object:
    agent = BCStochasticAgent(obs_size=dataset.obs_dim, act_size=dataset.act_dim)
    task_id = _task_id(dataset.id, AGENT_ID)
    agent.load(task_id, model_step)

    def eval_fn(observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        o = torch.as_tensor(np.asarray(observations, dtype=np.float32), dtype=torch.float32, device=agent.device)
        a = torch.as_tensor(np.asarray(actions, dtype=np.float32), dtype=torch.float32, device=agent.device)
        with torch.no_grad():
            mean, logstd = agent.actor.pi.dist(o)
            log_prob = Normal(mean, logstd.exp()).log_prob(a).sum(dim=-1)
        return log_prob.cpu().numpy().astype(np.float32)

    return probe_replay(
        task_id,
        dataset,
        ActionAxisProbe(PROBE_SAMPLE_COUNT),
        eval_fn,
        episodes=episodes,
        seed=seed,
    )


if __name__ == "__main__":
    dataset = make_dataset(DATASET_ID, device="cuda")
    probe_data = probe(
        dataset=dataset,
        episodes=EPISODES,
        seed=SEED,
    )
    print(f"probe_data={probe_data.path}")
