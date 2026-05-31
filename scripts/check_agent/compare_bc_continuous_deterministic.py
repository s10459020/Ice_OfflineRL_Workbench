import numpy as np
import torch
import d3rlpy
from d3rlpy.models.torch.imitators import compute_deterministic_imitation_loss
from d3rlpy.models.torch.policies import DeterministicPolicy
from d3rlpy.torch_utility import TorchMiniBatch
from ice_offline.agent.bc_continuous_deterministic import (
    BCAgentContinuousDeterministic,
)
from ice_offline.dataset._spec import TorchBuffer
from ice_offline.tools.printer import print_stage
from _lib import assert_callback
# ====================
# Config
# ====================

OBS_DIM = 8
ACT_DIM = 3
DEVICE = "cpu"
SEED = 42
BATCH_SIZE = 64
N_TEST_BATCHES = 30
# ====================
# Mapping: all_pairs
# ====================

def _all_pairs(our_agent: BCAgentContinuousDeterministic, d3_policy: DeterministicPolicy):
    return [
        (our_agent.policy.network[0].weight, d3_policy._encoder._layers[0].weight),
        (our_agent.policy.network[0].bias, d3_policy._encoder._layers[0].bias),
        (our_agent.policy.network[2].weight, d3_policy._encoder._layers[2].weight),
        (our_agent.policy.network[2].bias, d3_policy._encoder._layers[2].bias),
        (our_agent.policy.network[4].weight, d3_policy._fc.weight),
        (our_agent.policy.network[4].bias, d3_policy._fc.bias),
    ]
# ====================
# common
# ====================

def build_our_agent() -> BCAgentContinuousDeterministic:
    return BCAgentContinuousDeterministic(obs_size=OBS_DIM, act_size=ACT_DIM)

def build_d3rl():
    config = d3rlpy.algos.BCConfig()
    algo = config.create(device=DEVICE)
    algo.create_impl(observation_shape=(OBS_DIM,), action_size=ACT_DIM)
    assert algo.impl is not None
    return algo

def sample_observation(rng: np.random.Generator, batch: int, size: int) -> torch.Tensor:
    return torch.as_tensor(rng.standard_normal((batch, size)), dtype=torch.float32, device=DEVICE)

def sample_transition(
    rng: np.random.Generator, batch: int, obs_size: int, act_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    obs_t = sample_observation(rng, batch, obs_size)
    act_t = torch.as_tensor(rng.standard_normal((batch, act_size)), dtype=torch.float32, device=DEVICE)
    return obs_t, act_t

def _torch_batch(obs_t: torch.Tensor, act_t: torch.Tensor) -> TorchMiniBatch:
    zeros = torch.zeros((obs_t.shape[0], 1), dtype=torch.float32, device=DEVICE)
    ones = torch.ones_like(zeros)
    return TorchMiniBatch(
        observations=obs_t,
        actions=act_t,
        rewards=zeros,
        next_observations=obs_t,
        next_actions=act_t,
        returns_to_go=zeros,
        terminals=zeros,
        intervals=ones,
        device=DEVICE,
    )

def _torch_buffer(obs_t: torch.Tensor, act_t: torch.Tensor) -> TorchBuffer:
    zeros = torch.zeros((obs_t.shape[0], 1), dtype=torch.float32, device=DEVICE)
    return TorchBuffer(
        obs_list=obs_t,
        next_obs_list=obs_t,
        act_list=act_t,
        rew_list=zeros,
        done_list=zeros,
    )
# ====================
# Ref Math
# ====================

def d3rl_action_best_batch(policy: DeterministicPolicy, obs_t: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return policy(obs_t).squashed_mu.cpu().numpy()

def d3rl_action_best_single(policy: DeterministicPolicy, obs_t: torch.Tensor) -> np.ndarray:
    return d3rl_action_best_batch(policy, obs_t)[0]
# ====================
# Our Math
# ====================

def _d3rl_loss_actor(
    d3_policy: DeterministicPolicy,
    obs_t: torch.Tensor,
    act_t: torch.Tensor,
) -> torch.Tensor:
    loss = compute_deterministic_imitation_loss(
        policy=d3_policy,
        x=obs_t,
        action=act_t,
    ).loss
    return loss

def _ref_update_and_collect_params(
    d3rl,
    batch: TorchMiniBatch,
    step: int,
    our_agent: BCAgentContinuousDeterministic,
    d3_policy: DeterministicPolicy,
):
    _ = d3rl.impl.inner_update(batch, step)
    return [x for _, x in _all_pairs(our_agent, d3_policy)]

def _our_update_and_collect_params(
    our_agent: BCAgentContinuousDeterministic,
    obs_t: torch.Tensor,
    act_t: torch.Tensor,
    d3_policy: DeterministicPolicy,
):
    our_agent.update(_torch_buffer(obs_t, act_t))
    return [y for y, _ in _all_pairs(our_agent, d3_policy)]
# ====================
# Compare
# ====================
def init_compare() -> tuple[BCAgentContinuousDeterministic, object, DeterministicPolicy]:
    print_stage("Init")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    our_agent = build_our_agent()
    d3rl = build_d3rl()
    d3_policy = d3rl.impl.modules.imitator
    with torch.no_grad():
        for our_param, d3_param in _all_pairs(our_agent, d3_policy):
            our_param.copy_(d3_param)
    return our_agent, d3rl, d3_policy

def compare_act(
    our_agent: BCAgentContinuousDeterministic,
    d3_policy: DeterministicPolicy,
) -> None:
    print_stage("Act Compare")
    rng = np.random.default_rng(SEED)
    for i in range(1, N_TEST_BATCHES + 1):
        obs_single = sample_observation(rng, 1, OBS_DIM)
        obs_batch = sample_observation(rng, BATCH_SIZE, OBS_DIM)
        # act_single: d3rl best action vs our act
        assert_callback(
            lambda: [d3rl_action_best_single(d3_policy, obs_single)],
            lambda: [our_agent.act(obs_single[0], greedy=True)],
            label=f"act_single[{i}]",
            seed=SEED + i,
        )
        # act_batch: d3rl best batch action vs our act_batch
        assert_callback(
            lambda: [d3rl_action_best_batch(d3_policy, obs_batch)],
            lambda: [our_agent.act_batch(obs_batch.cpu().numpy(), greedy=True)],
            label=f"act_batch[{i}]",
            seed=SEED + 1000 + i,
        )
        print(f"batch={i}/{N_TEST_BATCHES} action_match=True")

def compare_loss(
    our_agent: BCAgentContinuousDeterministic,
    d3_policy: DeterministicPolicy,
) -> None:
    print_stage("Loss Compare")
    rng = np.random.default_rng(SEED + 1)
    for i in range(1, N_TEST_BATCHES + 1):
        obs_t, act_t = sample_transition(rng, BATCH_SIZE, OBS_DIM, ACT_DIM)
        assert_callback(
            lambda: [_d3rl_loss_actor(d3_policy, obs_t, act_t)],
            lambda: [our_agent.loss_actor(obs_t, act_t)],
            label=f"loss_actor[{i}]",
            seed=SEED + 2000 + i,
        )
        print(f"batch={i}/{N_TEST_BATCHES} loss_match=True")

def compare_param(
    our_agent: BCAgentContinuousDeterministic,
    d3rl,
    d3_policy: DeterministicPolicy,
) -> None:
    print_stage("Update Compare")
    rng = np.random.default_rng(SEED + 2)
    for i in range(1, N_TEST_BATCHES + 1):
        obs_t, act_t = sample_transition(rng, BATCH_SIZE, OBS_DIM, ACT_DIM)
        batch = _torch_batch(obs_t, act_t)
        assert_callback(
            lambda: _ref_update_and_collect_params(d3rl, batch, i, our_agent, d3_policy),
            lambda: _our_update_and_collect_params(our_agent, obs_t, act_t, d3_policy),
            label=f"update[{i}]",
            seed=SEED + 3000 + i,
        )
        print(f"batch={i}/{N_TEST_BATCHES} param_match=True")

def main() -> None:
    our_agent, d3rl, d3_policy = init_compare()
    compare_act(our_agent, d3_policy)
    compare_loss(our_agent, d3_policy)
    compare_param(our_agent, d3rl, d3_policy)
    print_stage("Result")
    print("PASS: act, act_batch, loss, and full update params are aligned with d3rl.")
# ====================
# __main__
# ====================

if __name__ == "__main__":
    main()
