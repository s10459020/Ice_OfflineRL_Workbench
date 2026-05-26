
import numpy as np
import torch

import d3rlpy
from d3rlpy.models.torch.imitators import compute_discrete_imitation_loss
from d3rlpy.models.torch.policies import CategoricalPolicy
from d3rlpy.torch_utility import TorchMiniBatch

from ice_offline.agent.bc_discrete import BCAgentDiscrete
from ice_offline.tools.printer import print_stage

OBS_DIM = 8
N_ACTIONS = 4
DEVICE = "cpu"
SEED = 42
BATCH_SIZE = 64
N_TEST_BATCHES = 30


# ====================
# 1) Flow Function
# ====================
def build_our_agent() -> BCAgentDiscrete:
    return BCAgentDiscrete(obs_size=OBS_DIM, act_size=N_ACTIONS)

def build_d3rl():
    config = d3rlpy.algos.DiscreteBCConfig()
    algo = config.create(device=DEVICE)
    algo.create_impl(observation_shape=(OBS_DIM,), action_size=N_ACTIONS)
    assert algo.impl is not None
    return algo

def copy_d3rl_weights_to_our(
    d3_policy: CategoricalPolicy, our_agent: BCAgentDiscrete
) -> None:
    with torch.no_grad():
        for our_param, d3_param in _all_pairs(our_agent, d3_policy):
            our_param.copy_(d3_param)

def sample_observation(rng: np.random.Generator, batch: int, size: int) -> torch.Tensor:
    return torch.as_tensor(rng.standard_normal((batch, size)), dtype=torch.float32, device=DEVICE)

def sample_transition(
    rng: np.random.Generator, batch: int, obs_size: int, act_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    obs_t = sample_observation(rng, batch, obs_size)
    act_t = torch.as_tensor(rng.integers(0, act_size, size=(batch,)), device=DEVICE)
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

def _assert_equal(pairs) -> None:
    max_diff = 0.0
    for x, y in pairs:
        if x is None or y is None:
            if x is None and y is None:
                continue
            raise SystemExit("FAIL: mismatch, one side is None")
        if torch.is_tensor(x) and torch.is_tensor(y):
            max_diff = max(max_diff, float((x - y).abs().max().item()))
        else:
            max_diff = max(max_diff, float(np.abs(x - y).max()))
    if max_diff != 0.0:
        raise SystemExit(f"FAIL: mismatch, max_abs_diff={max_diff:.12e}")


# ====================
# 2) Behavior Function
# ====================
def d3rl_action_best_batch(policy: CategoricalPolicy, obs_t: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return torch.argmax(policy(obs_t).logits, dim=1).cpu().numpy()

def d3rl_action_best_single(policy: CategoricalPolicy, obs_t: torch.Tensor) -> np.ndarray:
    return d3rl_action_best_batch(policy, obs_t)[0]

def _our_losses(
    our_agent: BCAgentDiscrete, obs_t: torch.Tensor, act_t: torch.Tensor
) -> torch.Tensor:
    logits = our_agent.policy(obs_t).logits
    loss_bc = our_agent.loss_bc(logits, act_t)
    loss_regular = our_agent.loss_regular(logits)
    loss = our_agent.loss_actor(obs_t, act_t)
    return torch.stack([loss, loss_bc, loss_regular])

def _d3rl_losses(
    d3_policy: CategoricalPolicy,
    obs_t: torch.Tensor,
    act_t: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    d3_loss = compute_discrete_imitation_loss(
        policy=d3_policy,
        x=obs_t,
        action=act_t,
        beta=beta,
    )
    return torch.stack(
        [d3_loss.loss, d3_loss.imitation_loss, d3_loss.regularization_loss]
    )

def _all_pairs(our_agent: BCAgentDiscrete, d3_policy: CategoricalPolicy):
    return [
        (our_agent.policy.network[0].weight, d3_policy._encoder._layers[0].weight),
        (our_agent.policy.network[0].bias, d3_policy._encoder._layers[0].bias),
        (our_agent.policy.network[2].weight, d3_policy._encoder._layers[2].weight),
        (our_agent.policy.network[2].bias, d3_policy._encoder._layers[2].bias),
        (our_agent.policy.network[4].weight, d3_policy._fc.weight),
        (our_agent.policy.network[4].bias, d3_policy._fc.bias),
    ]


# ====================
# 3) main Function
# ====================
def main() -> None:
    print_stage("Init")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    our_agent = build_our_agent()
    d3rl = build_d3rl()
    d3_policy = d3rl.impl.modules.imitator
    copy_d3rl_weights_to_our(d3_policy, our_agent)

    print_stage("Act Compare")
    rng = np.random.default_rng(SEED)
    for i in range(1, N_TEST_BATCHES + 1):
        obs_single = sample_observation(rng, 1, OBS_DIM)
        obs_batch = sample_observation(rng, BATCH_SIZE, OBS_DIM)

        # act_single: d3rl best action vs our act
        d3_act = d3rl_action_best_single(d3_policy, obs_single)
        our_act = our_agent.act(obs_single[0], epsilon=0.0)
        _assert_equal([(d3_act, our_act)])

        # act_batch: d3rl best batch action vs our act_batch
        d3_batch = d3rl_action_best_batch(d3_policy, obs_batch)
        our_batch = our_agent.act_batch(obs_batch.cpu().numpy(), epsilon=0.0)
        _assert_equal([(d3_batch, our_batch)])

        print(f"batch={i}/{N_TEST_BATCHES} action_match=True")


    print_stage("Loss Compare")
    rng = np.random.default_rng(SEED + 1)
    for i in range(1, N_TEST_BATCHES + 1):
        obs_t, act_t = sample_transition(rng, BATCH_SIZE, OBS_DIM, N_ACTIONS)
        our_losses = _our_losses(our_agent, obs_t, act_t)
        d3rl_losses = _d3rl_losses(d3_policy, obs_t, act_t, our_agent.beta)
        _assert_equal([(d3rl_losses, our_losses)])
        print(f"batch={i}/{N_TEST_BATCHES} loss_match=True")


    print_stage("Update Compare")
    rng = np.random.default_rng(SEED + 2)
    for i in range(1, N_TEST_BATCHES + 1):
        obs_t, act_t = sample_transition(rng, BATCH_SIZE, OBS_DIM, N_ACTIONS)
        batch = _torch_batch(obs_t, act_t)

        our_agent.update({"obs": obs_t, "act": act_t})
        _ = d3rl.impl.inner_update(batch, i)

        _assert_equal(_all_pairs(our_agent, d3_policy))
        print(f"batch={i}/{N_TEST_BATCHES} param_match=True")

    print_stage("Result")
    print("PASS: act, act_batch, loss, and full update params are aligned with d3rl.")


if __name__ == "__main__":
    main()

