import numpy as np
import torch
from _lib import assert_callback
from _lib import sample_transition
from _lib import torch_buffer
from d3rlpy_master.d3rlpy import algos
from d3rlpy_master.d3rlpy.models.torch.imitators import compute_deterministic_imitation_loss
from d3rlpy_master.d3rlpy.models.torch.policies import DeterministicPolicy
from d3rlpy_master.d3rlpy.torch_utility import TorchMiniBatch
from ice_offline.agent.bc_continuous_deterministic import (
    BCAgentContinuousDeterministic,
)
from ice_offline.tools.printer import print_stage


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
# Mapping
# ====================
def _all_pairs(our: BCAgentContinuousDeterministic, ref_policy: DeterministicPolicy):
    return [
        (our.policy.network[0].weight, ref_policy._encoder._layers[0].weight),
        (our.policy.network[0].bias, ref_policy._encoder._layers[0].bias),
        (our.policy.network[2].weight, ref_policy._encoder._layers[2].weight),
        (our.policy.network[2].bias, ref_policy._encoder._layers[2].bias),
        (our.policy.network[4].weight, ref_policy._fc.weight),
        (our.policy.network[4].bias, ref_policy._fc.bias),
    ]


# ====================
# Ref define
# ====================
def _torch_batch(
    s: torch.Tensor,
    a: torch.Tensor,
    r: torch.Tensor,
    sn: torch.Tensor,
    d: torch.Tensor,
) -> TorchMiniBatch:
    ones = torch.ones_like(r)
    return TorchMiniBatch(
        observations=s,
        actions=a,
        rewards=r,
        next_observations=sn,
        next_actions=a,
        returns_to_go=torch.zeros_like(r),
        terminals=d,
        intervals=ones,
        device=DEVICE,
    )

def ref_action_best_batch(ref_policy: DeterministicPolicy, obs_t: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return ref_policy(obs_t).squashed_mu.cpu().numpy()

def ref_action_best_single(ref_policy: DeterministicPolicy, obs_t: torch.Tensor) -> np.ndarray:
    return ref_action_best_batch(ref_policy, obs_t)[0]

def ref_loss_actor(
    ref_policy: DeterministicPolicy,
    obs_t: torch.Tensor,
    act_t: torch.Tensor,
) -> torch.Tensor:
    return compute_deterministic_imitation_loss(
        policy=ref_policy,
        x=obs_t,
        action=act_t,
    ).loss

def ref_update_and_collect_params(
    ref,
    batch: TorchMiniBatch,
    step: int,
    our: BCAgentContinuousDeterministic,
    ref_policy: DeterministicPolicy,
):
    _ = ref.impl.inner_update(batch, step)
    return [x for _, x in _all_pairs(our, ref_policy)]


# ====================
# Our define
# ====================
def our_update_and_collect_params(
    our: BCAgentContinuousDeterministic,
    s: torch.Tensor,
    a: torch.Tensor,
    r: torch.Tensor,
    sn: torch.Tensor,
    d: torch.Tensor,
    ref_policy: DeterministicPolicy,
):
    our.update(torch_buffer(s, a, r, sn, d))
    return [y for y, _ in _all_pairs(our, ref_policy)]


# ====================
# Compare
# ====================
def build_our() -> BCAgentContinuousDeterministic:
    return BCAgentContinuousDeterministic(obs_size=OBS_DIM, act_size=ACT_DIM)

def build_ref():
    config = algos.BCConfig()
    ref = config.create(device=DEVICE)
    ref.create_impl(observation_shape=(OBS_DIM,), action_size=ACT_DIM)
    assert ref.impl is not None
    return ref

def init_compare() -> tuple[BCAgentContinuousDeterministic, object, DeterministicPolicy]:
    print_stage("Init")
    our = build_our()
    ref = build_ref()
    ref_policy = ref.impl.modules.imitator
    with torch.no_grad():
        for our_param, ref_param in _all_pairs(our, ref_policy):
            our_param.copy_(ref_param)
    return our, ref, ref_policy

def compare_act(
    our: BCAgentContinuousDeterministic,
    ref_policy: DeterministicPolicy,
) -> None:
    print_stage("Act Compare")
    for i in range(1, N_TEST_BATCHES + 1):
        obs_single, _, _, _, _ = sample_transition(1, OBS_DIM, ACT_DIM, DEVICE)
        obs_batch, _, _, _, _ = sample_transition(BATCH_SIZE, OBS_DIM, ACT_DIM, DEVICE)

        # act best
        assert_callback(
            lambda: [ref_action_best_single(ref_policy, obs_single)],
            lambda: [our.act(obs_single[0])],
            label=f"act_single[{i}]",
            seed=SEED + i,
        )

        # act best batch
        assert_callback(
            lambda: [ref_action_best_batch(ref_policy, obs_batch)],
            lambda: [our.act_batch(obs_batch.cpu().numpy())],
            label=f"act_batch[{i}]",
            seed=SEED + i,
        )

        print(f"batch={i}/{N_TEST_BATCHES} action_match=True")

def compare_loss(
    our: BCAgentContinuousDeterministic,
    ref_policy: DeterministicPolicy,
) -> None:
    print_stage("Loss Compare")
    for i in range(1, N_TEST_BATCHES + 1):
        s, a, _, _, _ = sample_transition(BATCH_SIZE, OBS_DIM, ACT_DIM, DEVICE)

        # loss actor
        assert_callback(
            lambda: [ref_loss_actor(ref_policy, s, a)],
            lambda: [our.loss_actor(s, a)],
            label=f"loss_actor[{i}]",
            seed=SEED + i,
        )

        print(f"batch={i}/{N_TEST_BATCHES} loss_match=True")

def compare_param(
    our: BCAgentContinuousDeterministic,
    ref,
    ref_policy: DeterministicPolicy,
) -> None:
    print_stage("Update Compare")
    for i in range(1, N_TEST_BATCHES + 1):
        s, a, r, sn, d = sample_transition(BATCH_SIZE, OBS_DIM, ACT_DIM, DEVICE)
        batch = _torch_batch(s, a, r, sn, d)

        # update params
        assert_callback(
            lambda: ref_update_and_collect_params(ref, batch, i, our, ref_policy),
            lambda: our_update_and_collect_params(our, s, a, r, sn, d, ref_policy),
            label=f"update[{i}]",
            seed=SEED + i,
        )

        print(f"batch={i}/{N_TEST_BATCHES} param_match=True")


# ====================
# Main
# ====================
if __name__ == "__main__":
    our, ref, ref_policy = init_compare()
    compare_act(our, ref_policy)
    compare_loss(our, ref_policy)
    compare_param(our, ref, ref_policy)
    print_stage("Result")
    print("PASS: act, act_batch, loss, and full update params are aligned with d3rl.")
