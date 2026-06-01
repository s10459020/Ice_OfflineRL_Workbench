import numpy as np
import torch
from _lib import assert_callback
from _lib import sample_transition
from _lib import torch_buffer
from d3rlpy_master.d3rlpy import algos
from d3rlpy_master.d3rlpy.torch_utility import TorchMiniBatch
from ice_offline.agent.iql import IQLAgent
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
def _all_pairs(our: IQLAgent, ref):
    ref_policy = ref.impl.modules.policy
    ref_q1 = ref.impl.modules.q_funcs[0]
    ref_q2 = ref.impl.modules.q_funcs[1]
    ref_t1 = ref.impl.modules.targ_q_funcs[0]
    ref_t2 = ref.impl.modules.targ_q_funcs[1]
    ref_v = ref.impl.modules.value_func
    return [
        (our.actor.hidden[0].weight, ref_policy._encoder._layers[0].weight),
        (our.actor.hidden[0].bias, ref_policy._encoder._layers[0].bias),
        (our.actor.hidden[2].weight, ref_policy._encoder._layers[2].weight),
        (our.actor.hidden[2].bias, ref_policy._encoder._layers[2].bias),
        (our.actor.mean_head.weight, ref_policy._mu.weight),
        (our.actor.mean_head.bias, ref_policy._mu.bias),
        (our.actor.logstd, ref_policy._logstd),
        (our.critic.q1.network[0].weight, ref_q1._encoder._layers[0].weight),
        (our.critic.q1.network[0].bias, ref_q1._encoder._layers[0].bias),
        (our.critic.q1.network[2].weight, ref_q1._encoder._layers[2].weight),
        (our.critic.q1.network[2].bias, ref_q1._encoder._layers[2].bias),
        (our.critic.q1.network[4].weight, ref_q1._fc.weight),
        (our.critic.q1.network[4].bias, ref_q1._fc.bias),
        (our.critic.q2.network[0].weight, ref_q2._encoder._layers[0].weight),
        (our.critic.q2.network[0].bias, ref_q2._encoder._layers[0].bias),
        (our.critic.q2.network[2].weight, ref_q2._encoder._layers[2].weight),
        (our.critic.q2.network[2].bias, ref_q2._encoder._layers[2].bias),
        (our.critic.q2.network[4].weight, ref_q2._fc.weight),
        (our.critic.q2.network[4].bias, ref_q2._fc.bias),
        (our.critic.targ_q1.network[0].weight, ref_t1._encoder._layers[0].weight),
        (our.critic.targ_q1.network[0].bias, ref_t1._encoder._layers[0].bias),
        (our.critic.targ_q1.network[2].weight, ref_t1._encoder._layers[2].weight),
        (our.critic.targ_q1.network[2].bias, ref_t1._encoder._layers[2].bias),
        (our.critic.targ_q1.network[4].weight, ref_t1._fc.weight),
        (our.critic.targ_q1.network[4].bias, ref_t1._fc.bias),
        (our.critic.targ_q2.network[0].weight, ref_t2._encoder._layers[0].weight),
        (our.critic.targ_q2.network[0].bias, ref_t2._encoder._layers[0].bias),
        (our.critic.targ_q2.network[2].weight, ref_t2._encoder._layers[2].weight),
        (our.critic.targ_q2.network[2].bias, ref_t2._encoder._layers[2].bias),
        (our.critic.targ_q2.network[4].weight, ref_t2._fc.weight),
        (our.critic.targ_q2.network[4].bias, ref_t2._fc.bias),
        (our.v.network[0].weight, ref_v._encoder._layers[0].weight),
        (our.v.network[0].bias, ref_v._encoder._layers[0].bias),
        (our.v.network[2].weight, ref_v._encoder._layers[2].weight),
        (our.v.network[2].bias, ref_v._encoder._layers[2].bias),
        (our.v.network[4].weight, ref_v._fc.weight),
        (our.v.network[4].bias, ref_v._fc.bias),
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

def ref_action_best_batch(ref, obs_t: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return ref.impl.modules.policy(obs_t).squashed_mu.cpu().numpy()

def ref_action_sample_batch(ref, obs_t: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return ref.impl.inner_sample_action(obs_t).cpu().numpy()

def ref_action_best_single(ref, obs_t: torch.Tensor) -> np.ndarray:
    return ref_action_best_batch(ref, obs_t)[0]

def ref_action_sample_single(ref, obs_t: torch.Tensor) -> np.ndarray:
    return ref_action_sample_batch(ref, obs_t)[0]

def ref_loss_pack(ref, batch: TorchMiniBatch, obs_t: torch.Tensor):
    q_tpn = ref.impl.compute_target(batch)
    critic_obj = ref.impl.compute_critic_loss(batch, q_tpn)
    action = ref.impl.modules.policy(obs_t)
    actor = ref.impl.compute_actor_loss(batch, action).actor_loss
    return critic_obj, actor

def ref_update_and_collect_params(ref, batch: TorchMiniBatch, step: int, our: IQLAgent):
    _ = ref.impl.inner_update(batch, step)
    return [x for _, x in _all_pairs(our, ref)]


# ====================
# Our define
# ====================
def our_update_and_collect_params(
    our: IQLAgent,
    s: torch.Tensor,
    a: torch.Tensor,
    r: torch.Tensor,
    sn: torch.Tensor,
    d: torch.Tensor,
    ref,
):
    our.update(torch_buffer(s, a, r, sn, d))
    return [y for y, _ in _all_pairs(our, ref)]


# ====================
# Compare
# ====================
def build_our() -> IQLAgent:
    return IQLAgent(obs_size=OBS_DIM, act_size=ACT_DIM)

def build_ref():
    config = algos.IQLConfig()
    ref = config.create(device=DEVICE)
    ref.create_impl(observation_shape=(OBS_DIM,), action_size=ACT_DIM)
    assert ref.impl is not None
    return ref

def init_compare() -> tuple[IQLAgent, object]:
    print_stage("Init")
    our = build_our()
    ref = build_ref()
    with torch.no_grad():
        for our_param, ref_param in _all_pairs(our, ref):
            our_param.copy_(ref_param)
    return our, ref

def compare_act(our: IQLAgent, ref) -> None:
    print_stage("Act Compare")
    for i in range(1, N_TEST_BATCHES + 1):
        s_single, _, _, _, _ = sample_transition(1, OBS_DIM, ACT_DIM, DEVICE)
        s, _, _, _, _ = sample_transition(BATCH_SIZE, OBS_DIM, ACT_DIM, DEVICE)

        # act best
        assert_callback(
            lambda: [ref_action_best_single(ref, s_single)],
            lambda: [our.act(s_single[0], greedy=True)],
            label=f"act_best_single[{i}]",
            seed=SEED + i,
        )

        # act best batch
        assert_callback(
            lambda: [ref_action_best_batch(ref, s)],
            lambda: [our.act_batch(s.cpu().numpy(), greedy=True)],
            label=f"act_best_batch[{i}]",
            seed=SEED + i,
        )

        # act sample
        assert_callback(
            lambda: [ref_action_sample_single(ref, s_single)],
            lambda: [our.act(s_single[0], greedy=False)],
            label=f"act_sample_single[{i}]",
            seed=SEED + i,
        )

        # act sample batch
        assert_callback(
            lambda: [ref_action_sample_batch(ref, s)],
            lambda: [our.act_batch(s.cpu().numpy(), greedy=False)],
            label=f"act_sample_batch[{i}]",
            seed=SEED + i,
        )

        print(f"batch={i}/{N_TEST_BATCHES} act_match=True")

def compare_loss(our: IQLAgent, ref) -> None:
    print_stage("Loss Compare")
    for i in range(1, N_TEST_BATCHES + 1):
        s, a, r, sn, d = sample_transition(BATCH_SIZE, OBS_DIM, ACT_DIM, DEVICE)
        batch = _torch_batch(s, a, r, sn, d)

        # loss q
        assert_callback(
            lambda: [ref_loss_pack(ref, batch, s)[0].q_loss],
            lambda: [our.loss_q(s, a, r, sn, d)],
            label=f"loss_q[{i}]",
            seed=SEED + i,
        )

        # loss v
        assert_callback(
            lambda: [ref_loss_pack(ref, batch, s)[0].v_loss],
            lambda: [our.loss_v(s, a)],
            label=f"loss_v[{i}]",
            seed=SEED + i,
        )

        # loss actor
        assert_callback(
            lambda: [ref_loss_pack(ref, batch, s)[1]],
            lambda: [our.loss_actor(s, a)],
            label=f"loss_actor[{i}]",
            seed=SEED + i,
        )

        # loss critic
        assert_callback(
            lambda: [ref_loss_pack(ref, batch, s)[0].critic_loss],
            lambda: [our.loss_critic(s, a, r, sn, d)],
            label=f"loss_critic[{i}]",
            seed=SEED + i,
        )

        print(f"batch={i}/{N_TEST_BATCHES} loss_match=True")

def compare_param(our: IQLAgent, ref) -> None:
    print_stage("Update Compare")
    for i in range(1, N_TEST_BATCHES + 1):
        s, a, r, sn, d = sample_transition(BATCH_SIZE, OBS_DIM, ACT_DIM, DEVICE)
        batch = _torch_batch(s, a, r, sn, d)

        # update params
        assert_callback(
            lambda: ref_update_and_collect_params(ref, batch, i, our),
            lambda: our_update_and_collect_params(our, s, a, r, sn, d, ref),
            label=f"update[{i}]",
            seed=SEED + i,
        )

        print(f"batch={i}/{N_TEST_BATCHES} param_match=True")


# ====================
# Main
# ====================
if __name__ == "__main__":
    our, ref = init_compare()
    compare_act(our, ref)
    compare_loss(our, ref)
    compare_param(our, ref)
    print_stage("Result")
    print("PASS: sample, act, act_batch, loss, and full update params are aligned with d3rl.")
