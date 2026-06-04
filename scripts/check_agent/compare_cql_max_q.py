import numpy as np
import torch

from _lib import assert_callback
from _lib import sample_transition
from _lib import torch_buffer
from d3rlpy_master.d3rlpy import algos
from d3rlpy_master.d3rlpy.models.torch import build_squashed_gaussian_distribution
from d3rlpy_master.d3rlpy.models.torch import get_parameter
from d3rlpy_master.d3rlpy.torch_utility import TorchMiniBatch
from ice_offline.agent._spec import agent_batch
from ice_offline.agent.cql_max_q import CQLMaxQAgent
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
def _all_pairs(our: CQLMaxQAgent, ref):
    ref_policy = ref.impl.modules.policy
    ref_q1 = ref.impl.modules.q_funcs[0]
    ref_q2 = ref.impl.modules.q_funcs[1]
    ref_t1 = ref.impl.modules.targ_q_funcs[0]
    ref_t2 = ref.impl.modules.targ_q_funcs[1]
    return [
        (our.actor.pi.network[0].weight, ref_policy._encoder._layers[0].weight),
        (our.actor.pi.network[0].bias, ref_policy._encoder._layers[0].bias),
        (our.actor.pi.network[2].weight, ref_policy._encoder._layers[2].weight),
        (our.actor.pi.network[2].bias, ref_policy._encoder._layers[2].bias),
        (our.actor.pi.mean_head.weight, ref_policy._mu.weight),
        (our.actor.pi.mean_head.bias, ref_policy._mu.bias),
        (our.actor.pi.logstd_head.weight, ref_policy._logstd.weight),
        (our.actor.pi.logstd_head.bias, ref_policy._logstd.bias),
        (our.critic.q_networks[0].network[0].weight, ref_q1._encoder._layers[0].weight),
        (our.critic.q_networks[0].network[0].bias, ref_q1._encoder._layers[0].bias),
        (our.critic.q_networks[0].network[2].weight, ref_q1._encoder._layers[2].weight),
        (our.critic.q_networks[0].network[2].bias, ref_q1._encoder._layers[2].bias),
        (our.critic.q_networks[0].network[4].weight, ref_q1._fc.weight),
        (our.critic.q_networks[0].network[4].bias, ref_q1._fc.bias),
        (our.critic.q_networks[1].network[0].weight, ref_q2._encoder._layers[0].weight),
        (our.critic.q_networks[1].network[0].bias, ref_q2._encoder._layers[0].bias),
        (our.critic.q_networks[1].network[2].weight, ref_q2._encoder._layers[2].weight),
        (our.critic.q_networks[1].network[2].bias, ref_q2._encoder._layers[2].bias),
        (our.critic.q_networks[1].network[4].weight, ref_q2._fc.weight),
        (our.critic.q_networks[1].network[4].bias, ref_q2._fc.bias),
        (our.critic.tq_networks[0].network[0].weight, ref_t1._encoder._layers[0].weight),
        (our.critic.tq_networks[0].network[0].bias, ref_t1._encoder._layers[0].bias),
        (our.critic.tq_networks[0].network[2].weight, ref_t1._encoder._layers[2].weight),
        (our.critic.tq_networks[0].network[2].bias, ref_t1._encoder._layers[2].bias),
        (our.critic.tq_networks[0].network[4].weight, ref_t1._fc.weight),
        (our.critic.tq_networks[0].network[4].bias, ref_t1._fc.bias),
        (our.critic.tq_networks[1].network[0].weight, ref_t2._encoder._layers[0].weight),
        (our.critic.tq_networks[1].network[0].bias, ref_t2._encoder._layers[0].bias),
        (our.critic.tq_networks[1].network[2].weight, ref_t2._encoder._layers[2].weight),
        (our.critic.tq_networks[1].network[2].bias, ref_t2._encoder._layers[2].bias),
        (our.critic.tq_networks[1].network[4].weight, ref_t2._fc.weight),
        (our.critic.tq_networks[1].network[4].bias, ref_t2._fc.bias),
        (our.temp.log_alpha, ref.impl.modules.log_temp._parameter),
        (our.multiplier.log_cql_multiplier, ref.impl.modules.log_alpha._parameter),
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

def ref_loss_td(ref, batch: TorchMiniBatch) -> torch.Tensor:
    q_tpn = ref.impl.compute_target(batch)
    q_forwarders = ref.impl._q_func_forwarder.forwarders
    loss_td1 = q_forwarders[0].compute_error(
        observations=batch.observations,
        actions=batch.actions,
        rewards=batch.rewards,
        target=q_tpn,
        terminals=batch.terminals,
        gamma=ref.impl._gamma**batch.intervals,
        reduction="none",
    ).mean()
    loss_td2 = q_forwarders[1].compute_error(
        observations=batch.observations,
        actions=batch.actions,
        rewards=batch.rewards,
        target=q_tpn,
        terminals=batch.terminals,
        gamma=ref.impl._gamma**batch.intervals,
        reduction="none",
    ).mean()
    return loss_td1 + loss_td2

def ref_loss_suppress_scaled(ref, batch: TorchMiniBatch) -> torch.Tensor:
    return ref.impl._compute_conservative_loss(
        obs_t=batch.observations,
        act_t=batch.actions,
        obs_tp1=batch.next_observations,
        returns_to_go=batch.returns_to_go,
    )

def ref_loss_suppress(ref, batch: TorchMiniBatch) -> torch.Tensor:
    loss_scaled = ref_loss_suppress_scaled(ref, batch)
    return loss_scaled / ref.impl._conservative_weight + ref.impl._alpha_threshold

def ref_loss_critic(ref, batch: TorchMiniBatch) -> torch.Tensor:
    q_tpn = ref.impl.compute_target(batch)
    return ref.impl.compute_critic_loss(batch, q_tpn).critic_loss

def ref_loss_actor(ref, batch: TorchMiniBatch) -> torch.Tensor:
    action = ref.impl.modules.policy(batch.observations)
    return ref.impl.compute_actor_loss(batch, action).actor_loss

def ref_loss_alpha_sac(ref, obs_t: torch.Tensor) -> torch.Tensor:
    action_out = ref.impl.modules.policy(obs_t)
    dist = build_squashed_gaussian_distribution(action_out)
    _, log_prob = dist.sample_with_log_prob()
    return -(
        get_parameter(ref.impl.modules.log_temp).exp()
        * (log_prob.detach() - ref.impl.action_size)
    ).mean()

def ref_loss_cql_multiplier(
    ref,
    suppress_loss_detached: torch.Tensor,
) -> torch.Tensor:
    return -(
        get_parameter(ref.impl.modules.log_alpha).exp().clamp(0, 1e6)
        * suppress_loss_detached
    ).mean()

def ref_update_and_collect_params(ref, batch: TorchMiniBatch, step: int, our: CQLMaxQAgent):
    _ = ref.impl.inner_update(batch, step)
    return [x for _, x in _all_pairs(our, ref)]


# ====================
# Our define
# ====================
def our_update_and_collect_params(
    our: CQLMaxQAgent,
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
def build_our() -> CQLMaxQAgent:
    return CQLMaxQAgent(obs_size=OBS_DIM, act_size=ACT_DIM)

def build_ref():
    config = algos.CQLConfig(max_q_backup=True)
    ref = config.create(device=DEVICE)
    ref.create_impl(observation_shape=(OBS_DIM,), action_size=ACT_DIM)
    assert ref.impl is not None
    return ref

def init_compare() -> tuple[CQLMaxQAgent, object]:
    print_stage("Init")
    our = build_our()
    ref = build_ref()
    with torch.no_grad():
        for our_param, ref_param in _all_pairs(our, ref):
            our_param.copy_(ref_param)
    return our, ref

def compare_act(our: CQLMaxQAgent, ref) -> None:
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

def compare_loss(our: CQLMaxQAgent, ref) -> None:
    print_stage("Loss Compare")
    for i in range(1, N_TEST_BATCHES + 1):
        s, a, r, sn, d = sample_transition(BATCH_SIZE, OBS_DIM, ACT_DIM, DEVICE)
        batch = _torch_batch(s, a, r, sn, d)
        our_batch = agent_batch(torch_buffer(s, a, r, sn, d))
        suppress_scaled = ref_loss_suppress_scaled(ref, batch).detach()

        # loss td
        assert_callback(
            lambda: [ref_loss_td(ref, batch)],
            lambda: [our.loss_td(our_batch)],
            label=f"loss_td[{i}]",
            seed=SEED + i,
        )

        # loss suppress
        assert_callback(
            lambda: [ref_loss_suppress(ref, batch)],
            lambda: [our.loss_suppress(our_batch)],
            label=f"loss_suppress[{i}]",
            seed=SEED + i,
        )

        # loss actor
        assert_callback(
            lambda: [ref_loss_actor(ref, batch)],
            lambda: [our.loss_actor(our_batch)],
            label=f"loss_actor[{i}]",
            seed=SEED + i,
        )

        # loss critic
        assert_callback(
            lambda: [ref_loss_critic(ref, batch)],
            lambda: [our.loss_critic(our_batch)],
            label=f"loss_critic[{i}]",
            seed=SEED + i,
        )

        # loss alpha sac
        assert_callback(
            lambda: [ref_loss_alpha_sac(ref, s)],
            lambda: [our.temp.loss(our.actor.sample(s)[1])],
            label=f"loss_alpha_sac[{i}]",
            seed=SEED + i,
        )

        # loss alpha cql
        assert_callback(
            lambda: [ref_loss_cql_multiplier(ref, suppress_scaled)],
            lambda: [our.multiplier.loss(suppress_scaled)],
            label=f"loss_cql_multiplier[{i}]",
            seed=SEED + i,
        )

        print(f"batch={i}/{N_TEST_BATCHES} loss_match=True")

def compare_param(our: CQLMaxQAgent, ref) -> None:
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

