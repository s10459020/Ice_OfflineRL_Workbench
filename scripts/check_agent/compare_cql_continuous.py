import numpy as np
import torch
import torch.nn.functional as F

import d3rlpy
from d3rlpy.torch_utility import TorchMiniBatch
from d3rlpy.models.torch import build_squashed_gaussian_distribution, get_parameter

from ice_offline.agent.cql_continuous import CQLAgentContinuous
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
def _all_pairs(our: CQLAgentContinuous, ref):
    d3_policy = ref.impl.modules.policy
    d3_q1 = ref.impl.modules.q_funcs[0]
    d3_q2 = ref.impl.modules.q_funcs[1]
    d3_t1 = ref.impl.modules.targ_q_funcs[0]
    d3_t2 = ref.impl.modules.targ_q_funcs[1]
    return [
        (our.policy.hidden[0].weight, d3_policy._encoder._layers[0].weight),
        (our.policy.hidden[0].bias, d3_policy._encoder._layers[0].bias),
        (our.policy.hidden[2].weight, d3_policy._encoder._layers[2].weight),
        (our.policy.hidden[2].bias, d3_policy._encoder._layers[2].bias),
        (our.policy.mean_head.weight, d3_policy._mu.weight),
        (our.policy.mean_head.bias, d3_policy._mu.bias),
        (our.policy.logstd_head.weight, d3_policy._logstd.weight),
        (our.policy.logstd_head.bias, d3_policy._logstd.bias),
        (our.critic.q1.network[0].weight, d3_q1._encoder._layers[0].weight),
        (our.critic.q1.network[0].bias, d3_q1._encoder._layers[0].bias),
        (our.critic.q1.network[2].weight, d3_q1._encoder._layers[2].weight),
        (our.critic.q1.network[2].bias, d3_q1._encoder._layers[2].bias),
        (our.critic.q1.network[4].weight, d3_q1._fc.weight),
        (our.critic.q1.network[4].bias, d3_q1._fc.bias),
        (our.critic.q2.network[0].weight, d3_q2._encoder._layers[0].weight),
        (our.critic.q2.network[0].bias, d3_q2._encoder._layers[0].bias),
        (our.critic.q2.network[2].weight, d3_q2._encoder._layers[2].weight),
        (our.critic.q2.network[2].bias, d3_q2._encoder._layers[2].bias),
        (our.critic.q2.network[4].weight, d3_q2._fc.weight),
        (our.critic.q2.network[4].bias, d3_q2._fc.bias),
        (our.critic.targ_q1.network[0].weight, d3_t1._encoder._layers[0].weight),
        (our.critic.targ_q1.network[0].bias, d3_t1._encoder._layers[0].bias),
        (our.critic.targ_q1.network[2].weight, d3_t1._encoder._layers[2].weight),
        (our.critic.targ_q1.network[2].bias, d3_t1._encoder._layers[2].bias),
        (our.critic.targ_q1.network[4].weight, d3_t1._fc.weight),
        (our.critic.targ_q1.network[4].bias, d3_t1._fc.bias),
        (our.critic.targ_q2.network[0].weight, d3_t2._encoder._layers[0].weight),
        (our.critic.targ_q2.network[0].bias, d3_t2._encoder._layers[0].bias),
        (our.critic.targ_q2.network[2].weight, d3_t2._encoder._layers[2].weight),
        (our.critic.targ_q2.network[2].bias, d3_t2._encoder._layers[2].bias),
        (our.critic.targ_q2.network[4].weight, d3_t2._fc.weight),
        (our.critic.targ_q2.network[4].bias, d3_t2._fc.bias),
        (our.policy.log_alpha, ref.impl.modules.log_temp._parameter),
        (our.critic.log_alpha, ref.impl.modules.log_alpha._parameter),
    ]


# ====================
# common
# ====================
def build_our_agent() -> CQLAgentContinuous:
    return CQLAgentContinuous(obs_size=OBS_DIM, act_size=ACT_DIM)

def build_d3rl():
    config = d3rlpy.algos.CQLConfig()
    ref = config.create(device=DEVICE)
    ref.create_impl(observation_shape=(OBS_DIM,), action_size=ACT_DIM)
    assert ref.impl is not None
    return ref

def sample_transition(
    rng: np.random.Generator, batch: int, obs_size: int, act_size: int
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, TorchMiniBatch
]:
    obs_t = torch.as_tensor(rng.standard_normal((batch, obs_size)), dtype=torch.float32, device=DEVICE)
    act_t = torch.as_tensor(rng.standard_normal((batch, act_size)), dtype=torch.float32, device=DEVICE)
    rew_t = torch.as_tensor(rng.standard_normal((batch, 1)), dtype=torch.float32, device=DEVICE)
    next_obs_t = torch.as_tensor(rng.standard_normal((batch, obs_size)), dtype=torch.float32, device=DEVICE)
    done_t = torch.as_tensor(rng.integers(0, 2, size=(batch, 1)), dtype=torch.float32, device=DEVICE)
    batch_t = _torch_batch(obs_t, act_t, rew_t, next_obs_t, done_t)
    return obs_t, act_t, rew_t, next_obs_t, done_t, batch_t

def _torch_batch(obs_t, act_t, rew_t, next_obs_t, done_t) -> TorchMiniBatch:
    ones = torch.ones_like(rew_t)
    return TorchMiniBatch(
        observations=obs_t,
        actions=act_t,
        rewards=rew_t,
        next_observations=next_obs_t,
        next_actions=act_t,
        returns_to_go=torch.zeros_like(rew_t),
        terminals=done_t,
        intervals=ones,
        device=DEVICE,
    )


# ====================
# Ref Math
# ====================
def d3rl_action_best_batch(ref, obs_t: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return ref.impl.modules.policy(obs_t).squashed_mu.cpu().numpy()

def d3rl_action_sample_batch(ref, obs_t: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return ref.impl.inner_sample_action(obs_t).cpu().numpy()

def d3rl_action_best_single(ref, obs_t: torch.Tensor) -> np.ndarray:
    return d3rl_action_best_batch(ref, obs_t)[0]

def d3rl_action_sample_single(ref, obs_t: torch.Tensor) -> np.ndarray:
    return d3rl_action_sample_batch(ref, obs_t)[0]

def _d3rl_loss_td(d3rl, batch: TorchMiniBatch) -> torch.Tensor:
    q_tpn = d3rl.impl.compute_target(batch)
    q_forwarders = d3rl.impl._q_func_forwarder.forwarders
    loss_td1 = q_forwarders[0].compute_error(
        observations=batch.observations,
        actions=batch.actions,
        rewards=batch.rewards,
        target=q_tpn,
        terminals=batch.terminals,
        gamma=d3rl.impl._gamma**batch.intervals,
        reduction="none",
    ).mean()
    loss_td2 = q_forwarders[1].compute_error(
        observations=batch.observations,
        actions=batch.actions,
        rewards=batch.rewards,
        target=q_tpn,
        terminals=batch.terminals,
        gamma=d3rl.impl._gamma**batch.intervals,
        reduction="none",
    ).mean()
    return torch.stack([loss_td1, loss_td2], dim=0)

def _d3rl_loss_conservative_scaled(d3rl, batch: TorchMiniBatch) -> torch.Tensor:
    return d3rl.impl._compute_conservative_loss(
        obs_t=batch.observations,
        act_t=batch.actions,
        obs_tp1=batch.next_observations,
        returns_to_go=batch.returns_to_go,
    )

def _d3rl_loss_conservative(d3rl, batch: TorchMiniBatch) -> torch.Tensor:
    loss_scaled = _d3rl_loss_conservative_scaled(d3rl, batch)
    return (loss_scaled / d3rl.impl._conservative_weight + d3rl.impl._alpha_threshold)

def _d3rl_loss_critic(d3rl, batch: TorchMiniBatch) -> torch.Tensor:
    q_tpn = d3rl.impl.compute_target(batch)
    return d3rl.impl.compute_critic_loss(batch, q_tpn).critic_loss

def _d3rl_loss_actor(d3rl, obs_t: torch.Tensor) -> torch.Tensor:
    action_out = d3rl.impl.modules.policy(obs_t)
    dist = build_squashed_gaussian_distribution(action_out)
    sampled_action, log_prob = dist.sample_with_log_prob()
    q_t = d3rl.impl._q_func_forwarder.compute_expected_q(obs_t, sampled_action, "min")
    return (get_parameter(d3rl.impl.modules.log_temp).exp() * log_prob - q_t).mean()

def _d3rl_loss_alpha_sac(d3rl, obs_t: torch.Tensor) -> torch.Tensor:
    action_out = d3rl.impl.modules.policy(obs_t)
    dist = build_squashed_gaussian_distribution(action_out)
    _, log_prob = dist.sample_with_log_prob()
    return -(
        get_parameter(d3rl.impl.modules.log_temp).exp()
        * (log_prob.detach() - d3rl.impl.action_size)
    ).mean()

def _d3rl_loss_alpha_cql(d3rl, batch: TorchMiniBatch) -> torch.Tensor:
    conservative_scaled = _d3rl_loss_conservative_scaled(d3rl, batch)
    return -(
        get_parameter(d3rl.impl.modules.log_alpha).exp().clamp(0, 1e6)
        * conservative_scaled.detach()
    ).mean()

def _ref_update_and_collect_params(ref, batch: TorchMiniBatch, step: int, our: CQLAgentContinuous):
    _ = ref.impl.inner_update(batch, step)
    return [x for _, x in _all_pairs(our, ref)]

# ====================
# Our Math
# ====================

def _our_update_and_collect_params(
    our: CQLAgentContinuous,
    obs_t: torch.Tensor,
    act_t: torch.Tensor,
    rew_t: torch.Tensor,
    next_obs_t: torch.Tensor,
    done_t: torch.Tensor,
    ref,
):
    our.update({"obs": obs_t, "act": act_t, "rew": rew_t, "next_obs": next_obs_t, "done": done_t})
    return [y for y, _ in _all_pairs(our, ref)]


# ====================
# Compare
# ====================
def init_compare() -> tuple[CQLAgentContinuous, object]:
    print_stage("Init")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    our = build_our_agent()
    ref = build_d3rl()
    with torch.no_grad():
        for our_param, d3_param in _all_pairs(our, ref):
            our_param.copy_(d3_param)
    return our, ref


def compare_act(our: CQLAgentContinuous, ref) -> None:
    print_stage("Act Compare")
    rng = np.random.default_rng(SEED)

    for i in range(1, N_TEST_BATCHES + 1):
        o1, _, _, _, _, _ = sample_transition(rng, 1, OBS_DIM, ACT_DIM)
        o, _, _, _, _, _ = sample_transition(rng, BATCH_SIZE, OBS_DIM, ACT_DIM)

        assert_callback(
            lambda: [d3rl_action_best_single(ref, o1)],
            lambda: [our.act(o1[0], greedy=True)],
            label=f"act_best_single[{i}]",
            seed=SEED + 1000 + i,
        )
        assert_callback(
            lambda: [d3rl_action_best_batch(ref, o)],
            lambda: [our.act_batch(o.cpu().numpy(), greedy=True)],
            label=f"act_best_batch[{i}]",
            seed=SEED + 1001 + i,
        )

        assert_callback(
            lambda: [d3rl_action_sample_single(ref, o1)],
            lambda: [our.act(o1[0], greedy=False)],
            label=f"act_sample_single[{i}]",
            seed=SEED + 1002 + i,
        )

        assert_callback(
            lambda: [d3rl_action_sample_batch(ref, o)],
            lambda: [our.act_batch(o.cpu().numpy(), greedy=False)],
            label=f"act_sample_batch[{i}]",
            seed=SEED + 1003 + i,
        )

        print(f"batch={i}/{N_TEST_BATCHES} act_and_sample_match=True")


def compare_loss(our: CQLAgentContinuous, ref) -> None:
    print_stage("Loss Compare")
    rng = np.random.default_rng(SEED + 1)

    for i in range(1, N_TEST_BATCHES + 1):
        obs_t, act_t, rew_t, next_obs_t, done_t, batch = sample_transition(
            rng, BATCH_SIZE, OBS_DIM, ACT_DIM
        )

        assert_callback(
            lambda: [_d3rl_loss_td(ref, batch)],
            lambda: [our.loss_td(obs_t, act_t, rew_t, next_obs_t, done_t)],
            label=f"loss_td[{i}]",
            seed=SEED + 2000 + i * 100 + 1,
        )
        assert_callback(
            lambda: [_d3rl_loss_conservative(ref, batch)],
            lambda: [our.loss_conservative(obs_t, act_t, next_obs_t)],
            label=f"loss_conservative[{i}]",
            seed=SEED + 2000 + i * 100 + 2,
        )
        assert_callback(
            lambda: [_d3rl_loss_critic(ref, batch)],
            lambda: [our.loss_critic(obs_t, act_t, rew_t, next_obs_t, done_t, update_alpha=True)],
            label=f"loss_critic[{i}]",
            seed=SEED + 2000 + i * 100 + 3,
        )
        assert_callback(
            lambda: [_d3rl_loss_actor(ref, obs_t)],
            lambda: [our.loss_actor(obs_t, update_alpha=False)],
            label=f"loss_actor[{i}]",
            seed=SEED + 2000 + i * 100 + 4,
        )
        assert_callback(
            lambda: [_d3rl_loss_alpha_cql(ref, batch)],
            lambda: [our.loss_alpha_cql((5.0 * (our.loss_conservative(obs_t, act_t, next_obs_t) - our.critic.alpha_threshold)).detach())],
            label=f"loss_alpha_cql[{i}]",
            seed=SEED + 2000 + i * 100 + 5,
        )
        assert_callback(
            lambda: [_d3rl_loss_alpha_sac(ref, obs_t)],
            lambda: [our.loss_alpha_sac(our.policy.sample(obs_t)[1].detach())],
            label=f"loss_alpha_sac[{i}]",
            seed=SEED + 2000 + i * 100 + 6,
        )

        print(f"batch={i}/{N_TEST_BATCHES} loss_match=True")


def compare_param(our: CQLAgentContinuous, ref) -> None:
    print_stage("Param Compare")
    rng = np.random.default_rng(SEED + 2)

    for i in range(1, N_TEST_BATCHES + 1):
        obs_t, act_t, rew_t, next_obs_t, done_t, batch = sample_transition(
            rng, BATCH_SIZE, OBS_DIM, ACT_DIM
        )

        assert_callback(
            lambda: _ref_update_and_collect_params(ref, batch, i, our),
            lambda: _our_update_and_collect_params(our, obs_t, act_t, rew_t, next_obs_t, done_t, ref),
            label=f"update[{i}]",
            seed=SEED + 3000 + i,
        )

        print(f"batch={i}/{N_TEST_BATCHES} param_match=True")


# ====================
# __main__
# ====================
if __name__ == "__main__":
    our, ref = init_compare()
    compare_act(our, ref)
    compare_loss(our, ref)
    compare_param(our, ref)

    print_stage("Result")
    print("PASS: sample, act, act_batch, loss, and full update params are aligned with d3rl.")
