from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from _lib import assert_callback
from _lib import sample_transition
from _lib import torch_buffer
from SCAS_main import SCAS as ref_scas
from SCAS_main import model as ref_model
from ice_offline.agent.scas_min import ScasMinAgent
from ice_offline.agent.scas_min import ScasDynamicAgent
from ice_offline.tools.printer import print_stage


# ====================
# Config
# ====================
obs_size = 8
act_size = 3
DEVICE = "cpu"
SEED = 42
BATCH_SIZE = 64
N_TEST_BATCHES = 30
MAX_ACTION = 1.0


# ====================
# Mapping
# ====================
def _all_pairs(
    ref: Any,
    our: ScasMinAgent,
    ref_dynamics: torch.nn.Module,
    our_dynamics: ScasDynamicAgent,
):
    return [
        # dynamics M
        (our_dynamics.model.network[0].weight, ref_dynamics.l1.weight),
        (our_dynamics.model.network[0].bias, ref_dynamics.l1.bias),
        (our_dynamics.model.network[2].weight, ref_dynamics.l2.weight),
        (our_dynamics.model.network[2].bias, ref_dynamics.l2.bias),
        (our_dynamics.model.network[4].weight, ref_dynamics.l3.weight),
        (our_dynamics.model.network[4].bias, ref_dynamics.l3.bias),
        (our_dynamics.model.network[6].weight, ref_dynamics.l4.weight),
        (our_dynamics.model.network[6].bias, ref_dynamics.l4.bias),
        (our_dynamics.model.network[8].weight, ref_dynamics.l5.weight),
        (our_dynamics.model.network[8].bias, ref_dynamics.l5.bias),
        # actor pi
        (our.actor.pi.network[0].weight, ref.actor.l1.weight),
        (our.actor.pi.network[0].bias, ref.actor.l1.bias),
        (our.actor.pi.network[2].weight, ref.actor.l2.weight),
        (our.actor.pi.network[2].bias, ref.actor.l2.bias),
        (our.actor.pi.network[4].weight, ref.actor.l3.weight),
        (our.actor.pi.network[4].bias, ref.actor.l3.bias),
        # actor target
        (our.actor.tpi.network[0].weight, ref.actor_target.l1.weight),
        (our.actor.tpi.network[0].bias, ref.actor_target.l1.bias),
        (our.actor.tpi.network[2].weight, ref.actor_target.l2.weight),
        (our.actor.tpi.network[2].bias, ref.actor_target.l2.bias),
        (our.actor.tpi.network[4].weight, ref.actor_target.l3.weight),
        (our.actor.tpi.network[4].bias, ref.actor_target.l3.bias),
        # critic q1
        (our.critic.q_networks[0].network[0].weight, ref.critic.l1.weight),
        (our.critic.q_networks[0].network[0].bias, ref.critic.l1.bias),
        (our.critic.q_networks[0].network[2].weight, ref.critic.l2.weight),
        (our.critic.q_networks[0].network[2].bias, ref.critic.l2.bias),
        (our.critic.q_networks[0].network[4].weight, ref.critic.l3.weight),
        (our.critic.q_networks[0].network[4].bias, ref.critic.l3.bias),
        # critic q2
        (our.critic.q_networks[1].network[0].weight, ref.critic.l4.weight),
        (our.critic.q_networks[1].network[0].bias, ref.critic.l4.bias),
        (our.critic.q_networks[1].network[2].weight, ref.critic.l5.weight),
        (our.critic.q_networks[1].network[2].bias, ref.critic.l5.bias),
        (our.critic.q_networks[1].network[4].weight, ref.critic.l6.weight),
        (our.critic.q_networks[1].network[4].bias, ref.critic.l6.bias),
        # critic q3
        (our.critic.q_networks[2].network[0].weight, ref.critic.l7.weight),
        (our.critic.q_networks[2].network[0].bias, ref.critic.l7.bias),
        (our.critic.q_networks[2].network[2].weight, ref.critic.l8.weight),
        (our.critic.q_networks[2].network[2].bias, ref.critic.l8.bias),
        (our.critic.q_networks[2].network[4].weight, ref.critic.l9.weight),
        (our.critic.q_networks[2].network[4].bias, ref.critic.l9.bias),
        # critic q4
        (our.critic.q_networks[3].network[0].weight, ref.critic.l10.weight),
        (our.critic.q_networks[3].network[0].bias, ref.critic.l10.bias),
        (our.critic.q_networks[3].network[2].weight, ref.critic.l11.weight),
        (our.critic.q_networks[3].network[2].bias, ref.critic.l11.bias),
        (our.critic.q_networks[3].network[4].weight, ref.critic.l12.weight),
        (our.critic.q_networks[3].network[4].bias, ref.critic.l12.bias),
        # critic target q1
        (our.critic.tq_networks[0].network[0].weight, ref.critic_target.l1.weight),
        (our.critic.tq_networks[0].network[0].bias, ref.critic_target.l1.bias),
        (our.critic.tq_networks[0].network[2].weight, ref.critic_target.l2.weight),
        (our.critic.tq_networks[0].network[2].bias, ref.critic_target.l2.bias),
        (our.critic.tq_networks[0].network[4].weight, ref.critic_target.l3.weight),
        (our.critic.tq_networks[0].network[4].bias, ref.critic_target.l3.bias),
        # critic target q2
        (our.critic.tq_networks[1].network[0].weight, ref.critic_target.l4.weight),
        (our.critic.tq_networks[1].network[0].bias, ref.critic_target.l4.bias),
        (our.critic.tq_networks[1].network[2].weight, ref.critic_target.l5.weight),
        (our.critic.tq_networks[1].network[2].bias, ref.critic_target.l5.bias),
        (our.critic.tq_networks[1].network[4].weight, ref.critic_target.l6.weight),
        (our.critic.tq_networks[1].network[4].bias, ref.critic_target.l6.bias),
        # critic target q3
        (our.critic.tq_networks[2].network[0].weight, ref.critic_target.l7.weight),
        (our.critic.tq_networks[2].network[0].bias, ref.critic_target.l7.bias),
        (our.critic.tq_networks[2].network[2].weight, ref.critic_target.l8.weight),
        (our.critic.tq_networks[2].network[2].bias, ref.critic_target.l8.bias),
        (our.critic.tq_networks[2].network[4].weight, ref.critic_target.l9.weight),
        (our.critic.tq_networks[2].network[4].bias, ref.critic_target.l9.bias),
        # critic target q4
        (our.critic.tq_networks[3].network[0].weight, ref.critic_target.l10.weight),
        (our.critic.tq_networks[3].network[0].bias, ref.critic_target.l10.bias),
        (our.critic.tq_networks[3].network[2].weight, ref.critic_target.l11.weight),
        (our.critic.tq_networks[3].network[2].bias, ref.critic_target.l11.bias),
        (our.critic.tq_networks[3].network[4].weight, ref.critic_target.l12.weight),
        (our.critic.tq_networks[3].network[4].bias, ref.critic_target.l12.bias),
    ]


# ====================
# Ref define
# ====================
def ref_loss_critic(
    ref: Any,
    s: torch.Tensor,
    a: torch.Tensor,
    r: torch.Tensor,
    sn: torch.Tensor,
    d: torch.Tensor,
) -> torch.Tensor:
    not_done = 1.0 - d
    with torch.no_grad():
        noise = (torch.randn_like(a) * 0.2).clamp(-0.5, 0.5)
        next_action = (ref.actor_target(sn) + noise).clamp(-ref.max_action, ref.max_action)
        target_q1, target_q2, target_q3, target_q4 = ref.critic_target(sn, next_action)
        target_q = torch.cat([target_q1, target_q2, target_q3, target_q4], dim=1)
        target_q, _ = torch.min(target_q, dim=1, keepdim=True)
        y = r + not_done * ref.discount * target_q
    q1, q2, q3, q4 = ref.critic(s, a)
    loss_q1 = F.mse_loss(q1, y)
    loss_q2 = F.mse_loss(q2, y)
    loss_q3 = F.mse_loss(q3, y)
    loss_q4 = F.mse_loss(q4, y)
    return loss_q1 + loss_q2 + loss_q3 + loss_q4

def ref_loss_actor_td3(ref: Any, s: torch.Tensor, sn: torch.Tensor) -> torch.Tensor:
    pi = ref.actor(s)
    v1, v2, v3, v4 = ref.critic(s, pi)
    v = torch.cat([v1, v2, v3, v4], dim=1)
    v_min, _ = torch.min(v, dim=1)
    lmbda = 1.0 / v_min.abs().mean().detach()
    maxq_loss = -lmbda * v_min.mean()
    return maxq_loss

def ref_actor_weight_and_state_hat(
    ref: Any,
    s: torch.Tensor,
    sn: torch.Tensor,
    v1: torch.Tensor,
    v2: torch.Tensor,
    v3: torch.Tensor,
    v4: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        v_mean = (v1 + v2 + v3 + v4) / 4
        next_pi = ref.actor(sn)
        nv1, nv2, nv3, nv4 = ref.critic(sn, next_pi)
        next_v_mean = (nv1 + nv2 + nv3 + nv4) / 4
        weight = (ref.temp * (next_v_mean.detach() - v_mean.detach())).exp().clamp(max=ref.max_weight)
        state_hat = s + torch.randn(s.shape, device=DEVICE) * ref.beta
    return weight, state_hat

def ref_loss_actor_correction(ref: Any, s: torch.Tensor, sn: torch.Tensor) -> torch.Tensor:
    pi = ref.actor(s)
    v1, v2, v3, v4 = ref.critic(s, pi)
    weight, state_hat = ref_actor_weight_and_state_hat(ref, s, sn, v1, v2, v3, v4)
    pred_next_state = ref.dynamics(state_hat, pi)
    state_recovery_loss = (weight * (pred_next_state - sn) ** 2).mean()
    return state_recovery_loss

def ref_loss_actor_total(ref: Any, s: torch.Tensor, sn: torch.Tensor) -> torch.Tensor:
    return (1 - ref.lam) * ref_loss_actor_td3(ref, s, sn) + ref.lam * ref_loss_actor_correction(ref, s, sn)

def ref_loss_dynamic(ref_dynamics: torch.nn.Module, s: torch.Tensor, a: torch.Tensor, sn: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(ref_dynamics(s, a), sn)

def ref_update(ref: Any, s: torch.Tensor, a: torch.Tensor, r: torch.Tensor, sn: torch.Tensor, d: torch.Tensor) -> None:
    ref.total_it += 1
    critic_loss = ref_loss_critic(ref, s, a, r, sn, d)
    ref.critic_optimizer.zero_grad()
    critic_loss.backward()
    ref.critic_optimizer.step()
    if ref.total_it % ref.policy_freq == 0:
        actor_loss = ref_loss_actor_total(ref, s, sn)
        ref.actor_optimizer.zero_grad()
        actor_loss.backward()
        ref.actor_optimizer.step()
        if ref.schedule:
            ref.actor_lr_schedule.step()
        for param, target_param in zip(ref.critic.parameters(), ref.critic_target.parameters()):
            target_param.data.copy_(ref.tau * param.data + (1 - ref.tau) * target_param.data)
        for param, target_param in zip(ref.actor.parameters(), ref.actor_target.parameters()):
            target_param.data.copy_(ref.tau * param.data + (1 - ref.tau) * target_param.data)

def ref_update_and_collect_params(
    ref: Any,
    our: ScasMinAgent,
    ref_dynamics: torch.nn.Module,
    our_dynamics: ScasDynamicAgent,
    s: torch.Tensor,
    a: torch.Tensor,
    r: torch.Tensor,
    sn: torch.Tensor,
    d: torch.Tensor,
) -> list[torch.Tensor]:
    ref_update(ref, s, a, r, sn, d)
    return [ref_param.detach().cpu().clone() for _, ref_param in _all_pairs(ref, our, ref_dynamics, our_dynamics)]


# ====================
# Our define
# ====================
def our_update_and_collect_params(
    ref: Any,
    our: ScasMinAgent,
    ref_dynamics: torch.nn.Module,
    our_dynamics: ScasDynamicAgent,
    s: torch.Tensor,
    a: torch.Tensor,
    r: torch.Tensor,
    sn: torch.Tensor,
    d: torch.Tensor,
) -> list[torch.Tensor]:
    our.update(torch_buffer(s, a, r, sn, d))
    return [our_param.detach().cpu().clone() for our_param, _ in _all_pairs(ref, our, ref_dynamics, our_dynamics)]


# ====================
# Compare
# ====================
def build_our() -> tuple[ScasMinAgent, ScasDynamicAgent]:
    our_dynamics = ScasDynamicAgent(obs_size=obs_size, act_size=act_size, device=DEVICE)
    our = ScasMinAgent(
        obs_size=obs_size,
        act_size=act_size,
        dynamics=our_dynamics,
        max_action=MAX_ACTION,
        device=DEVICE,
    )
    return our, our_dynamics

def build_ref() -> tuple[Any, torch.nn.Module]:
    ref_model.device = torch.device(DEVICE)
    ref_scas.device = torch.device(DEVICE)
    ref_dynamics = ref_model.Dynamics(obs_size, act_size).to(DEVICE)
    ref = ref_scas.SCAS(
        state_dim=obs_size,
        action_dim=act_size,
        max_action=MAX_ACTION,
        replay_buffer=None,
        dynamics=ref_dynamics,
        antmaze=False,
    )
    return ref, ref_dynamics

def init_compare() -> tuple[Any, ScasMinAgent, torch.nn.Module, ScasDynamicAgent]:
    print_stage("Init")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    ref, ref_dynamics = build_ref()
    our, our_dynamics = build_our()
    with torch.no_grad():
        for our_param, ref_param in _all_pairs(ref, our, ref_dynamics, our_dynamics):
            our_param.copy_(ref_param)
    return ref, our, ref_dynamics, our_dynamics

def compare_act(ref: Any, our: ScasMinAgent) -> None:
    print_stage("Act Compare")
    for i in range(1, N_TEST_BATCHES + 1):
        s, _, _, _, _ = sample_transition(BATCH_SIZE, obs_size, act_size, DEVICE)

        # act best
        assert_callback(
            lambda: [ref.select_action(s[0].cpu().numpy())],
            lambda: [our.act(s[0].cpu().numpy())],
            label=f"act_best[{i}]",
            seed=SEED + i,
        )

        # act best batch
        assert_callback(
            lambda: [ref.actor(s).detach().cpu().numpy()],
            lambda: [our.act_batch(s.cpu().numpy())],
            label=f"act_best_batch[{i}]",
            seed=SEED + i,
        )

        print(f"batch={i}/{N_TEST_BATCHES} act_match=True")

def compare_loss(
    ref: Any,
    our: ScasMinAgent,
    ref_dynamics: torch.nn.Module,
    our_dynamics: ScasDynamicAgent,
) -> None:
    print_stage("Loss Compare")
    for i in range(1, N_TEST_BATCHES + 1):
        s, a, r, sn, d = sample_transition(BATCH_SIZE, obs_size, act_size, DEVICE)

        # loss dynamic
        assert_callback(
            lambda: [ref_loss_dynamic(ref_dynamics, s, a, sn)],
            lambda: [our_dynamics.loss_dynamic(s, a, sn)],
            label=f"loss_dynamic[{i}]",
            seed=SEED + i,
        )

        # loss td3
        assert_callback(
            lambda: [ref_loss_actor_td3(ref, s, sn)],
            lambda: [our.loss_td3(s)],
            label=f"loss_td3[{i}]",
            seed=SEED + i,
        )

        # loss correction
        assert_callback(
            lambda: [ref_loss_actor_correction(ref, s, sn)],
            lambda: [our.loss_correction(s, sn)],
            label=f"loss_correction[{i}]",
            seed=SEED + i,
        )

        # loss actor
        assert_callback(
            lambda: [ref_loss_actor_total(ref, s, sn)],
            lambda: [our.loss_actor(s, sn)],
            label=f"loss_actor[{i}]",
            seed=SEED + i,
        )

        # loss critic
        assert_callback(
            lambda: [ref_loss_critic(ref, s, a, r, sn, d)],
            lambda: [our.loss_critic(s, a, r, sn, d)],
            label=f"loss_critic[{i}]",
            seed=SEED + i,
        )

        print(f"batch={i}/{N_TEST_BATCHES} loss_match=True")

def compare_param(
    ref: Any,
    our: ScasMinAgent,
    ref_dynamics: torch.nn.Module,
    our_dynamics: ScasDynamicAgent,
) -> None:
    print_stage("Update Compare")
    for i in range(1, N_TEST_BATCHES + 1):
        s, a, r, sn, d = sample_transition(1, obs_size, act_size, DEVICE)

        # update params
        assert_callback(
            lambda: ref_update_and_collect_params(ref, our, ref_dynamics, our_dynamics, s, a, r, sn, d),
            lambda: our_update_and_collect_params(ref, our, ref_dynamics, our_dynamics, s, a, r, sn, d),
            label=f"update_single[{i}]",
            seed=SEED + i,
        )

        print(f"batch={i}/{N_TEST_BATCHES} single_param_match=True")

    for i in range(1, N_TEST_BATCHES + 1):
        s, a, r, sn, d = sample_transition(BATCH_SIZE, obs_size, act_size, DEVICE)

        # update params
        assert_callback(
            lambda: ref_update_and_collect_params(ref, our, ref_dynamics, our_dynamics, s, a, r, sn, d),
            lambda: our_update_and_collect_params(ref, our, ref_dynamics, our_dynamics, s, a, r, sn, d),
            label=f"update_batch[{i}]",
            seed=SEED + i,
        )

        print(f"batch={i}/{N_TEST_BATCHES} batch_param_match=True")


# ====================
# Main
# ====================
if __name__ == "__main__":
    ref, our, ref_dynamics, our_dynamics = init_compare()
    compare_act(ref, our)
    compare_loss(ref, our, ref_dynamics, our_dynamics)
    compare_param(ref, our, ref_dynamics, our_dynamics)
    print_stage("Result")
    print("PASS: compare_scas finished.")

