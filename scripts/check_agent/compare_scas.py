from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from SCAS_main import SCAS as ref_scas
from SCAS_main import model as ref_model
from ice_offline.agent.scas import ScasAgent
from ice_offline.agent.scas import ScasDynamic
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
MAX_ACTION = 1.0
REF: Any | None = None
OUR: ScasAgent | None = None
OUR_DYNAMICS: ScasDynamic | None = None
REF_DYNAMICS: torch.nn.Module | None = None
# ====================
# Mapping: all_pairs
# ====================

def _all_pairs(ref: Any, our: ScasAgent, ref_dynamics: torch.nn.Module, our_dynamics: ScasDynamic):
    return [
        # dynamics M: l1..l5
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
        # actor pi: l1..l3
        (our.actor.pi.network[0].weight, ref.actor.l1.weight),
        (our.actor.pi.network[0].bias, ref.actor.l1.bias),
        (our.actor.pi.network[2].weight, ref.actor.l2.weight),
        (our.actor.pi.network[2].bias, ref.actor.l2.bias),
        (our.actor.pi.network[4].weight, ref.actor.l3.weight),
        (our.actor.pi.network[4].bias, ref.actor.l3.bias),
        # actor target: l1..l3
        (our.actor.tpi.network[0].weight, ref.actor_target.l1.weight),
        (our.actor.tpi.network[0].bias, ref.actor_target.l1.bias),
        (our.actor.tpi.network[2].weight, ref.actor_target.l2.weight),
        (our.actor.tpi.network[2].bias, ref.actor_target.l2.bias),
        (our.actor.tpi.network[4].weight, ref.actor_target.l3.weight),
        (our.actor.tpi.network[4].bias, ref.actor_target.l3.bias),
        # critic q1: l1..l3
        (our.critic.q1.network[0].weight, ref.critic.l1.weight),
        (our.critic.q1.network[0].bias, ref.critic.l1.bias),
        (our.critic.q1.network[2].weight, ref.critic.l2.weight),
        (our.critic.q1.network[2].bias, ref.critic.l2.bias),
        (our.critic.q1.network[4].weight, ref.critic.l3.weight),
        (our.critic.q1.network[4].bias, ref.critic.l3.bias),
        # critic q2: l4..l6
        (our.critic.q2.network[0].weight, ref.critic.l4.weight),
        (our.critic.q2.network[0].bias, ref.critic.l4.bias),
        (our.critic.q2.network[2].weight, ref.critic.l5.weight),
        (our.critic.q2.network[2].bias, ref.critic.l5.bias),
        (our.critic.q2.network[4].weight, ref.critic.l6.weight),
        (our.critic.q2.network[4].bias, ref.critic.l6.bias),
        # critic q3: l7..l9
        (our.critic.q3.network[0].weight, ref.critic.l7.weight),
        (our.critic.q3.network[0].bias, ref.critic.l7.bias),
        (our.critic.q3.network[2].weight, ref.critic.l8.weight),
        (our.critic.q3.network[2].bias, ref.critic.l8.bias),
        (our.critic.q3.network[4].weight, ref.critic.l9.weight),
        (our.critic.q3.network[4].bias, ref.critic.l9.bias),
        # critic q4: l10..l12
        (our.critic.q4.network[0].weight, ref.critic.l10.weight),
        (our.critic.q4.network[0].bias, ref.critic.l10.bias),
        (our.critic.q4.network[2].weight, ref.critic.l11.weight),
        (our.critic.q4.network[2].bias, ref.critic.l11.bias),
        (our.critic.q4.network[4].weight, ref.critic.l12.weight),
        (our.critic.q4.network[4].bias, ref.critic.l12.bias),
        # critic target q1: l1..l3
        (our.critic.tq1.network[0].weight, ref.critic_target.l1.weight),
        (our.critic.tq1.network[0].bias, ref.critic_target.l1.bias),
        (our.critic.tq1.network[2].weight, ref.critic_target.l2.weight),
        (our.critic.tq1.network[2].bias, ref.critic_target.l2.bias),
        (our.critic.tq1.network[4].weight, ref.critic_target.l3.weight),
        (our.critic.tq1.network[4].bias, ref.critic_target.l3.bias),
        # critic target q2: l4..l6
        (our.critic.tq2.network[0].weight, ref.critic_target.l4.weight),
        (our.critic.tq2.network[0].bias, ref.critic_target.l4.bias),
        (our.critic.tq2.network[2].weight, ref.critic_target.l5.weight),
        (our.critic.tq2.network[2].bias, ref.critic_target.l5.bias),
        (our.critic.tq2.network[4].weight, ref.critic_target.l6.weight),
        (our.critic.tq2.network[4].bias, ref.critic_target.l6.bias),
        # critic target q3: l7..l9
        (our.critic.tq3.network[0].weight, ref.critic_target.l7.weight),
        (our.critic.tq3.network[0].bias, ref.critic_target.l7.bias),
        (our.critic.tq3.network[2].weight, ref.critic_target.l8.weight),
        (our.critic.tq3.network[2].bias, ref.critic_target.l8.bias),
        (our.critic.tq3.network[4].weight, ref.critic_target.l9.weight),
        (our.critic.tq3.network[4].bias, ref.critic_target.l9.bias),
        # critic target q4: l10..l12
        (our.critic.tq4.network[0].weight, ref.critic_target.l10.weight),
        (our.critic.tq4.network[0].bias, ref.critic_target.l10.bias),
        (our.critic.tq4.network[2].weight, ref.critic_target.l11.weight),
        (our.critic.tq4.network[2].bias, ref.critic_target.l11.bias),
        (our.critic.tq4.network[4].weight, ref.critic_target.l12.weight),
        (our.critic.tq4.network[4].bias, ref.critic_target.l12.bias),
    ]
# ====================
# common
# ====================

def init_compare() -> tuple[Any, ScasAgent, torch.nn.Module, ScasDynamic]:
    global REF, OUR, OUR_DYNAMICS, REF_DYNAMICS
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    ref_model.device = torch.device(DEVICE)
    ref_scas.device = torch.device(DEVICE)
    RefDynamics = ref_model.Dynamics
    RefSCAS = ref_scas.SCAS
    REF_DYNAMICS = RefDynamics(OBS_DIM, ACT_DIM).to(DEVICE)
    REF = RefSCAS(
        state_dim=OBS_DIM,
        action_dim=ACT_DIM,
        max_action=MAX_ACTION,
        replay_buffer=None,
        dynamics=REF_DYNAMICS,
        antmaze=False,
    )
    OUR_DYNAMICS = ScasDynamic(obs_dim=OBS_DIM, act_dim=ACT_DIM, device=DEVICE)
    OUR = ScasAgent(
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        dynamics=OUR_DYNAMICS,
        max_action=MAX_ACTION,
        device=DEVICE,
    )
    with torch.no_grad():
        for our_param, ref_param in _all_pairs(REF, OUR, REF_DYNAMICS, OUR_DYNAMICS):
            our_param.copy_(ref_param)
    return REF, OUR, REF_DYNAMICS, OUR_DYNAMICS

def _sample_transition(batch_size: int = BATCH_SIZE):
    s = torch.as_tensor(np.random.standard_normal((batch_size, OBS_DIM)), dtype=torch.float32, device=DEVICE)
    a = torch.as_tensor(np.random.standard_normal((batch_size, ACT_DIM)), dtype=torch.float32, device=DEVICE)
    r = torch.as_tensor(np.random.standard_normal((batch_size, 1)), dtype=torch.float32, device=DEVICE)
    sn = torch.as_tensor(np.random.standard_normal((batch_size, OBS_DIM)), dtype=torch.float32, device=DEVICE)
    d = torch.as_tensor(np.random.randint(0, 2, size=(batch_size, 1)), dtype=torch.float32, device=DEVICE)
    return s, a, r, sn, d
# ====================
# Ref Math
# ====================

def _ref_loss_critic_parts(
    ref: Any, s: torch.Tensor, a: torch.Tensor, r: torch.Tensor, sn: torch.Tensor, d: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    loss_q = loss_q1 + loss_q2 + loss_q3 + loss_q4
    return loss_q1, loss_q2, loss_q3, loss_q4, loss_q

def _ref_loss_actor_td3(ref: Any, s: torch.Tensor, sn: torch.Tensor) -> torch.Tensor:
    pi = ref.actor(s)
    v1, v2, v3, v4 = ref.critic(s, pi)
    v = torch.cat([v1, v2, v3, v4], dim=1)
    v_min, _ = torch.min(v, dim=1)
    lmbda = 1.0 / v_min.abs().mean().detach()
    maxq_loss = -lmbda * v_min.mean()
    return maxq_loss

def _ref_actor_weight_and_state_hat(
    ref: Any, s: torch.Tensor, sn: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor, v3: torch.Tensor, v4: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # SCAS.py train(): weight + state_hat
    with torch.no_grad():
        v_mean = (v1 + v2 + v3 + v4) / 4
        next_pi = ref.actor(sn)
        nv1, nv2, nv3, nv4 = ref.critic(sn, next_pi)
        next_v_mean = (nv1 + nv2 + nv3 + nv4) / 4
        weight = (ref.temp * (next_v_mean.detach() - v_mean.detach())).exp().clamp(max=ref.max_weight)
        state_hat = s + torch.randn(s.shape, device=DEVICE) * ref.beta
    return weight, state_hat

def _ref_loss_actor_correction(ref: Any, s: torch.Tensor, sn: torch.Tensor) -> torch.Tensor:
    pi = ref.actor(s)
    v1, v2, v3, v4 = ref.critic(s, pi)
    weight, state_hat = _ref_actor_weight_and_state_hat(ref, s, sn, v1, v2, v3, v4)
    pred_next_state = ref.dynamics(state_hat, pi)
    state_recovery_loss = (weight * (pred_next_state - sn) ** 2).mean()
    return state_recovery_loss

def _ref_loss_actor_total(ref: Any, s: torch.Tensor, sn: torch.Tensor) -> torch.Tensor:
    return (1 - ref.lam) * _ref_loss_actor_td3(ref, s, sn) + ref.lam * _ref_loss_actor_correction(ref, s, sn)

def _ref_loss_dynamic(ref_dynamics: torch.nn.Module, s: torch.Tensor, a: torch.Tensor, sn: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(ref_dynamics(s, a), sn)

def _ref_loss_critic_total(ref: Any, s: torch.Tensor, a: torch.Tensor, r: torch.Tensor, sn: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    return _ref_loss_critic_parts(ref, s, a, r, sn, d)[4]

def _update_ref(ref: Any, s: torch.Tensor, a: torch.Tensor, r: torch.Tensor, sn: torch.Tensor, d: torch.Tensor) -> None:
    ref.total_it += 1  # SCAS.py:121
    critic_loss = _ref_loss_critic_total(ref, s, a, r, sn, d)  # SCAS.py:139
    ref.critic_optimizer.zero_grad()  # SCAS.py:147
    critic_loss.backward()  # SCAS.py:148
    ref.critic_optimizer.step()  # SCAS.py:149
    if ref.total_it % ref.policy_freq == 0:  # SCAS.py:152
        actor_loss = _ref_loss_actor_total(ref, s, sn)  # SCAS.py:172
        ref.actor_optimizer.zero_grad()  # SCAS.py:174
        actor_loss.backward()  # SCAS.py:175
        ref.actor_optimizer.step()  # SCAS.py:176
        if ref.schedule:
            ref.actor_lr_schedule.step()  # SCAS.py:177-178
        for param, target_param in zip(ref.critic.parameters(), ref.critic_target.parameters()):
            target_param.data.copy_(ref.tau * param.data + (1 - ref.tau) * target_param.data)  # SCAS.py:185-186
        for param, target_param in zip(ref.actor.parameters(), ref.actor_target.parameters()):
            target_param.data.copy_(ref.tau * param.data + (1 - ref.tau) * target_param.data)  # SCAS.py:188-189

def _ref_update_and_collect_params(
    s: torch.Tensor, a: torch.Tensor, r: torch.Tensor, sn: torch.Tensor, d: torch.Tensor
) -> list[torch.Tensor]:
    _update_ref(REF, s, a, r, sn, d)
    return [ref_param.detach().cpu().clone() for _, ref_param in _all_pairs(REF, OUR, REF_DYNAMICS, OUR_DYNAMICS)]


# ====================
# Our Math
# ====================
def _our_update_and_collect_params(
    s: torch.Tensor, a: torch.Tensor, r: torch.Tensor, sn: torch.Tensor, d: torch.Tensor
) -> list[torch.Tensor]:
    OUR.update({"obs": s, "act": a, "rew": r, "next_obs": sn, "done": d})
    return [our_param.detach().cpu().clone() for our_param, _ in _all_pairs(REF, OUR, REF_DYNAMICS, OUR_DYNAMICS)]

def compare_act() -> None:
    print_stage("Act Compare")
    for i in range(1, N_TEST_BATCHES + 1):
        s, _, _, _, _ = _sample_transition()
        assert_callback(
            lambda: [REF.select_action(s[0].cpu().numpy())],
            lambda: [OUR.act(s[0].cpu().numpy(), greedy=True)],
            label=f"act_single_{i}",
            seed=SEED + i,
        )
        assert_callback(
            lambda: [REF.actor(s).detach().cpu().numpy()],
            lambda: [OUR.act_batch(s.cpu().numpy(), greedy=True)],
            label=f"act_batch_{i}",
            seed=SEED + 100 + i,
        )

def compare_loss() -> None:
    print_stage("Loss Compare")
    for i in range(1, N_TEST_BATCHES + 1):
        s, a, r, sn, d = _sample_transition()
        assert_callback(
            lambda: [_ref_loss_dynamic(REF_DYNAMICS, s, a, sn)],
            lambda: [OUR_DYNAMICS.loss_dynamic(s, a, sn)],
            label=f"loss_dynamic_m_{i}",
            seed=SEED + 900 + i,
        )
        assert_callback(
            lambda: [_ref_loss_critic_total(REF, s, a, r, sn, d)],
            lambda: [OUR.loss_critic(s, a, r, sn, d)],
            label=f"loss_critic_total_{i}",
            seed=SEED + 1000 + i,
        )
        assert_callback(
            lambda: [_ref_loss_actor_td3(REF, s, sn)],
            lambda: [OUR.loss_td3(s)],
            label=f"loss_actor_td3_{i}",
            seed=SEED + 2000 + i,
        )
        assert_callback(
            lambda: [_ref_loss_actor_correction(REF, s, sn)],
            lambda: [OUR.loss_correction(s, sn)],
            label=f"loss_actor_correction_{i}",
            seed=SEED + 2100 + i,
        )
        assert_callback(
            lambda: [_ref_loss_actor_total(REF, s, sn)],
            lambda: [OUR.loss_actor(s, sn)],
            label=f"loss_actor_total_{i}",
            seed=SEED + 2200 + i,
        )

def compare_param() -> None:
    print_stage("Param Compare")
    for i in range(1, N_TEST_BATCHES + 1):
        s, a, r, sn, d = _sample_transition(batch_size=1)
        assert_callback(
            lambda: _ref_update_and_collect_params(s, a, r, sn, d),
            lambda: _our_update_and_collect_params(s, a, r, sn, d),
            label=f"param_single_{i}",
            seed=SEED + 3000 + i,
        )
    for i in range(1, N_TEST_BATCHES + 1):
        s, a, r, sn, d = _sample_transition(batch_size=BATCH_SIZE)
        assert_callback(
            lambda: _ref_update_and_collect_params(s, a, r, sn, d),
            lambda: _our_update_and_collect_params(s, a, r, sn, d),
            label=f"param_batch_{i}",
            seed=SEED + 4000 + i,
        )
# ====================
# Compare
# ====================

def main() -> None:
    print_stage("Init")
    init_compare()
    compare_act()
    compare_loss()
    compare_param()
    print_stage("Result")
    print("PASS: compare_scas finished.")
# ====================
# __main__
# ====================

if __name__ == "__main__":
    main()
