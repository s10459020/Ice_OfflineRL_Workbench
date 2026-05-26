import sys
import types

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import qmc

# Dummy library
sys.modules["d4rl"] = types.ModuleType("d4rl")
from ice_offline.agent.aspl import AsplAgent
from ice_offline.tools.printer import print_stage
import _lib
from _lib import assert_callback, assert_list

from ASPL_source.agent.policy.aspl_policy import ASPLPolicy
import ASPL_source.agent.policy.aspl_policy as aspl_policy_module
import ASPL_source.agent.policy.base_policy as base_policy_module



# ====================
# Config
# ====================
OBS_DIM = 8
ACT_DIM = 3
SEED = 42
BATCH_SIZE = 64
N_TEST_BATCHES = 30
MAX_ACTION = 1.0
REF_QMC_SAMPLER: qmc.LatinHypercube | None = None
REF: ASPLPolicy | None = None
OUR: AsplAgent | None = None


# ====================
# Mapping: all_pairs
# ====================
def _all_pairs(ref: ASPLPolicy, our: AsplAgent):
    return [
        # actor pi
        (ref.actor.layers[0].weight, our.actor.pi.network[0].weight),
        (ref.actor.layers[0].bias, our.actor.pi.network[0].bias),
        (ref.actor.layers[2].weight, our.actor.pi.network[2].weight),
        (ref.actor.layers[2].bias, our.actor.pi.network[2].bias),
        (ref.actor.layers[4].weight, our.actor.pi.network[4].weight),
        (ref.actor.layers[4].bias, our.actor.pi.network[4].bias),
        (ref.actor.layers[6].weight, our.actor.pi.network[6].weight),
        (ref.actor.layers[6].bias, our.actor.pi.network[6].bias),
        # actor target
        (ref.actor_target.layers[0].weight, our.actor.tpi.network[0].weight),
        (ref.actor_target.layers[0].bias, our.actor.tpi.network[0].bias),
        (ref.actor_target.layers[2].weight, our.actor.tpi.network[2].weight),
        (ref.actor_target.layers[2].bias, our.actor.tpi.network[2].bias),
        (ref.actor_target.layers[4].weight, our.actor.tpi.network[4].weight),
        (ref.actor_target.layers[4].bias, our.actor.tpi.network[4].bias),
        (ref.actor_target.layers[6].weight, our.actor.tpi.network[6].weight),
        (ref.actor_target.layers[6].bias, our.actor.tpi.network[6].bias),
        # critic q1
        (ref.critic.q_networks[0][0].weight, our.critic.q1.network[0].weight),
        (ref.critic.q_networks[0][0].bias, our.critic.q1.network[0].bias),
        (ref.critic.q_networks[0][2].weight, our.critic.q1.network[2].weight),
        (ref.critic.q_networks[0][2].bias, our.critic.q1.network[2].bias),
        (ref.critic.q_networks[0][3].weight, our.critic.q1.network[3].weight),
        (ref.critic.q_networks[0][3].bias, our.critic.q1.network[3].bias),
        (ref.critic.q_networks[0][5].weight, our.critic.q1.network[5].weight),
        (ref.critic.q_networks[0][5].bias, our.critic.q1.network[5].bias),
        (ref.critic.q_networks[0][6].weight, our.critic.q1.network[6].weight),
        (ref.critic.q_networks[0][6].bias, our.critic.q1.network[6].bias),
        (ref.critic.q_networks[0][8].weight, our.critic.q1.network[8].weight),
        (ref.critic.q_networks[0][8].bias, our.critic.q1.network[8].bias),
        (ref.critic.q_networks[0][9].weight, our.critic.q1.network[9].weight),
        (ref.critic.q_networks[0][9].bias, our.critic.q1.network[9].bias),
        (ref.critic.q_networks[0][11].weight, our.critic.q1.network[11].weight),
        (ref.critic.q_networks[0][11].bias, our.critic.q1.network[11].bias),
        (ref.critic.q_networks[0][12].weight, our.critic.q1.network[12].weight),
        (ref.critic.q_networks[0][12].bias, our.critic.q1.network[12].bias),
        # critic q2
        (ref.critic.q_networks[1][0].weight, our.critic.q2.network[0].weight),
        (ref.critic.q_networks[1][0].bias, our.critic.q2.network[0].bias),
        (ref.critic.q_networks[1][2].weight, our.critic.q2.network[2].weight),
        (ref.critic.q_networks[1][2].bias, our.critic.q2.network[2].bias),
        (ref.critic.q_networks[1][3].weight, our.critic.q2.network[3].weight),
        (ref.critic.q_networks[1][3].bias, our.critic.q2.network[3].bias),
        (ref.critic.q_networks[1][5].weight, our.critic.q2.network[5].weight),
        (ref.critic.q_networks[1][5].bias, our.critic.q2.network[5].bias),
        (ref.critic.q_networks[1][6].weight, our.critic.q2.network[6].weight),
        (ref.critic.q_networks[1][6].bias, our.critic.q2.network[6].bias),
        (ref.critic.q_networks[1][8].weight, our.critic.q2.network[8].weight),
        (ref.critic.q_networks[1][8].bias, our.critic.q2.network[8].bias),
        (ref.critic.q_networks[1][9].weight, our.critic.q2.network[9].weight),
        (ref.critic.q_networks[1][9].bias, our.critic.q2.network[9].bias),
        (ref.critic.q_networks[1][11].weight, our.critic.q2.network[11].weight),
        (ref.critic.q_networks[1][11].bias, our.critic.q2.network[11].bias),
        (ref.critic.q_networks[1][12].weight, our.critic.q2.network[12].weight),
        (ref.critic.q_networks[1][12].bias, our.critic.q2.network[12].bias),
        # critic target q1
        (ref.critic_target.q_networks[0][0].weight, our.critic.tq1.network[0].weight),
        (ref.critic_target.q_networks[0][0].bias, our.critic.tq1.network[0].bias),
        (ref.critic_target.q_networks[0][2].weight, our.critic.tq1.network[2].weight),
        (ref.critic_target.q_networks[0][2].bias, our.critic.tq1.network[2].bias),
        (ref.critic_target.q_networks[0][3].weight, our.critic.tq1.network[3].weight),
        (ref.critic_target.q_networks[0][3].bias, our.critic.tq1.network[3].bias),
        (ref.critic_target.q_networks[0][5].weight, our.critic.tq1.network[5].weight),
        (ref.critic_target.q_networks[0][5].bias, our.critic.tq1.network[5].bias),
        (ref.critic_target.q_networks[0][6].weight, our.critic.tq1.network[6].weight),
        (ref.critic_target.q_networks[0][6].bias, our.critic.tq1.network[6].bias),
        (ref.critic_target.q_networks[0][8].weight, our.critic.tq1.network[8].weight),
        (ref.critic_target.q_networks[0][8].bias, our.critic.tq1.network[8].bias),
        (ref.critic_target.q_networks[0][9].weight, our.critic.tq1.network[9].weight),
        (ref.critic_target.q_networks[0][9].bias, our.critic.tq1.network[9].bias),
        (ref.critic_target.q_networks[0][11].weight, our.critic.tq1.network[11].weight),
        (ref.critic_target.q_networks[0][11].bias, our.critic.tq1.network[11].bias),
        (ref.critic_target.q_networks[0][12].weight, our.critic.tq1.network[12].weight),
        (ref.critic_target.q_networks[0][12].bias, our.critic.tq1.network[12].bias),
        # critic target q2
        (ref.critic_target.q_networks[1][0].weight, our.critic.tq2.network[0].weight),
        (ref.critic_target.q_networks[1][0].bias, our.critic.tq2.network[0].bias),
        (ref.critic_target.q_networks[1][2].weight, our.critic.tq2.network[2].weight),
        (ref.critic_target.q_networks[1][2].bias, our.critic.tq2.network[2].bias),
        (ref.critic_target.q_networks[1][3].weight, our.critic.tq2.network[3].weight),
        (ref.critic_target.q_networks[1][3].bias, our.critic.tq2.network[3].bias),
        (ref.critic_target.q_networks[1][5].weight, our.critic.tq2.network[5].weight),
        (ref.critic_target.q_networks[1][5].bias, our.critic.tq2.network[5].bias),
        (ref.critic_target.q_networks[1][6].weight, our.critic.tq2.network[6].weight),
        (ref.critic_target.q_networks[1][6].bias, our.critic.tq2.network[6].bias),
        (ref.critic_target.q_networks[1][8].weight, our.critic.tq2.network[8].weight),
        (ref.critic_target.q_networks[1][8].bias, our.critic.tq2.network[8].bias),
        (ref.critic_target.q_networks[1][9].weight, our.critic.tq2.network[9].weight),
        (ref.critic_target.q_networks[1][9].bias, our.critic.tq2.network[9].bias),
        (ref.critic_target.q_networks[1][11].weight, our.critic.tq2.network[11].weight),
        (ref.critic_target.q_networks[1][11].bias, our.critic.tq2.network[11].bias),
        (ref.critic_target.q_networks[1][12].weight, our.critic.tq2.network[12].weight),
        (ref.critic_target.q_networks[1][12].bias, our.critic.tq2.network[12].bias),
    ]


# ====================
# Init: init, sample, copy
# ====================
def _sample_transition(batch_size: int, device: torch.device):
    rng = np.random.default_rng()
    s = torch.as_tensor(rng.standard_normal((batch_size, OBS_DIM)).astype(np.float32), dtype=torch.float32, device=device)
    a = torch.as_tensor(rng.standard_normal((batch_size, ACT_DIM)).astype(np.float32), dtype=torch.float32, device=device)
    sn = torch.as_tensor(rng.standard_normal((batch_size, OBS_DIM)).astype(np.float32), dtype=torch.float32, device=device)
    r = torch.as_tensor(rng.standard_normal((batch_size, 1)).astype(np.float32), dtype=torch.float32, device=device)
    d = torch.as_tensor(rng.integers(0, 2, size=(batch_size, 1)).astype(np.float32), dtype=torch.float32, device=device)
    return s, a, sn, r, d


def _build_ref() -> ASPLPolicy:
    return ASPLPolicy(
        state_dim=OBS_DIM,
        action_dim=ACT_DIM,
        max_action=MAX_ACTION,
        use_lr_scheduler="none",
    )


def _build_our() -> AsplAgent:
    return AsplAgent(
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        max_action=MAX_ACTION,
    )


def set_seed(seed: int) -> None:
    global REF_QMC_SAMPLER
    REF_QMC_SAMPLER = qmc.LatinHypercube(d=OUR.act_dim, seed=seed)
    OUR.set_seed(seed)


# ====================
# Ref Math: 原生方法的封裝
# ====================
def _ref_td_target(ref: ASPLPolicy, sn: torch.Tensor, r: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    not_done = 1.0 - d
    target_Q = ref._compute_target_q(sn, r, not_done)
    return target_Q


def _ref_loss_td_with_target(ref: ASPLPolicy, s: torch.Tensor, a: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
    action_Q1, action_Q2 = ref.critic(s, a)
    critic_loss_supervised = F.mse_loss(action_Q1, q_target) + F.mse_loss(action_Q2, q_target)
    return critic_loss_supervised


def _ref_loss_punish_with_target(
    ref: ASPLPolicy,
    s: torch.Tensor,
    a: torch.Tensor,
    q_target: torch.Tensor,
) -> torch.Tensor:
    sampled_actions = ref.sample_actions(s.shape[0], a.shape[1])

    action_expanded = a.unsqueeze(0).expand(ref.num_sampled_actions, -1, -1)
    action_diff = (sampled_actions - action_expanded) ** 2
    f_penalty = action_diff / (2 * ref.max_action) ** 2
    f_penalty_avg = f_penalty.mean(dim=2).view(-1, 1)

    sampled_actions_flat = sampled_actions.view(-1, a.shape[1])
    state_expanded_flat = s.unsqueeze(0).expand(ref.num_sampled_actions, -1, -1).reshape(-1, s.shape[1])
    Q1_sampled, Q2_sampled = ref.critic(state_expanded_flat, sampled_actions_flat)

    target_Q_expanded = q_target.repeat(ref.num_sampled_actions, 1)
    pseudo_labels_Q1, pseudo_labels_Q2 = ref._compute_pseudo_labels(target_Q_expanded, f_penalty_avg)
    critic_loss_unsupervised = F.mse_loss(Q1_sampled, pseudo_labels_Q1) + F.mse_loss(Q2_sampled, pseudo_labels_Q2)
    return critic_loss_unsupervised


def _ref_loss_critic(
    ref: ASPLPolicy,
    s: torch.Tensor,
    a: torch.Tensor,
    sn: torch.Tensor,
    r: torch.Tensor,
    d: torch.Tensor,
) -> torch.Tensor:
    # source update() critic path (self-contained):
    # target_Q -> supervised(td) + alpha * unsupervised(punish)
    not_done = 1.0 - d
    target_Q = ref._compute_target_q(sn, r, not_done)

    action_Q1, action_Q2 = ref.critic(s, a)
    critic_loss_supervised = F.mse_loss(action_Q1, target_Q) + F.mse_loss(action_Q2, target_Q)

    sampled_actions = ref.sample_actions(s.shape[0], a.shape[1])
    action_expanded = a.unsqueeze(0).expand(ref.num_sampled_actions, -1, -1)
    action_diff = (sampled_actions - action_expanded) ** 2
    f_penalty = action_diff / (2 * ref.max_action) ** 2
    f_penalty_avg = f_penalty.mean(dim=2).view(-1, 1)

    sampled_actions_flat = sampled_actions.view(-1, a.shape[1])
    state_expanded_flat = s.unsqueeze(0).expand(ref.num_sampled_actions, -1, -1).reshape(-1, s.shape[1])
    Q1_sampled, Q2_sampled = ref.critic(state_expanded_flat, sampled_actions_flat)

    target_Q_expanded = target_Q.repeat(ref.num_sampled_actions, 1)
    pseudo_labels_Q1, pseudo_labels_Q2 = ref._compute_pseudo_labels(target_Q_expanded, f_penalty_avg)
    critic_loss_unsupervised = F.mse_loss(Q1_sampled, pseudo_labels_Q1) + F.mse_loss(Q2_sampled, pseudo_labels_Q2)

    critic_loss = critic_loss_supervised + ref.alpha * critic_loss_unsupervised
    return critic_loss


def _ref_update_and_collect_params(
    s: torch.Tensor,
    a: torch.Tensor,
    sn: torch.Tensor,
    r: torch.Tensor,
    d: torch.Tensor,
) -> list[torch.Tensor]:
    class _Buffer:
        def sample(self, batch_size, s=s, a=a, sn=sn, r=r, d=d):
            return s, a, sn, r, 1.0 - d

    buffer = _Buffer()
    _ = REF.update(buffer, batch_size=BATCH_SIZE)
    return [x for x, _ in _all_pairs(REF, OUR)]


def _our_update_and_collect_params(
    batch: dict[str, torch.Tensor],
) -> list[torch.Tensor]:
    _ = OUR.update(batch)
    return [y for _, y in _all_pairs(REF, OUR)]


# ====================
# Compare: 比較的本體
# ====================
def init_compare() -> tuple[ASPLPolicy, AsplAgent]:
    print_stage("Init")
    base_policy_module.device = torch.device("cpu")
    aspl_policy_module.device = torch.device("cpu")
    ref = _build_ref()
    our = _build_our()
    global REF, OUR
    REF = ref
    OUR = our

    # override sample by seed
    def _sample_actions_seeded(batch_size: int, action_dim: int) -> torch.Tensor:
        samples = REF_QMC_SAMPLER.random(n=ref.num_sampled_actions)
        scaled_samples = qmc.scale(samples, [-ref.max_action] * action_dim, [ref.max_action] * action_dim)
        sampled_actions_base = torch.as_tensor(scaled_samples, dtype=torch.float32, device=base_policy_module.device)
        return sampled_actions_base.unsqueeze(1).repeat(1, batch_size, 1)
    ref.sample_actions = _sample_actions_seeded

    # override set seed by add qmc seed
    def _set_seed_patched(seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        import random
        random.seed(seed)
        set_seed(seed)
    _lib._set_seed = _set_seed_patched
    
    with torch.no_grad():
        for ref_param, our_param in _all_pairs(ref, our):
            our_param.copy_(ref_param.to(our_param.device))
    return ref, our


def compare_act(ref: ASPLPolicy, our: AsplAgent) -> None:
    print_stage("Act Compare")
    for i in range(1, N_TEST_BATCHES + 1):
        obs_single = np.random.default_rng().standard_normal((OBS_DIM,)).astype(np.float32)
        obs_batch_np = np.random.default_rng().standard_normal((BATCH_SIZE, OBS_DIM)).astype(np.float32)
        obs_batch_t = torch.as_tensor(obs_batch_np, dtype=torch.float32, device="cpu")

        assert_callback(
            lambda: [ref.select_action(obs_single)],
            lambda: [our.act(obs_single, greedy=True)],
            label=f"act_single[{i}]",
            seed=SEED + i,
        )
        assert_callback(
            lambda: [ref.actor(obs_batch_t).detach().cpu().numpy()],
            lambda: [our.act_batch(obs_batch_np, greedy=True)],
            label=f"act_batch[{i}]",
            seed=SEED + 1000 + i,
        )


def compare_loss(ref: ASPLPolicy, our: AsplAgent) -> None:
    print_stage("Loss Compare")
    for i in range(1, N_TEST_BATCHES + 1):
        s, a, sn, r, d = _sample_transition(BATCH_SIZE, torch.device("cpu"))
        ref.total_it = i
        our.update_step = i
        ref.mean_abs_q = 0.0
        our.q_mean = 0.0

        assert_callback(
            lambda: [_ref_loss_td_with_target(ref, s, a, _ref_td_target(ref, sn, r, d)).detach().cpu()],
            lambda: [our.loss_td_with_target(s, a, our.td_target(sn, r, d)).detach().cpu()],
            label=f"loss_td[{i}]",
            seed=SEED + 2000 + i,
        )
        assert_callback(
            lambda: [_ref_loss_punish_with_target(ref, s, a, _ref_td_target(ref, sn, r, d)).detach().cpu()],
            lambda: [our.loss_punish_with_target(s, a, our.td_target(sn, r, d)).detach().cpu()],
            label=f"loss_punish[{i}]",
            seed=SEED + 2100 + i,
        )
        assert_callback(
            lambda: [_ref_loss_critic(ref, s, a, sn, r, d).detach().cpu()],
            lambda: [our.loss_critic(s, a, r, sn, d).detach().cpu()],
            label=f"loss_critic[{i}]",
            seed=SEED + 2200 + i,
        )
        assert_callback(
            lambda: [ref._compute_actor_loss(s).detach().cpu()],
            lambda: [our.loss_td3_variant(s).detach().cpu()],
            label=f"loss_actor[{i}]",
            seed=SEED + 2300 + i,
        )


def compare_param(ref: ASPLPolicy, our: AsplAgent) -> None:
    print_stage("Param Compare")
    ref.total_it = 0
    our.update_step = 0
    ref.mean_abs_q = 0.0
    our.q_mean = 0.0

    for i in range(1, N_TEST_BATCHES + 1):
        s, a, sn, r, d = _sample_transition(BATCH_SIZE, torch.device("cpu"))
        assert_callback(
            lambda: _ref_update_and_collect_params(s, a, sn, r, d),
            lambda: _our_update_and_collect_params({"obs": s, "act": a, "rew": r, "next_obs": sn, "done": d}),
            label=f"update[{i}]",
            seed=SEED + 3000 + i,
        )


# ====================
# main & __main__
# ====================
def main() -> None:
    ref, our = init_compare()
    compare_act(ref, our)
    compare_loss(ref, our)
    compare_param(ref, our)
    print_stage("Result")
    print("PASS: ASPL act/loss/update/params are exactly aligned.")


if __name__ == "__main__":
    main()
