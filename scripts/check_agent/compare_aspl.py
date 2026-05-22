import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

from ice_offline.agent.aspl import AsplAgent
from ice_offline.tools.printer import print_stage
from _lib import assert_callback
from _lib import assert_list

# Dummy library
if "d4rl" not in sys.modules:
    sys.modules["d4rl"] = types.ModuleType("d4rl")
from ASPL_source.agent.policy.aspl_policy import ASPLPolicy



# ====================
# Config
# ====================
OBS_DIM = 8
ACT_DIM = 3
SEED = 42
BATCH_SIZE = 64
N_TEST_BATCHES = 30
MAX_ACTION = 1.0


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
        # actor target
        (ref.actor_target.layers[0].weight, our.actor.tpi.network[0].weight),
        (ref.actor_target.layers[0].bias, our.actor.tpi.network[0].bias),
        (ref.actor_target.layers[2].weight, our.actor.tpi.network[2].weight),
        (ref.actor_target.layers[2].bias, our.actor.tpi.network[2].bias),
        (ref.actor_target.layers[4].weight, our.actor.tpi.network[4].weight),
        (ref.actor_target.layers[4].bias, our.actor.tpi.network[4].bias),
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
def _sample_transition(rng: np.random.Generator, batch_size: int, device: torch.device):
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


def _ref_loss_punish_with_target(ref: ASPLPolicy, s: torch.Tensor, a: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
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


def _ref_loss_critic(ref: ASPLPolicy, s: torch.Tensor, a: torch.Tensor, sn: torch.Tensor, r: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
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


# ====================
# Compare: 比較的本體
# ====================
def init_compare() -> tuple[ASPLPolicy, AsplAgent]:
    print_stage("Init")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    ref = _build_ref()
    our = _build_our()
    with torch.no_grad():
        for ref_param, our_param in _all_pairs(ref, our):
            our_param.copy_(ref_param.to(our_param.device))
    return ref, our


def compare_act(ref: ASPLPolicy, our: AsplAgent) -> None:
    print_stage("Act Compare")
    rng = np.random.default_rng(SEED)
    for i in range(1, N_TEST_BATCHES + 1):
        obs_single = rng.standard_normal((OBS_DIM,)).astype(np.float32)
        obs_batch_np = rng.standard_normal((BATCH_SIZE, OBS_DIM)).astype(np.float32)
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
    rng = np.random.default_rng(SEED + 1)
    for i in range(1, N_TEST_BATCHES + 1):
        s, a, sn, r, d = _sample_transition(rng, BATCH_SIZE, torch.device("cpu"))
        ref.total_it = i
        our.update_step = i
        ref.mean_abs_q = 0.0
        our.q_mean = 0.0
        q_target_ref = _ref_td_target(ref, sn, r, d)
        q_target_our = our._td_target(sn, r, d)

        assert_callback(
            lambda: [_ref_loss_td_with_target(ref, s, a, q_target_ref).detach().cpu()],
            lambda: [our.loss_td_with_target(s, a, q_target_our).detach().cpu()],
            label=f"loss_td[{i}]",
            seed=SEED + 2000 + i,
        )
        assert_callback(
            lambda: [_ref_loss_punish_with_target(ref, s, a, q_target_ref).detach().cpu()],
            lambda: [our.loss_punish_with_target(s, a, q_target_our).detach().cpu()],
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
    rng = np.random.default_rng(SEED + 2)
    ref.total_it = 0
    our.update_step = 0
    ref.mean_abs_q = 0.0
    our.q_mean = 0.0

    for i in range(1, N_TEST_BATCHES + 1):
        s, a, sn, r, d = _sample_transition(rng, BATCH_SIZE, torch.device("cpu"))
        class _Buffer:
            def sample(self, batch_size, s=s, a=a, sn=sn, r=r, d=d):
                return s, a, sn, r, 1.0 - d

        buffer = _Buffer()

        assert_callback(
            lambda: [ref.update(buffer, batch_size=BATCH_SIZE)],
            lambda: [our.update({"obs": s, "act": a, "rew": r, "next_obs": sn, "done": d})],
            label=f"update[{i}]",
            seed=SEED + 3000 + i,
        )

        pair_list = _all_pairs(ref, our)
        ref_list = [x for x, _ in pair_list]
        our_list = [y for _, y in pair_list]
        assert_list(ref_list, our_list, label=f"params[{i}]")


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
