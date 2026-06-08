import sys
import types

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import qmc

# Dummy library
sys.modules["d4rl"] = types.ModuleType("d4rl")

import ASPL_source.agent.policy.aspl_policy as aspl_policy_module
import ASPL_source.agent.policy.base_policy as base_policy_module
from _lib import assert_callback
from _lib import sample_transition
from _lib import torch_buffer
from ASPL_source.agent.policy.aspl_policy import ASPLPolicy
from ice_offline.agent.aspl import AsplAgent
from ice_offline.tools.printer import print_stage
import _lib


# ====================
# Config
# ====================
obs_size = 8
act_size = 3
SEED = 42
BATCH_SIZE = 64
N_TEST_BATCHES = 30
MAX_ACTION = 1.0


# ====================
# Mapping
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
        (ref.critic.q_networks[0][0].weight, our.critic.q_networks[0].network[0].weight),
        (ref.critic.q_networks[0][0].bias, our.critic.q_networks[0].network[0].bias),
        (ref.critic.q_networks[0][2].weight, our.critic.q_networks[0].network[2].weight),
        (ref.critic.q_networks[0][2].bias, our.critic.q_networks[0].network[2].bias),
        (ref.critic.q_networks[0][3].weight, our.critic.q_networks[0].network[3].weight),
        (ref.critic.q_networks[0][3].bias, our.critic.q_networks[0].network[3].bias),
        (ref.critic.q_networks[0][5].weight, our.critic.q_networks[0].network[5].weight),
        (ref.critic.q_networks[0][5].bias, our.critic.q_networks[0].network[5].bias),
        (ref.critic.q_networks[0][6].weight, our.critic.q_networks[0].network[6].weight),
        (ref.critic.q_networks[0][6].bias, our.critic.q_networks[0].network[6].bias),
        (ref.critic.q_networks[0][8].weight, our.critic.q_networks[0].network[8].weight),
        (ref.critic.q_networks[0][8].bias, our.critic.q_networks[0].network[8].bias),
        (ref.critic.q_networks[0][9].weight, our.critic.q_networks[0].network[9].weight),
        (ref.critic.q_networks[0][9].bias, our.critic.q_networks[0].network[9].bias),
        (ref.critic.q_networks[0][11].weight, our.critic.q_networks[0].network[11].weight),
        (ref.critic.q_networks[0][11].bias, our.critic.q_networks[0].network[11].bias),
        (ref.critic.q_networks[0][12].weight, our.critic.q_networks[0].network[12].weight),
        (ref.critic.q_networks[0][12].bias, our.critic.q_networks[0].network[12].bias),
        # critic q2
        (ref.critic.q_networks[1][0].weight, our.critic.q_networks[1].network[0].weight),
        (ref.critic.q_networks[1][0].bias, our.critic.q_networks[1].network[0].bias),
        (ref.critic.q_networks[1][2].weight, our.critic.q_networks[1].network[2].weight),
        (ref.critic.q_networks[1][2].bias, our.critic.q_networks[1].network[2].bias),
        (ref.critic.q_networks[1][3].weight, our.critic.q_networks[1].network[3].weight),
        (ref.critic.q_networks[1][3].bias, our.critic.q_networks[1].network[3].bias),
        (ref.critic.q_networks[1][5].weight, our.critic.q_networks[1].network[5].weight),
        (ref.critic.q_networks[1][5].bias, our.critic.q_networks[1].network[5].bias),
        (ref.critic.q_networks[1][6].weight, our.critic.q_networks[1].network[6].weight),
        (ref.critic.q_networks[1][6].bias, our.critic.q_networks[1].network[6].bias),
        (ref.critic.q_networks[1][8].weight, our.critic.q_networks[1].network[8].weight),
        (ref.critic.q_networks[1][8].bias, our.critic.q_networks[1].network[8].bias),
        (ref.critic.q_networks[1][9].weight, our.critic.q_networks[1].network[9].weight),
        (ref.critic.q_networks[1][9].bias, our.critic.q_networks[1].network[9].bias),
        (ref.critic.q_networks[1][11].weight, our.critic.q_networks[1].network[11].weight),
        (ref.critic.q_networks[1][11].bias, our.critic.q_networks[1].network[11].bias),
        (ref.critic.q_networks[1][12].weight, our.critic.q_networks[1].network[12].weight),
        (ref.critic.q_networks[1][12].bias, our.critic.q_networks[1].network[12].bias),
        # critic target q1
        (ref.critic_target.q_networks[0][0].weight, our.critic.tq_networks[0].network[0].weight),
        (ref.critic_target.q_networks[0][0].bias, our.critic.tq_networks[0].network[0].bias),
        (ref.critic_target.q_networks[0][2].weight, our.critic.tq_networks[0].network[2].weight),
        (ref.critic_target.q_networks[0][2].bias, our.critic.tq_networks[0].network[2].bias),
        (ref.critic_target.q_networks[0][3].weight, our.critic.tq_networks[0].network[3].weight),
        (ref.critic_target.q_networks[0][3].bias, our.critic.tq_networks[0].network[3].bias),
        (ref.critic_target.q_networks[0][5].weight, our.critic.tq_networks[0].network[5].weight),
        (ref.critic_target.q_networks[0][5].bias, our.critic.tq_networks[0].network[5].bias),
        (ref.critic_target.q_networks[0][6].weight, our.critic.tq_networks[0].network[6].weight),
        (ref.critic_target.q_networks[0][6].bias, our.critic.tq_networks[0].network[6].bias),
        (ref.critic_target.q_networks[0][8].weight, our.critic.tq_networks[0].network[8].weight),
        (ref.critic_target.q_networks[0][8].bias, our.critic.tq_networks[0].network[8].bias),
        (ref.critic_target.q_networks[0][9].weight, our.critic.tq_networks[0].network[9].weight),
        (ref.critic_target.q_networks[0][9].bias, our.critic.tq_networks[0].network[9].bias),
        (ref.critic_target.q_networks[0][11].weight, our.critic.tq_networks[0].network[11].weight),
        (ref.critic_target.q_networks[0][11].bias, our.critic.tq_networks[0].network[11].bias),
        (ref.critic_target.q_networks[0][12].weight, our.critic.tq_networks[0].network[12].weight),
        (ref.critic_target.q_networks[0][12].bias, our.critic.tq_networks[0].network[12].bias),
        # critic target q2
        (ref.critic_target.q_networks[1][0].weight, our.critic.tq_networks[1].network[0].weight),
        (ref.critic_target.q_networks[1][0].bias, our.critic.tq_networks[1].network[0].bias),
        (ref.critic_target.q_networks[1][2].weight, our.critic.tq_networks[1].network[2].weight),
        (ref.critic_target.q_networks[1][2].bias, our.critic.tq_networks[1].network[2].bias),
        (ref.critic_target.q_networks[1][3].weight, our.critic.tq_networks[1].network[3].weight),
        (ref.critic_target.q_networks[1][3].bias, our.critic.tq_networks[1].network[3].bias),
        (ref.critic_target.q_networks[1][5].weight, our.critic.tq_networks[1].network[5].weight),
        (ref.critic_target.q_networks[1][5].bias, our.critic.tq_networks[1].network[5].bias),
        (ref.critic_target.q_networks[1][6].weight, our.critic.tq_networks[1].network[6].weight),
        (ref.critic_target.q_networks[1][6].bias, our.critic.tq_networks[1].network[6].bias),
        (ref.critic_target.q_networks[1][8].weight, our.critic.tq_networks[1].network[8].weight),
        (ref.critic_target.q_networks[1][8].bias, our.critic.tq_networks[1].network[8].bias),
        (ref.critic_target.q_networks[1][9].weight, our.critic.tq_networks[1].network[9].weight),
        (ref.critic_target.q_networks[1][9].bias, our.critic.tq_networks[1].network[9].bias),
        (ref.critic_target.q_networks[1][11].weight, our.critic.tq_networks[1].network[11].weight),
        (ref.critic_target.q_networks[1][11].bias, our.critic.tq_networks[1].network[11].bias),
        (ref.critic_target.q_networks[1][12].weight, our.critic.tq_networks[1].network[12].weight),
        (ref.critic_target.q_networks[1][12].bias, our.critic.tq_networks[1].network[12].bias),
    ]


# ====================
# Ref define
# ====================
def ref_td_target(ref: ASPLPolicy, sn: torch.Tensor, r: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    not_done = 1.0 - d
    return ref._compute_target_q(sn, r, not_done)

def ref_loss_td_with_target(
    ref: ASPLPolicy,
    s: torch.Tensor,
    a: torch.Tensor,
    q_target: torch.Tensor,
) -> torch.Tensor:
    action_q1, action_q2 = ref.critic(s, a)
    return F.mse_loss(action_q1, q_target) + F.mse_loss(action_q2, q_target)

def ref_loss_punish_with_target(
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
    q1_sampled, q2_sampled = ref.critic(state_expanded_flat, sampled_actions_flat)
    target_q_expanded = q_target.repeat(ref.num_sampled_actions, 1)
    pseudo_labels_q1, pseudo_labels_q2 = ref._compute_pseudo_labels(target_q_expanded, f_penalty_avg)
    return F.mse_loss(q1_sampled, pseudo_labels_q1) + F.mse_loss(q2_sampled, pseudo_labels_q2)

def ref_loss_critic(
    ref: ASPLPolicy,
    s: torch.Tensor,
    a: torch.Tensor,
    sn: torch.Tensor,
    r: torch.Tensor,
    d: torch.Tensor,
) -> torch.Tensor:
    not_done = 1.0 - d
    target_q = ref._compute_target_q(sn, r, not_done)
    loss_td = ref_loss_td_with_target(ref, s, a, target_q)
    loss_punish = ref_loss_punish_with_target(ref, s, a, target_q)
    return loss_td + ref.alpha * loss_punish

def ref_update_and_collect_params(
    ref: ASPLPolicy,
    our: AsplAgent,
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
    _ = ref.update(buffer, batch_size=BATCH_SIZE)
    return [x for x, _ in _all_pairs(ref, our)]


# ====================
# Our define
# ====================
def our_update_and_collect_params(
    ref: ASPLPolicy,
    our: AsplAgent,
    s: torch.Tensor,
    a: torch.Tensor,
    r: torch.Tensor,
    sn: torch.Tensor,
    d: torch.Tensor,
) -> list[torch.Tensor]:
    _ = our.update(torch_buffer(s, a, r, sn, d))
    return [y for _, y in _all_pairs(ref, our)]


# ====================
# Compare
# ====================
def build_our() -> AsplAgent:
    return AsplAgent(
        obs_size=obs_size,
        act_size=act_size,
    )

def build_ref() -> ASPLPolicy:
    return ASPLPolicy(
        state_dim=obs_size,
        action_dim=act_size,
        max_action=MAX_ACTION,
        use_lr_scheduler="none",
    )

def init_compare() -> tuple[ASPLPolicy, AsplAgent]:
    print_stage("Init")
    base_policy_module.device = torch.device("cpu")
    aspl_policy_module.device = torch.device("cpu")
    ref = build_ref()
    our = build_our()
    sampler_holder = {"sampler": qmc.LatinHypercube(d=our.act_size, seed=SEED)}

    def _sample_actions_seeded(batch_size: int, action_dim: int) -> torch.Tensor:
        samples = sampler_holder["sampler"].random(n=ref.num_sampled_actions)
        scaled_samples = qmc.scale(samples, [-ref.max_action] * action_dim, [ref.max_action] * action_dim)
        sampled_actions_base = torch.as_tensor(scaled_samples, dtype=torch.float32, device=base_policy_module.device)
        return sampled_actions_base.unsqueeze(1).repeat(1, batch_size, 1)

    ref.sample_actions = _sample_actions_seeded

    def _set_seed_patched(seed: int) -> None:
        import random

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        sampler_holder["sampler"] = qmc.LatinHypercube(d=our.act_size, seed=seed)
        our.actor.set_seed(seed)

    _lib._set_seed = _set_seed_patched

    with torch.no_grad():
        for ref_param, our_param in _all_pairs(ref, our):
            our_param.copy_(ref_param.to(our_param.device))

    return ref, our

def compare_act(ref: ASPLPolicy, our: AsplAgent) -> None:
    print_stage("Act Compare")
    for i in range(1, N_TEST_BATCHES + 1):
        s_single, _, _, _, _ = sample_transition(1, obs_size, act_size, "cpu")
        s, _, _, _, _ = sample_transition(BATCH_SIZE, obs_size, act_size, "cpu")

        # act best
        assert_callback(
            lambda: [ref.select_action(s_single[0].cpu().numpy())],
            lambda: [our.act(s_single[0].cpu().numpy())],
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

def compare_loss(ref: ASPLPolicy, our: AsplAgent) -> None:
    print_stage("Loss Compare")
    for i in range(1, N_TEST_BATCHES + 1):
        s, a, r, sn, d = sample_transition(BATCH_SIZE, obs_size, act_size, "cpu")
        batch = torch_buffer(s, a, r, sn, d)
        ref.total_it = i
        our.update_step = i
        ref.mean_abs_q = 0.0
        our.critic.q_mean = 0.0

        # loss td
        assert_callback(
            lambda: [ref_loss_td_with_target(ref, s, a, ref_td_target(ref, sn, r, d)).detach().cpu()],
            lambda: [our.loss_td_with_target(torch_buffer(s, a, r, sn, d), our.target_td3(sn, r, d)).detach().cpu()],
            label=f"loss_td[{i}]",
            seed=SEED + i,
        )

        # loss punish
        assert_callback(
            lambda: [ref_loss_punish_with_target(ref, s, a, ref_td_target(ref, sn, r, d)).detach().cpu()],
            lambda: [our.loss_punish_with_target(torch_buffer(s, a, r, sn, d), our.target_td3(sn, r, d)).detach().cpu()],
            label=f"loss_punish[{i}]",
            seed=SEED + i,
        )

        # loss actor
        assert_callback(
            lambda: [ref._compute_actor_loss(s).detach().cpu()],
            lambda: [our.loss_td3(batch).detach().cpu()],
            label=f"loss_actor[{i}]",
            seed=SEED + i,
        )

        # loss critic
        assert_callback(
            lambda: [ref_loss_critic(ref, s, a, sn, r, d).detach().cpu()],
            lambda: [our.loss_critic(batch).detach().cpu()],
            label=f"loss_critic[{i}]",
            seed=SEED + i,
        )

        print(f"batch={i}/{N_TEST_BATCHES} loss_match=True")

def compare_param(ref: ASPLPolicy, our: AsplAgent) -> None:
    print_stage("Update Compare")
    ref.total_it = 0
    our.update_step = 0
    ref.mean_abs_q = 0.0
    our.critic.q_mean = 0.0
    for i in range(1, N_TEST_BATCHES + 1):
        s, a, r, sn, d = sample_transition(BATCH_SIZE, obs_size, act_size, "cpu")

        # update params
        assert_callback(
            lambda: ref_update_and_collect_params(ref, our, s, a, sn, r, d),
            lambda: our_update_and_collect_params(ref, our, s, a, r, sn, d),
            label=f"update[{i}]",
            seed=SEED + i,
        )

        print(f"batch={i}/{N_TEST_BATCHES} param_match=True")


# ====================
# Main
# ====================
if __name__ == "__main__":
    ref, our = init_compare()
    compare_act(ref, our)
    compare_loss(ref, our)
    compare_param(ref, our)
    print_stage("Result")
    print("PASS: ASPL act/loss/update/params are exactly aligned.")
