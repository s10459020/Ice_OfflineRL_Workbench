from __future__ import annotations

import numpy as np
import torch

import d3rlpy
from d3rlpy.models.torch.imitators import compute_discrete_imitation_loss
from d3rlpy.models.torch.policies import CategoricalPolicy

from ice_offline.agent.bc_agent_discrete import BCAgentDiscrete
from ice_offline.tools.printer import print_stage

# ====================
# Config
# ====================
OBS_DIM = 8
N_ACTIONS = 4
DEVICE = "cpu"
SEED = 123
BATCH_SIZE = 64
N_TEST_BATCHES = 20
TOL_GRAD = 1e-8


# ====================
# Build
# ====================
def build_our_agent() -> BCAgentDiscrete:
    return BCAgentDiscrete(obs_size=OBS_DIM, act_size=N_ACTIONS)


def build_d3rl_policy() -> tuple[CategoricalPolicy, torch.optim.Optimizer]:
    config = d3rlpy.algos.DiscreteBCConfig()
    algo = config.create(device=DEVICE)
    algo.create_impl(observation_shape=(OBS_DIM,), action_size=N_ACTIONS)
    assert algo.impl is not None
    return algo.impl.modules.imitator, algo.impl.modules.optim.optim


def copy_d3rl_weights_to_our(
    d3_policy: CategoricalPolicy, our_agent: BCAgentDiscrete
) -> None:
    with torch.no_grad():
        our_agent.policy.network[0].weight.copy_(d3_policy._encoder._layers[0].weight)
        our_agent.policy.network[0].bias.copy_(d3_policy._encoder._layers[0].bias)
        our_agent.policy.network[2].weight.copy_(d3_policy._encoder._layers[2].weight)
        our_agent.policy.network[2].bias.copy_(d3_policy._encoder._layers[2].bias)
        our_agent.policy.network[4].weight.copy_(d3_policy._fc.weight)
        our_agent.policy.network[4].bias.copy_(d3_policy._fc.bias)


def d3rl_predict_best_action(policy: CategoricalPolicy, obs_np: np.ndarray) -> np.ndarray:
    obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        dist = policy(obs_t)
        action_t = torch.argmax(dist.logits, dim=1)
    return action_t.cpu().numpy()


def _pairs(our_agent: BCAgentDiscrete, d3_policy: CategoricalPolicy):
    return [
        (our_agent.policy.network[0].weight, d3_policy._encoder._layers[0].weight),
        (our_agent.policy.network[0].bias, d3_policy._encoder._layers[0].bias),
        (our_agent.policy.network[2].weight, d3_policy._encoder._layers[2].weight),
        (our_agent.policy.network[2].bias, d3_policy._encoder._layers[2].bias),
        (our_agent.policy.network[4].weight, d3_policy._fc.weight),
        (our_agent.policy.network[4].bias, d3_policy._fc.bias),
    ]


def main() -> None:
    print_stage("Init")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    our_agent = build_our_agent()
    d3_policy, d3_optim = build_d3rl_policy()
    copy_d3rl_weights_to_our(d3_policy, our_agent)

    print_stage("Act Compare")
    rng = np.random.default_rng(SEED)
    for i in range(1, N_TEST_BATCHES + 1):
        obs_np = rng.standard_normal((BATCH_SIZE, OBS_DIM), dtype=np.float32)

        # compare action path (action_best)
        d3_act = d3rl_predict_best_action(d3_policy, obs_np)
        our_act = our_agent.action_best_batch(obs_np)

        if not np.array_equal(d3_act, our_act):
            mismatch = int((d3_act != our_act).sum())
            raise SystemExit(
                f"FAIL: action mismatch (d3rl vs bc_agent_discrete) at batch={i}, mismatch_count={mismatch}"
            )

        if i == 1 or i % 5 == 0 or i == N_TEST_BATCHES:
            print(
                f"batch={i}/{N_TEST_BATCHES} action_match=True"
            )

    print_stage("Loss Compare")
    rng = np.random.default_rng(SEED + 1)
    for i in range(1, N_TEST_BATCHES + 1):
        obs_np = rng.standard_normal((BATCH_SIZE, OBS_DIM), dtype=np.float32)
        act_np = rng.integers(0, N_ACTIONS, size=(BATCH_SIZE,), dtype=np.int64)
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=DEVICE)
        act_t = torch.as_tensor(act_np, dtype=torch.long, device=DEVICE)

        our_dist = our_agent.policy(obs_t)
        our_loss_bc = our_agent._loss_bc(our_dist.logits, act_t)
        our_loss_regular = our_agent._loss_regular(our_dist.logits)
        our_loss = our_agent._loss(our_dist.logits, act_t)
        d3_loss = compute_discrete_imitation_loss(
            policy=d3_policy,
            x=obs_t,
            action=act_t,
            beta=our_agent.beta,
        )

        loss_diff = float(abs(our_loss.item() - d3_loss.loss.item()))
        loss_bc_diff = float(abs(our_loss_bc.item() - d3_loss.imitation_loss.item()))
        loss_regular_diff = float(
            abs(our_loss_regular.item() - d3_loss.regularization_loss.item())
        )
        if max(loss_diff, loss_bc_diff, loss_regular_diff) != 0.0:
            raise SystemExit(
                f"FAIL: loss mismatch at batch={i} "
                f"loss={loss_diff:.12e} bc={loss_bc_diff:.12e} reg={loss_regular_diff:.12e}"
            )

        if i == 1 or i % 5 == 0 or i == N_TEST_BATCHES:
            print(
                f"batch={i}/{N_TEST_BATCHES} "
                f"loss_diff={loss_diff:.12e} "
                f"loss_bc_diff={loss_bc_diff:.12e} "
                f"loss_regular_diff={loss_regular_diff:.12e}"
            )

    print_stage("Update Compare")
    rng = np.random.default_rng(SEED + 2)
    for i in range(1, N_TEST_BATCHES + 1):
        obs_np = rng.standard_normal((BATCH_SIZE, OBS_DIM), dtype=np.float32)
        act_np = rng.integers(0, N_ACTIONS, size=(BATCH_SIZE,), dtype=np.int64)
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=DEVICE)
        act_t = torch.as_tensor(act_np, dtype=torch.long, device=DEVICE)

        our_agent.optimizer.zero_grad()
        our_dist = our_agent.policy(obs_t)
        our_loss = our_agent._loss(our_dist.logits, act_t)
        our_loss.backward()

        d3_optim.zero_grad()
        d3_loss = compute_discrete_imitation_loss(
            policy=d3_policy,
            x=obs_t,
            action=act_t,
            beta=our_agent.beta,
        )
        d3_loss.loss.backward()

        grad_max_diff = 0.0
        for our_param, d3_param in _pairs(our_agent, d3_policy):
            diff = float((our_param.grad - d3_param.grad).abs().max().item())
            grad_max_diff = max(grad_max_diff, diff)    
        if grad_max_diff > TOL_GRAD:
            raise SystemExit(
                f"FAIL: gradient mismatch at batch={i}, max_abs_diff={grad_max_diff:.12e}"
            )

        our_agent.optimizer.step()
        d3_optim.step()

        param_max_diff = 0.0
        for our_param, d3_param in _pairs(our_agent, d3_policy):
            diff = float((our_param - d3_param).abs().max().item())
            param_max_diff = max(param_max_diff, diff)
        if param_max_diff != 0.0:
            raise SystemExit(
                f"FAIL: param mismatch at batch={i}, max_abs_diff={param_max_diff:.12e}"
            )

        if i == 1 or i % 5 == 0 or i == N_TEST_BATCHES:
            print(
                f"batch={i}/{N_TEST_BATCHES} "
                f"grad_max_abs_diff={grad_max_diff:.12e} "
                f"param_max_abs_diff={param_max_diff:.12e}"
            )

    print_stage("Result")
    print("PASS: act, loss, gradient, and parameter updates are aligned with d3rl.")


if __name__ == "__main__":
    main()
