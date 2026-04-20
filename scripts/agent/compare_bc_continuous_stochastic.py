from __future__ import annotations

import numpy as np
import torch

import d3rlpy
from d3rlpy.models.torch.imitators import compute_stochastic_imitation_loss
from d3rlpy.models.torch.policies import NormalPolicy

from ice_offline.agent.bc_agent_continuous_stochastic import (
    BCAgentContinuousStochastic,
)
from ice_offline.tools.printer import print_stage

# ====================
# Config
# ====================
OBS_DIM = 8
ACT_DIM = 3
DEVICE = "cpu"
SEED = 123
BATCH_SIZE = 64
N_TEST_BATCHES = 20
TOL_GRAD = 1e-8


def build_our_agent() -> BCAgentContinuousStochastic:
    return BCAgentContinuousStochastic(obs_size=OBS_DIM, act_size=ACT_DIM)


def build_d3rl_policy() -> tuple[NormalPolicy, torch.optim.Optimizer]:
    config = d3rlpy.algos.BCConfig(policy_type="stochastic")
    algo = config.create(device=DEVICE)
    algo.create_impl(observation_shape=(OBS_DIM,), action_size=ACT_DIM)
    assert algo.impl is not None
    return algo.impl.modules.imitator, algo.impl.modules.optim.optim


def copy_d3rl_weights_to_our(
    d3_policy: NormalPolicy, our_agent: BCAgentContinuousStochastic
) -> None:
    with torch.no_grad():
        our_agent.policy.network[0].weight.copy_(d3_policy._encoder._layers[0].weight)
        our_agent.policy.network[0].bias.copy_(d3_policy._encoder._layers[0].bias)
        our_agent.policy.network[2].weight.copy_(d3_policy._encoder._layers[2].weight)
        our_agent.policy.network[2].bias.copy_(d3_policy._encoder._layers[2].bias)
        our_agent.policy.mean_head.weight.copy_(d3_policy._mu.weight)
        our_agent.policy.mean_head.bias.copy_(d3_policy._mu.bias)
        our_agent.policy.logstd_head.weight.copy_(d3_policy._logstd.weight)
        our_agent.policy.logstd_head.bias.copy_(d3_policy._logstd.bias)


def d3rl_predict_best_action(policy: NormalPolicy, obs_np: np.ndarray) -> np.ndarray:
    obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        action_t = policy(obs_t).squashed_mu
    return action_t.cpu().numpy()


def _pairs(our_agent: BCAgentContinuousStochastic, d3_policy: NormalPolicy):
    return [
        (our_agent.policy.network[0].weight, d3_policy._encoder._layers[0].weight),
        (our_agent.policy.network[0].bias, d3_policy._encoder._layers[0].bias),
        (our_agent.policy.network[2].weight, d3_policy._encoder._layers[2].weight),
        (our_agent.policy.network[2].bias, d3_policy._encoder._layers[2].bias),
        (our_agent.policy.mean_head.weight, d3_policy._mu.weight),
        (our_agent.policy.mean_head.bias, d3_policy._mu.bias),
        (our_agent.policy.logstd_head.weight, d3_policy._logstd.weight),
        (our_agent.policy.logstd_head.bias, d3_policy._logstd.bias),
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

        d3_act = d3rl_predict_best_action(d3_policy, obs_np)
        our_act = our_agent.action_best_batch(obs_np)
        if not np.array_equal(d3_act, our_act):
            max_diff = float(np.abs(d3_act - our_act).max())
            raise SystemExit(
                f"FAIL: action mismatch (d3rl vs bc_agent_continuous_stochastic) at batch={i}, max_abs_diff={max_diff:.12e}"
            )

        if i == 1 or i % 5 == 0 or i == N_TEST_BATCHES:
            print(f"batch={i}/{N_TEST_BATCHES} action_match=True")

    print_stage("Loss Compare")
    rng = np.random.default_rng(SEED + 1)
    for i in range(1, N_TEST_BATCHES + 1):
        obs_np = rng.standard_normal((BATCH_SIZE, OBS_DIM), dtype=np.float32)
        act_np = rng.standard_normal((BATCH_SIZE, ACT_DIM), dtype=np.float32)
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=DEVICE)
        act_t = torch.as_tensor(act_np, dtype=torch.float32, device=DEVICE)

        torch.manual_seed(SEED + 1000 + i)
        our_loss = our_agent._loss_from_obs(obs_t, act_t)

        torch.manual_seed(SEED + 1000 + i)
        d3_loss = compute_stochastic_imitation_loss(
            policy=d3_policy,
            x=obs_t,
            action=act_t,
        )

        loss_diff = float(abs(our_loss.item() - d3_loss.loss.item()))
        if loss_diff != 0.0:
            raise SystemExit(f"FAIL: loss mismatch at batch={i}, loss={loss_diff:.12e}")

        if i == 1 or i % 5 == 0 or i == N_TEST_BATCHES:
            print(f"batch={i}/{N_TEST_BATCHES} loss_diff={loss_diff:.12e}")

    print_stage("Update Compare")
    rng = np.random.default_rng(SEED + 2)
    for i in range(1, N_TEST_BATCHES + 1):
        obs_np = rng.standard_normal((BATCH_SIZE, OBS_DIM), dtype=np.float32)
        act_np = rng.standard_normal((BATCH_SIZE, ACT_DIM), dtype=np.float32)
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=DEVICE)
        act_t = torch.as_tensor(act_np, dtype=torch.float32, device=DEVICE)

        our_agent.optimizer.zero_grad()
        torch.manual_seed(SEED + 2000 + i)
        our_loss = our_agent._loss_from_obs(obs_t, act_t)
        our_loss.backward()

        d3_optim.zero_grad()
        torch.manual_seed(SEED + 2000 + i)
        d3_loss = compute_stochastic_imitation_loss(
            policy=d3_policy,
            x=obs_t,
            action=act_t,
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
