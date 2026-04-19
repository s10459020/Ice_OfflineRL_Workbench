from __future__ import annotations

import numpy as np
import torch

import d3rlpy
from d3rlpy.models.torch.imitators import compute_discrete_imitation_loss
from d3rlpy.models.torch.policies import CategoricalPolicy
from ice_offline.agent.bc_d3rl_discrete import DiscreteBCAgent
from ice_offline.tools.printer import print_stage

# ====================
# Config
# ====================
OBS_DIM = 8
N_ACTIONS = 4
HIDDEN_UNITS = (256, 256)
BETA = 0.5
LEARNING_RATE = 1e-3
DEVICE = "cpu"
SEED = 123
BATCH_SIZE = 32
TRAIN_STEPS = 200
TOL_LOSS = 1e-6
TOL_PARAM = 1e-6


# ====================
# Build
# ====================
def build_our_agent() -> DiscreteBCAgent:
    return DiscreteBCAgent(
        n_actions=N_ACTIONS,
        obs_dim=OBS_DIM,
        learning_rate=LEARNING_RATE,
        beta=BETA,
        hidden_units=HIDDEN_UNITS,
        device=DEVICE,
        seed=SEED,
    )


def build_d3rl_policy() -> tuple[CategoricalPolicy, torch.optim.Optimizer]:
    config = d3rlpy.algos.DiscreteBCConfig(
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        beta=BETA,
    )
    algo = config.create(device=DEVICE)
    algo.create_impl(
        observation_shape=(OBS_DIM,),
        action_size=N_ACTIONS,
    )
    assert algo.impl is not None
    policy = algo.impl.modules.imitator
    optim = algo.impl.modules.optim.optim
    return policy, optim


def copy_our_weights_to_d3rl(our: DiscreteBCAgent, d3_policy: CategoricalPolicy) -> None:
    with torch.no_grad():
        d3_policy._encoder._layers[0].weight.copy_(our.encoder[0].weight)
        d3_policy._encoder._layers[0].bias.copy_(our.encoder[0].bias)
        d3_policy._encoder._layers[2].weight.copy_(our.encoder[2].weight)
        d3_policy._encoder._layers[2].bias.copy_(our.encoder[2].bias)
        d3_policy._fc.weight.copy_(our.policy_head.weight)
        d3_policy._fc.bias.copy_(our.policy_head.bias)


# ====================
# Checks
# ====================
def max_param_diff(our: DiscreteBCAgent, d3_policy: CategoricalPolicy) -> float:
    diffs = [
        (our.encoder[0].weight - d3_policy._encoder._layers[0].weight).abs().max().item(),
        (our.encoder[0].bias - d3_policy._encoder._layers[0].bias).abs().max().item(),
        (our.encoder[2].weight - d3_policy._encoder._layers[2].weight).abs().max().item(),
        (our.encoder[2].bias - d3_policy._encoder._layers[2].bias).abs().max().item(),
        (our.policy_head.weight - d3_policy._fc.weight).abs().max().item(),
        (our.policy_head.bias - d3_policy._fc.bias).abs().max().item(),
    ]
    return float(max(diffs))


def main() -> None:
    print_stage("Init")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    our = build_our_agent()
    d3_policy, d3_optim = build_d3rl_policy()
    copy_our_weights_to_d3rl(our, d3_policy)

    print_stage("Precheck")
    precheck_param_diff = max_param_diff(our, d3_policy)
    print(f"precheck_param_max_abs_diff={precheck_param_diff:.12e}")
    if precheck_param_diff > TOL_LOSS:
        raise SystemExit("FAIL: initial parameters are not aligned.")

    print_stage("Train Compare")
    rng = np.random.default_rng(SEED)

    for step in range(1, TRAIN_STEPS + 1):
        obs_np = rng.standard_normal((BATCH_SIZE, OBS_DIM), dtype=np.float32)
        act_np = rng.integers(0, N_ACTIONS, size=(BATCH_SIZE,), dtype=np.int64)

        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=our.device)
        act_t = torch.as_tensor(act_np, dtype=torch.long, device=our.device)

        our.encoder.train()
        our.policy_head.train()
        our.optim.zero_grad()
        our_total, our_imi, our_reg = our._loss(obs_t, act_t)
        our_total.backward()

        d3_optim.zero_grad()
        d3_loss = compute_discrete_imitation_loss(
            policy=d3_policy,
            x=obs_t,
            action=act_t,
            beta=BETA,
        )
        d3_loss.loss.backward()

        loss_diff = abs(our_total.item() - d3_loss.loss.item())
        imi_diff = abs(our_imi.item() - d3_loss.imitation_loss.item())
        reg_diff = abs(our_reg.item() - d3_loss.regularization_loss.item())

        if max(loss_diff, imi_diff, reg_diff) > TOL_LOSS:
            raise SystemExit(
                "FAIL: loss mismatch at step="
                f"{step} total={loss_diff:.12e} imi={imi_diff:.12e} reg={reg_diff:.12e}"
            )

        our.optim.step()
        d3_optim.step()

        param_diff = max_param_diff(our, d3_policy)
        if param_diff > TOL_PARAM:
            raise SystemExit(
                "FAIL: param mismatch at step="
                f"{step} max_abs_diff={param_diff:.12e}"
            )

        if step % 20 == 0 or step == 1 or step == TRAIN_STEPS:
            print(
                f"step={step}/{TRAIN_STEPS} "
                f"loss={our_total.item():.8f} "
                f"param_max_abs_diff={param_diff:.12e}"
            )

    print_stage("Result")
    print("PASS: our DiscreteBCAgent and d3rl implementation are numerically aligned.")


if __name__ == "__main__":
    main()
