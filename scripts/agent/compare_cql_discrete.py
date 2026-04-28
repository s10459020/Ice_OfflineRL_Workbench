from __future__ import annotations

import numpy as np
import torch

import d3rlpy
from d3rlpy.torch_utility import TorchMiniBatch

from ice_offline.agent.cql_discrete import CQLAgentDiscrete
from ice_offline.tools.printer import print_stage

OBS_DIM = 8
N_ACTIONS = 4
DEVICE = "cpu"
SEED = 42
BATCH_SIZE = 64
N_TEST_BATCHES = 30


# ====================
# 1) Flow Function
# ====================
def build_our_agent() -> CQLAgentDiscrete:
    return CQLAgentDiscrete(obs_size=OBS_DIM, act_size=N_ACTIONS)

def build_d3rl():
    config = d3rlpy.algos.DiscreteCQLConfig()
    algo = config.create(device=DEVICE)
    algo.create_impl(observation_shape=(OBS_DIM,), action_size=N_ACTIONS)
    assert algo.impl is not None
    return algo

def copy_d3rl_weights_to_our(algo, our_agent: CQLAgentDiscrete) -> None:
    with torch.no_grad():
        for our_param, d3_param in _all_pairs(our_agent, algo):
            our_param.copy_(d3_param)

def sample_observation(rng: np.random.Generator, batch: int, size: int) -> torch.Tensor:
    return torch.as_tensor(rng.standard_normal((batch, size)), dtype=torch.float32, device=DEVICE)

def sample_transition(
    rng: np.random.Generator, batch: int, obs_size: int, action_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    obs_t = sample_observation(rng, batch, obs_size)
    act_t = torch.as_tensor(rng.integers(0, action_size, size=(batch,)), device=DEVICE)
    rew_t = torch.as_tensor(rng.standard_normal((batch, 1)), dtype=torch.float32, device=DEVICE)
    next_obs_t = sample_observation(rng, batch, obs_size)
    done_t = torch.as_tensor(rng.integers(0, 2, size=(batch, 1)), dtype=torch.float32, device=DEVICE)
    return obs_t, act_t, rew_t, next_obs_t, done_t

def _torch_batch(
    obs_t: torch.Tensor,
    act_t: torch.Tensor,
    rew_t: torch.Tensor,
    next_obs_t: torch.Tensor,
    done_t: torch.Tensor,
) -> TorchMiniBatch:
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

def _assert_equal(pairs) -> None:
    max_diff = 0.0
    for x, y in pairs:
        if x is None or y is None:
            if x is None and y is None:
                continue
            raise SystemExit("FAIL: mismatch, one side is None")
        if torch.is_tensor(x) and torch.is_tensor(y):
            max_diff = max(max_diff, float((x - y).abs().max().item()))
        else:
            max_diff = max(max_diff, float(np.abs(x - y).max()))
    if max_diff != 0.0:
        raise SystemExit(f"FAIL: mismatch, max_abs_diff={max_diff:.12e}")


# ====================
# 2) Behavior Function
# ====================
def d3rl_action_best_batch(algo, obs_t: torch.Tensor) -> np.ndarray:
    return algo.impl.inner_predict_best_action(obs_t).cpu().numpy()

def _d3rl_losses(algo, obs_t, act_t, rew_t, next_obs_t, done_t) -> torch.Tensor:
    q_values = algo.impl._q_func_forwarder.compute_expected_q(obs_t)
    next_q_values = algo.impl._q_func_forwarder.compute_expected_q(next_obs_t)
    d3_action = next_q_values.argmax(dim=1)
    q_tpn = algo.impl._targ_q_func_forwarder.compute_target(
        next_obs_t, d3_action, reduction="min"
    )
    td_loss = algo.impl._q_func_forwarder.compute_error(
        observations=obs_t,
        actions=act_t.long(),
        rewards=rew_t,
        target=q_tpn,
        terminals=done_t,
        gamma=algo.impl._gamma,
    )
    logsumexp = torch.logsumexp(q_values, dim=1, keepdim=True)
    one_hot = torch.nn.functional.one_hot(act_t.view(-1), num_classes=N_ACTIONS).float()
    data_values = (q_values * one_hot).sum(dim=1, keepdim=True)
    conservative = (logsumexp - data_values).mean()
    total = td_loss + algo.impl._alpha * conservative
    return torch.stack([total, td_loss, conservative])

def _our_losses(our_agent: CQLAgentDiscrete, obs_t, act_t, rew_t, next_obs_t, done_t) -> torch.Tensor:
    td = our_agent._loss_td(obs_t, act_t, rew_t, next_obs_t, done_t)
    cql = our_agent._loss_cql(obs_t, act_t)
    total = our_agent._loss(obs_t, act_t, rew_t, next_obs_t, done_t)
    return torch.stack([total, td, cql])

def _all_pairs(our_agent: CQLAgentDiscrete, algo):
    d3_q = algo.impl.modules.q_funcs[0]
    d3_targ_q = algo.impl.modules.targ_q_funcs[0]
    our_q = our_agent.critic._q
    our_tq = our_agent.critic._targ_q
    return [
        (our_q.network[0].weight, d3_q._encoder._layers[0].weight),
        (our_q.network[0].bias, d3_q._encoder._layers[0].bias),
        (our_q.network[2].weight, d3_q._encoder._layers[2].weight),
        (our_q.network[2].bias, d3_q._encoder._layers[2].bias),
        (our_q.network[4].weight, d3_q._fc.weight),
        (our_q.network[4].bias, d3_q._fc.bias),
        (our_tq.network[0].weight, d3_targ_q._encoder._layers[0].weight),
        (our_tq.network[0].bias, d3_targ_q._encoder._layers[0].bias),
        (our_tq.network[2].weight, d3_targ_q._encoder._layers[2].weight),
        (our_tq.network[2].bias, d3_targ_q._encoder._layers[2].bias),
        (our_tq.network[4].weight, d3_targ_q._fc.weight),
        (our_tq.network[4].bias, d3_targ_q._fc.bias),
    ]


# ====================
# 3) main Function
# ====================
def main() -> None:
    print_stage("Init")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    our_agent = build_our_agent()
    d3rl = build_d3rl()
    copy_d3rl_weights_to_our(d3rl, our_agent)


    print_stage("Act Compare")
    rng = np.random.default_rng(SEED)
    for i in range(1, N_TEST_BATCHES + 1):
        obs_t = sample_observation(rng, BATCH_SIZE, OBS_DIM)
        d3_act = d3rl_action_best_batch(d3rl, obs_t)
        our_act = our_agent.act_batch(obs_t, epsilon=0.0)
        _assert_equal([(d3_act, our_act)])
        print(f"batch={i}/{N_TEST_BATCHES} action_match=True")


    print_stage("Loss Compare")
    rng = np.random.default_rng(SEED + 1)
    for i in range(1, N_TEST_BATCHES + 1):
        obs_t, act_t, rew_t, next_obs_t, done_t = sample_transition(
            rng, BATCH_SIZE, OBS_DIM, N_ACTIONS
        )
        d3rl_losses = _d3rl_losses(d3rl, obs_t, act_t, rew_t, next_obs_t, done_t)
        our_losses = _our_losses(our_agent, obs_t, act_t, rew_t, next_obs_t, done_t)
        _assert_equal([(d3rl_losses, our_losses)])
        print(f"batch={i}/{N_TEST_BATCHES} loss_match=True")


    print_stage("Update Compare")
    rng = np.random.default_rng(SEED + 2)
    for i in range(1, N_TEST_BATCHES + 1):
        obs_t, act_t, rew_t, next_obs_t, done_t = sample_transition(
            rng, BATCH_SIZE, OBS_DIM, N_ACTIONS
        )
        batch = _torch_batch(obs_t, act_t, rew_t, next_obs_t, done_t)

        our_agent.update(
            {
                "obs": obs_t,
                "act": act_t,
                "rew": rew_t,
                "next_obs": next_obs_t,
                "done": done_t,
            },
            i,
        )
        _ = d3rl.impl.inner_update(batch, i)

        _assert_equal(_all_pairs(our_agent, d3rl))
        print(f"batch={i}/{N_TEST_BATCHES} param_match=True")

    print_stage("Result")
    print("PASS: act, loss, and full update params are aligned with d3rl.")


if __name__ == "__main__":
    main()

