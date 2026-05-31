﻿import numpy as np
import torch
import d3rlpy
from d3rlpy.torch_utility import TorchMiniBatch
from ice_offline.agent.discrete.cql_discrete import CQLAgentDiscrete
from ice_offline.dataset._spec import TorchBuffer
from ice_offline.tools.printer import print_stage
from _lib import assert_callback
# ====================
# Config
# ====================

OBS_DIM = 8
N_ACTIONS = 4
DEVICE = "cpu"
SEED = 42
BATCH_SIZE = 64
N_TEST_BATCHES = 30
# ====================
# Mapping: all_pairs
# ====================

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
# common
# ====================

def build_our_agent() -> CQLAgentDiscrete:
    return CQLAgentDiscrete(obs_size=OBS_DIM, act_size=N_ACTIONS)

def build_d3rl():
    config = d3rlpy.algos.DiscreteCQLConfig()
    algo = config.create(device=DEVICE)
    algo.create_impl(observation_shape=(OBS_DIM,), action_size=N_ACTIONS)
    assert algo.impl is not None
    return algo

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

def _torch_buffer(
    obs_t: torch.Tensor,
    act_t: torch.Tensor,
    rew_t: torch.Tensor,
    next_obs_t: torch.Tensor,
    done_t: torch.Tensor,
) -> TorchBuffer:
    return TorchBuffer(
        obs_list=obs_t,
        next_obs_list=next_obs_t,
        act_list=act_t,
        rew_list=rew_t,
        done_list=done_t,
    )
# ====================
# Ref Math
# ====================

def d3rl_action_best_batch(algo, obs_t: torch.Tensor) -> np.ndarray:
    return algo.impl.inner_predict_best_action(obs_t).cpu().numpy()

def d3rl_action_best_single(algo, obs_t: torch.Tensor) -> np.ndarray:
    return d3rl_action_best_batch(algo, obs_t)[0]

def _d3rl_loss_pack(algo, obs_t, act_t, rew_t, next_obs_t, done_t):
    batch = _torch_batch(obs_t, act_t, rew_t, next_obs_t, done_t)
    q_tpn = algo.impl.compute_target(batch)
    return algo.impl.compute_loss(batch, q_tpn)

def _d3rl_loss_td(algo, obs_t, act_t, rew_t, next_obs_t, done_t) -> torch.Tensor:
    return _d3rl_loss_pack(algo, obs_t, act_t, rew_t, next_obs_t, done_t).td_loss

def _d3rl_loss_conservative(algo, obs_t, act_t, rew_t, next_obs_t, done_t) -> torch.Tensor:
    return _d3rl_loss_pack(algo, obs_t, act_t, rew_t, next_obs_t, done_t).conservative_loss

def _d3rl_loss_critic(algo, obs_t, act_t, rew_t, next_obs_t, done_t) -> torch.Tensor:
    return _d3rl_loss_pack(algo, obs_t, act_t, rew_t, next_obs_t, done_t).loss
# ====================
# Our Math
# ====================

def _ref_update_and_collect_params(d3rl, batch: TorchMiniBatch, step: int, our_agent: CQLAgentDiscrete):
    _ = d3rl.impl.inner_update(batch, step)
    return [x for _, x in _all_pairs(our_agent, d3rl)]

def _our_update_and_collect_params(
    our_agent: CQLAgentDiscrete,
    obs_t: torch.Tensor,
    act_t: torch.Tensor,
    rew_t: torch.Tensor,
    next_obs_t: torch.Tensor,
    done_t: torch.Tensor,
    d3rl,
):
    our_agent.update(_torch_buffer(obs_t, act_t, rew_t, next_obs_t, done_t))
    return [y for y, _ in _all_pairs(our_agent, d3rl)]
# ====================
# Compare
# ====================
def init_compare() -> tuple[CQLAgentDiscrete, object]:
    print_stage("Init")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    our_agent = build_our_agent()
    d3rl = build_d3rl()
    with torch.no_grad():
        for our_param, d3_param in _all_pairs(our_agent, d3rl):
            our_param.copy_(d3_param)
    return our_agent, d3rl

def compare_act(our_agent: CQLAgentDiscrete, d3rl) -> None:
    print_stage("Act Compare")
    rng = np.random.default_rng(SEED)
    for i in range(1, N_TEST_BATCHES + 1):
        obs_single = sample_observation(rng, 1, OBS_DIM)
        obs_batch = sample_observation(rng, BATCH_SIZE, OBS_DIM)
        # act_single: d3rl best action vs our act
        assert_callback(
            lambda: [d3rl_action_best_single(d3rl, obs_single)],
            lambda: [our_agent.act(obs_single[0], epsilon=0.0)],
            label=f"act_single[{i}]",
            seed=SEED + i,
        )
        # act_batch: d3rl best batch action vs our act_batch
        assert_callback(
            lambda: [d3rl_action_best_batch(d3rl, obs_batch)],
            lambda: [our_agent.act_batch(obs_batch.cpu().numpy(), epsilon=0.0)],
            label=f"act_batch[{i}]",
            seed=SEED + 1000 + i,
        )
        print(f"batch={i}/{N_TEST_BATCHES} action_match=True")

def compare_loss(our_agent: CQLAgentDiscrete, d3rl) -> None:
    print_stage("Loss Compare")
    rng = np.random.default_rng(SEED + 1)
    for i in range(1, N_TEST_BATCHES + 1):
        obs_t, act_t, rew_t, next_obs_t, done_t = sample_transition(
            rng, BATCH_SIZE, OBS_DIM, N_ACTIONS
        )
        assert_callback(
            lambda: [_d3rl_loss_td(d3rl, obs_t, act_t, rew_t, next_obs_t, done_t)],
            lambda: [our_agent.loss_td(obs_t, act_t, rew_t, next_obs_t, done_t)],
            label=f"loss_td[{i}]",
            seed=SEED + 2000 + i,
        )
        assert_callback(
            lambda: [_d3rl_loss_conservative(d3rl, obs_t, act_t, rew_t, next_obs_t, done_t)],
            lambda: [our_agent.loss_conservative(obs_t, act_t)],
            label=f"loss_conservative[{i}]",
            seed=SEED + 2100 + i,
        )
        assert_callback(
            lambda: [_d3rl_loss_critic(d3rl, obs_t, act_t, rew_t, next_obs_t, done_t)],
            lambda: [our_agent.loss_critic(obs_t, act_t, rew_t, next_obs_t, done_t)],
            label=f"loss_critic[{i}]",
            seed=SEED + 2200 + i,
        )
        print(f"batch={i}/{N_TEST_BATCHES} loss_match=True")

def compare_param(our_agent: CQLAgentDiscrete, d3rl) -> None:
    print_stage("Update Compare")
    rng = np.random.default_rng(SEED + 2)
    for i in range(1, N_TEST_BATCHES + 1):
        obs_t, act_t, rew_t, next_obs_t, done_t = sample_transition(
            rng, BATCH_SIZE, OBS_DIM, N_ACTIONS
        )
        batch = _torch_batch(obs_t, act_t, rew_t, next_obs_t, done_t)
        assert_callback(
            lambda: _ref_update_and_collect_params(d3rl, batch, i, our_agent),
            lambda: _our_update_and_collect_params(our_agent, obs_t, act_t, rew_t, next_obs_t, done_t, d3rl),
            label=f"update[{i}]",
            seed=SEED + 3000 + i,
        )
        print(f"batch={i}/{N_TEST_BATCHES} param_match=True")

def main() -> None:
    our_agent, d3rl = init_compare()
    compare_act(our_agent, d3rl)
    compare_loss(our_agent, d3rl)
    compare_param(our_agent, d3rl)
    print_stage("Result")
    print("PASS: act, act_batch, loss, and full update params are aligned with d3rl.")
# ====================
# __main__
# ====================

if __name__ == "__main__":
    main()
