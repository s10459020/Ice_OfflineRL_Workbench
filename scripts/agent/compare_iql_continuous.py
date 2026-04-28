from __future__ import annotations

import numpy as np
import torch

import d3rlpy
from d3rlpy.torch_utility import TorchMiniBatch

from ice_offline.agent.iql_agent_continuous import IQLAgentContinuous
from ice_offline.tools.printer import print_stage

OBS_DIM = 8
ACT_DIM = 3
DEVICE = "cpu"
SEED = 42
BATCH_SIZE = 64
N_TEST_BATCHES = 30


# ====================
# 1) Flow Function
# ====================
def build_our_agent() -> IQLAgentContinuous:
    return IQLAgentContinuous(obs_size=OBS_DIM, act_size=ACT_DIM)

def build_d3rl():
    config = d3rlpy.algos.IQLConfig()
    algo = config.create(device=DEVICE)
    algo.create_impl(observation_shape=(OBS_DIM,), action_size=ACT_DIM)
    assert algo.impl is not None
    return algo

def copy_d3rl_weights_to_our(algo, our: IQLAgentContinuous) -> None:
    with torch.no_grad():
        for our_param, d3_param in _all_pairs(our, algo):
            our_param.copy_(d3_param)

def sample_observation(rng: np.random.Generator, batch: int, size: int) -> torch.Tensor:
    return torch.as_tensor(rng.standard_normal((batch, size)), dtype=torch.float32, device=DEVICE)

def sample_transition(
    rng: np.random.Generator, batch: int, obs_size: int, act_size: int
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, TorchMiniBatch
]:
    obs_t = torch.as_tensor(rng.standard_normal((batch, obs_size)), dtype=torch.float32, device=DEVICE)
    act_t = torch.as_tensor(rng.standard_normal((batch, act_size)), dtype=torch.float32, device=DEVICE)
    rew_t = torch.as_tensor(rng.standard_normal((batch, 1)), dtype=torch.float32, device=DEVICE)
    next_obs_t = torch.as_tensor(rng.standard_normal((batch, obs_size)), dtype=torch.float32, device=DEVICE)
    done_t = torch.as_tensor(rng.integers(0, 2, size=(batch, 1)), dtype=torch.float32, device=DEVICE)
    batch_t = _torch_batch(obs_t, act_t, rew_t, next_obs_t, done_t)
    return obs_t, act_t, rew_t, next_obs_t, done_t, batch_t

def _torch_batch(obs_t, act_t, rew_t, next_obs_t, done_t) -> TorchMiniBatch:
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
    with torch.no_grad():
        return algo.impl.modules.policy(obs_t).squashed_mu.cpu().numpy()

def d3rl_action_sample_batch(algo, obs_t: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return algo.impl.inner_sample_action(obs_t).cpu().numpy()

def _our_losses(
    our: IQLAgentContinuous,
    obs_t: torch.Tensor,
    act_t: torch.Tensor,
    rew_t: torch.Tensor,
    next_obs_t: torch.Tensor,
    done_t: torch.Tensor,
) -> torch.Tensor:
    q_loss = our._loss_q(obs_t, act_t, rew_t, next_obs_t, done_t)
    v_loss = our._loss_v(obs_t, act_t)
    critic = q_loss + v_loss
    actor = our._loss_actor(obs_t, act_t)
    return torch.stack([critic, q_loss, v_loss, actor])

def _d3rl_losses(d3rl, batch: TorchMiniBatch, obs_t: torch.Tensor) -> torch.Tensor:
    q_tpn = d3rl.impl.compute_target(batch)
    critic_obj = d3rl.impl.compute_critic_loss(batch, q_tpn)
    action = d3rl.impl.modules.policy(obs_t)
    actor = d3rl.impl.compute_actor_loss(batch, action).actor_loss
    return torch.stack([critic_obj.critic_loss, critic_obj.q_loss, critic_obj.v_loss, actor])

def _all_pairs(our: IQLAgentContinuous, algo):
    d3_policy = algo.impl.modules.policy
    d3_q1 = algo.impl.modules.q_funcs[0]
    d3_q2 = algo.impl.modules.q_funcs[1]
    d3_t1 = algo.impl.modules.targ_q_funcs[0]
    d3_t2 = algo.impl.modules.targ_q_funcs[1]
    d3_v = algo.impl.modules.value_func
    return [
        (our.actor.hidden[0].weight, d3_policy._encoder._layers[0].weight),
        (our.actor.hidden[0].bias, d3_policy._encoder._layers[0].bias),
        (our.actor.hidden[2].weight, d3_policy._encoder._layers[2].weight),
        (our.actor.hidden[2].bias, d3_policy._encoder._layers[2].bias),
        (our.actor.mean_head.weight, d3_policy._mu.weight),
        (our.actor.mean_head.bias, d3_policy._mu.bias),
        (our.actor.logstd, d3_policy._logstd),
        (our.critic.q1.network[0].weight, d3_q1._encoder._layers[0].weight),
        (our.critic.q1.network[0].bias, d3_q1._encoder._layers[0].bias),
        (our.critic.q1.network[2].weight, d3_q1._encoder._layers[2].weight),
        (our.critic.q1.network[2].bias, d3_q1._encoder._layers[2].bias),
        (our.critic.q1.network[4].weight, d3_q1._fc.weight),
        (our.critic.q1.network[4].bias, d3_q1._fc.bias),
        (our.critic.q2.network[0].weight, d3_q2._encoder._layers[0].weight),
        (our.critic.q2.network[0].bias, d3_q2._encoder._layers[0].bias),
        (our.critic.q2.network[2].weight, d3_q2._encoder._layers[2].weight),
        (our.critic.q2.network[2].bias, d3_q2._encoder._layers[2].bias),
        (our.critic.q2.network[4].weight, d3_q2._fc.weight),
        (our.critic.q2.network[4].bias, d3_q2._fc.bias),
        (our.critic.targ_q1.network[0].weight, d3_t1._encoder._layers[0].weight),
        (our.critic.targ_q1.network[0].bias, d3_t1._encoder._layers[0].bias),
        (our.critic.targ_q1.network[2].weight, d3_t1._encoder._layers[2].weight),
        (our.critic.targ_q1.network[2].bias, d3_t1._encoder._layers[2].bias),
        (our.critic.targ_q1.network[4].weight, d3_t1._fc.weight),
        (our.critic.targ_q1.network[4].bias, d3_t1._fc.bias),
        (our.critic.targ_q2.network[0].weight, d3_t2._encoder._layers[0].weight),
        (our.critic.targ_q2.network[0].bias, d3_t2._encoder._layers[0].bias),
        (our.critic.targ_q2.network[2].weight, d3_t2._encoder._layers[2].weight),
        (our.critic.targ_q2.network[2].bias, d3_t2._encoder._layers[2].bias),
        (our.critic.targ_q2.network[4].weight, d3_t2._fc.weight),
        (our.critic.targ_q2.network[4].bias, d3_t2._fc.bias),
        (our.v.network[0].weight, d3_v._encoder._layers[0].weight),
        (our.v.network[0].bias, d3_v._encoder._layers[0].bias),
        (our.v.network[2].weight, d3_v._encoder._layers[2].weight),
        (our.v.network[2].bias, d3_v._encoder._layers[2].bias),
        (our.v.network[4].weight, d3_v._fc.weight),
        (our.v.network[4].bias, d3_v._fc.bias),
    ]

# ====================
# 3) main Function
# ====================
def main() -> None:
    print_stage("Init")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    our = build_our_agent()
    algo = build_d3rl()
    copy_d3rl_weights_to_our(algo, our)


    print_stage("Act Compare")
    rng = np.random.default_rng(SEED)
    for i in range(1, N_TEST_BATCHES + 1):
        obs_t = sample_observation(rng, BATCH_SIZE, OBS_DIM)
        d3_act = d3rl_action_best_batch(algo, obs_t)
        our_act = our.act_batch(obs_t, greedy=True)
        _assert_equal([(d3_act, our_act)])

        torch.manual_seed(SEED + 5000 + i)
        d3_sample = d3rl_action_sample_batch(algo, obs_t)
        torch.manual_seed(SEED + 5000 + i)
        our_sample = our.act_batch(obs_t, greedy=False)
        _assert_equal([(d3_sample, our_sample)])

        torch.manual_seed(SEED + 7000 + i)
        single_sample_batch = our.act(obs_t[0].cpu().numpy(), greedy=False)
        torch.manual_seed(SEED + 7000 + i)
        single_sample = our.act(obs_t[0].cpu().numpy(), greedy=False)
        _assert_equal([
            (single_sample, single_sample_batch),
        ])
        print(f"batch={i}/{N_TEST_BATCHES} action_match=True")


    print_stage("Loss Compare")
    rng = np.random.default_rng(SEED + 1)
    for i in range(1, N_TEST_BATCHES + 1):
        obs_t, act_t, rew_t, next_obs_t, done_t, batch = sample_transition(
            rng, BATCH_SIZE, OBS_DIM, ACT_DIM
        )

        our_losses = _our_losses(our, obs_t, act_t, rew_t, next_obs_t, done_t)
        d3rl_losses = _d3rl_losses(algo, batch, obs_t)

        _assert_equal([(d3rl_losses, our_losses)])
        print(f"batch={i}/{N_TEST_BATCHES} loss_match=True")


    print_stage("Update Compare")
    rng = np.random.default_rng(SEED + 2)
    for i in range(1, N_TEST_BATCHES + 1):
        obs_t, act_t, rew_t, next_obs_t, done_t, batch = sample_transition(
            rng, BATCH_SIZE, OBS_DIM, ACT_DIM
        )

        _ = algo.impl.inner_update(batch, i)
        our.update(
            {
                "obs": obs_t,
                "act": act_t,
                "rew": rew_t,
                "next_obs": next_obs_t,
                "done": done_t,
            }
        )

        _assert_equal(_all_pairs(our, algo))
        print(f"batch={i}/{N_TEST_BATCHES} param_match=True")

    print_stage("Result")
    print("PASS: act, loss, and full update params are aligned with d3rl.")


if __name__ == "__main__":
    main()
