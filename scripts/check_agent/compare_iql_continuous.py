import numpy as np
import torch
import d3rlpy
from d3rlpy.torch_utility import TorchMiniBatch
from ice_offline.agent.iql_continuous import IQLAgentContinuous
from ice_offline.dataset._spec import TorchBuffer
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


# ====================
# Mapping: all_pairs
# ====================
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
# common
# ====================
def build_our_agent() -> IQLAgentContinuous:
    return IQLAgentContinuous(obs_size=OBS_DIM, act_size=ACT_DIM)

def build_d3rl():
    config = d3rlpy.algos.IQLConfig()
    algo = config.create(device=DEVICE)
    algo.create_impl(observation_shape=(OBS_DIM,), action_size=ACT_DIM)
    assert algo.impl is not None
    return algo

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

def _torch_buffer(obs_t, act_t, rew_t, next_obs_t, done_t) -> TorchBuffer:
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
    with torch.no_grad():
        return algo.impl.modules.policy(obs_t).squashed_mu.cpu().numpy()

def d3rl_action_sample_batch(algo, obs_t: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return algo.impl.inner_sample_action(obs_t).cpu().numpy()

def d3rl_action_best_single(algo, obs_t: torch.Tensor) -> np.ndarray:
    return d3rl_action_best_batch(algo, obs_t)[0]

def d3rl_action_sample_single(algo, obs_t: torch.Tensor) -> np.ndarray:
    return d3rl_action_sample_batch(algo, obs_t)[0]
# ====================
# Our Math
# ====================

def _d3rl_loss_pack(d3rl, batch: TorchMiniBatch, obs_t: torch.Tensor):
    q_tpn = d3rl.impl.compute_target(batch)
    critic_obj = d3rl.impl.compute_critic_loss(batch, q_tpn)
    action = d3rl.impl.modules.policy(obs_t)
    actor = d3rl.impl.compute_actor_loss(batch, action).actor_loss
    return critic_obj, actor

def _ref_update_and_collect_params(algo, batch: TorchMiniBatch, step: int, our: IQLAgentContinuous):
    _ = algo.impl.inner_update(batch, step)
    return [x for _, x in _all_pairs(our, algo)]

def _our_update_and_collect_params(
    our: IQLAgentContinuous,
    obs_t: torch.Tensor,
    act_t: torch.Tensor,
    rew_t: torch.Tensor,
    next_obs_t: torch.Tensor,
    done_t: torch.Tensor,
    algo,
):
    our.update(_torch_buffer(obs_t, act_t, rew_t, next_obs_t, done_t))
    return [y for y, _ in _all_pairs(our, algo)]
# ====================
# Compare
# ====================
def init_compare() -> tuple[IQLAgentContinuous, object]:
    print_stage("Init")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    our = build_our_agent()
    algo = build_d3rl()
    with torch.no_grad():
        for our_param, d3_param in _all_pairs(our, algo):
            our_param.copy_(d3_param)
    return our, algo

def compare_act(our: IQLAgentContinuous, algo) -> None:
    print_stage("Act Compare")
    rng = np.random.default_rng(SEED)
    for i in range(1, N_TEST_BATCHES + 1):
        obs_single = sample_observation(rng, 1, OBS_DIM)
        obs_batch = sample_observation(rng, BATCH_SIZE, OBS_DIM)
        # act_single: d3rl best action vs our act
        assert_callback(
            lambda: [d3rl_action_best_single(algo, obs_single)],
            lambda: [our.act(obs_single[0], greedy=True)],
            label=f"act_best_single[{i}]",
            seed=SEED + i,
        )
        # act_batch: d3rl best batch action vs our act_batch
        assert_callback(
            lambda: [d3rl_action_best_batch(algo, obs_batch)],
            lambda: [our.act_batch(obs_batch.cpu().numpy(), greedy=True)],
            label=f"act_best_batch[{i}]",
            seed=SEED + 1000 + i,
        )
        # sample_single: d3rl sampled action vs our act
        sample_seed_single = SEED + 5000 + i
        assert_callback(
            lambda: [d3rl_action_sample_single(algo, obs_single)],
            lambda: [our.act(obs_single[0], greedy=False)],
            label=f"act_sample_single[{i}]",
            seed=sample_seed_single,
        )
        # sample_batch: d3rl sampled batch action vs our act_batch
        sample_seed_batch = SEED + 7000 + i
        assert_callback(
            lambda: [d3rl_action_sample_batch(algo, obs_batch)],
            lambda: [our.act_batch(obs_batch.cpu().numpy(), greedy=False)],
            label=f"act_sample_batch[{i}]",
            seed=sample_seed_batch,
        )
        print(f"batch={i}/{N_TEST_BATCHES} act_and_sample_match=True")

def compare_loss(our: IQLAgentContinuous, algo) -> None:
    print_stage("Loss Compare")
    rng = np.random.default_rng(SEED + 1)
    for i in range(1, N_TEST_BATCHES + 1):
        obs_t, act_t, rew_t, next_obs_t, done_t, batch = sample_transition(
            rng, BATCH_SIZE, OBS_DIM, ACT_DIM
        )
        assert_callback(
            lambda: [_d3rl_loss_pack(algo, batch, obs_t)[0].critic_loss],
            lambda: [our.loss_q(obs_t, act_t, rew_t, next_obs_t, done_t) + our.loss_v(obs_t, act_t)],
            label=f"loss_critic[{i}]",
            seed=SEED + 2000 + i,
        )
        assert_callback(
            lambda: [_d3rl_loss_pack(algo, batch, obs_t)[0].q_loss],
            lambda: [our.loss_q(obs_t, act_t, rew_t, next_obs_t, done_t)],
            label=f"loss_q[{i}]",
            seed=SEED + 2100 + i,
        )
        assert_callback(
            lambda: [_d3rl_loss_pack(algo, batch, obs_t)[0].v_loss],
            lambda: [our.loss_v(obs_t, act_t)],
            label=f"loss_v[{i}]",
            seed=SEED + 2200 + i,
        )
        assert_callback(
            lambda: [_d3rl_loss_pack(algo, batch, obs_t)[1]],
            lambda: [our.loss_actor(obs_t, act_t)],
            label=f"loss_actor[{i}]",
            seed=SEED + 2300 + i,
        )
        print(f"batch={i}/{N_TEST_BATCHES} loss_match=True")

def compare_param(our: IQLAgentContinuous, algo) -> None:
    print_stage("Update Compare")
    rng = np.random.default_rng(SEED + 2)
    for i in range(1, N_TEST_BATCHES + 1):
        obs_t, act_t, rew_t, next_obs_t, done_t, batch = sample_transition(
            rng, BATCH_SIZE, OBS_DIM, ACT_DIM
        )
        assert_callback(
            lambda: _ref_update_and_collect_params(algo, batch, i, our),
            lambda: _our_update_and_collect_params(our, obs_t, act_t, rew_t, next_obs_t, done_t, algo),
            label=f"update[{i}]",
            seed=SEED + 3000 + i,
        )
        print(f"batch={i}/{N_TEST_BATCHES} param_match=True")

def main() -> None:
    our, algo = init_compare()
    compare_act(our, algo)
    compare_loss(our, algo)
    compare_param(our, algo)
    print_stage("Result")
    print("PASS: sample, act, act_batch, loss, and full update params are aligned with d3rl.")
# ====================
# __main__
# ====================

if __name__ == "__main__":
    main()
