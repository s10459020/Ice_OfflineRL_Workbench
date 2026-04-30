
import gymnasium as gym
import numpy as np

from ice_offline.agent import DiscretePolicyGradientAgent
from ice_offline.agent._interface import model_ref
from ice_offline.tools.printer import print_stage

ENV_ID = "CartPole-v1"
RUNNER_ID = "pg_discrete"
LOAD_STEP: int | None = None
MAX_EPISODES = 1_000_000
SAVE_EVERY_EPISODES = 100_000
LOG_EVERY_EPISODES = 100


def main() -> None:
    print_stage("Init")
    env = gym.make(ENV_ID)
    obs_dim = int(np.prod(env.observation_space.shape))

    agent = DiscretePolicyGradientAgent(
        n_actions=env.action_space.n,
        obs_dim=obs_dim,
    )
    if LOAD_STEP is not None:
        agent = DiscretePolicyGradientAgent.load(RUNNER_ID, LOAD_STEP)
        print(
            f"load step={LOAD_STEP} "
            f"path={model_path(RUNNER_ID, LOAD_STEP, '.npz')}"
        )

    print_stage("Train")
    total_steps = 0
    log_episode_count = 0
    log_steps_sum = 0
    log_reward_sum = 0.0
    try:
        for episode in range(1, MAX_EPISODES + 1):
            obs, _ = env.reset(seed=42 + episode)
            done = False
            truncated = False
            episode_reward = 0.0
            episode_steps = 0

            while not (done or truncated):
                action = agent.act(obs)
                next_obs, reward, done, truncated, _ = env.step(action)
                agent.record_step(obs, action, reward)

                obs = next_obs
                episode_reward += reward
                episode_steps += 1
                total_steps += 1

            agent.update()
            log_episode_count += 1
            log_steps_sum += episode_steps
            log_reward_sum += episode_reward

            if episode % SAVE_EVERY_EPISODES == 0:
                save_path = agent.save(RUNNER_ID, total_steps)
                print(
                    f"save episode={episode} "
                    f"step={total_steps} "
                    f"path={save_path}"
                )

            if episode % LOG_EVERY_EPISODES == 0:
                avg_episode_steps = log_steps_sum / log_episode_count
                avg_episode_reward = log_reward_sum / log_episode_count
                print(
                    f"episode={episode}/{MAX_EPISODES} "
                    f"avg_episode_steps={avg_episode_steps:.3f} "
                    f"avg_episode_reward={avg_episode_reward:.3f} "
                    f"total_steps={total_steps}"
                )
                log_episode_count = 0
                log_steps_sum = 0
                log_reward_sum = 0.0

        print_stage("Summary")
        save_path = agent.save(RUNNER_ID, total_steps)
        print(f"final_save step={total_steps} path={save_path}")
        print(f"env_id={ENV_ID}")
        print(f"runner_id={RUNNER_ID}")
        print(f"episodes={MAX_EPISODES}")
        print(f"total_steps={total_steps}")
    finally:
        env.close()


if __name__ == "__main__":
    main()



