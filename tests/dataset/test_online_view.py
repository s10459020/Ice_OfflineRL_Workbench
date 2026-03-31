import minari

from ice_offline.env.common import insert_render_quiet_innermost
from ice_offline.env.replay.state_inject_wrapper import StateInjectWrapper
from ice_offline.tools import print_stage


# ====================
# Config
# ====================
DATASET_ID = "test_collect-v0"


# ====================
# Replay
# ====================
print_stage("Replay")
dataset = minari.load_dataset(DATASET_ID)
env = dataset.recover_environment(eval_env=True, render_mode="human")
env = insert_render_quiet_innermost(env)
env = StateInjectWrapper(env, dataset=dataset, random_episode=False)

steps_online = 0
for episode in range(1, dataset.total_episodes + 1):
    _, _ = env.reset()
    env.render()

    episode_step = 0
    while True:
        _, reward, terminated, truncated, info = env.step(None)
        episode_step += 1
        steps_online += 1

        env.render()

        print(
            f"step={steps_online} episode={episode} episode_step={episode_step} "
            f"action={info.get('action')} reward={reward:.3f} "
            f"terminated={terminated} truncated={truncated}"
        )

        if terminated or truncated:
            break

env.close()
print(f"online_steps={steps_online}")
