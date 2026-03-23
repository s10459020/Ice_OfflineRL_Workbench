import gymnasium as gym

from ice_offline.strategy import data
from ice_offline.tools import print_stage


# ====================
# Script Main
# ====================
print_stage("Replay: Online")
steps_online = data.online_view(
    env=gym.make("BabyAI-OneRoomS8-v0"),
    dataset="test_collect-v0",
    max_episodes=3,
    seed=123,
    random_episode=False,
    render_interval=None,
    print_interval=10,
)
print(f"online_steps={steps_online}")
