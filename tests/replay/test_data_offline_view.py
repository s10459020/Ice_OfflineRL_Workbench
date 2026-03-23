import gymnasium as gym

from ice_offline.strategy import data
from ice_offline.tools import print_stage


# ====================
# Script Main
# ====================
# ---- Replay: sequential episodes ----
print_stage("Replay: Sequential")
steps_seq = data.offline_view(
    dataset="test_collect-v0",
    max_episodes=3,
    print_interval=10,
)
print(f"sequential_steps={steps_seq}")