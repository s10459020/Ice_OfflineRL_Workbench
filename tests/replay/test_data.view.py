from ice_offline.strategy import data
from ice_offline.tools import print_stage


# ====================
# Script Main
# ====================
# ---- Replay: sequential episodes ----
print_stage("Replay: Sequential")
steps_seq = data.view(
    dataset="test_collector-v0",
    max_episodes=3,
    print_interval=10,
)
print(f"sequential_steps={steps_seq}")

# ---- Replay: more episodes ----
print_stage("Replay: More")
steps_more = data.view(
    dataset="test_collector-v0",
    max_episodes=5,
    print_interval=10,
)
print(f"more_steps={steps_more}")
