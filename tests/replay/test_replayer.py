from ice_offline.strategy import replayer
from ice_offline.tools import print_stage


# ====================
# Script Main
# ====================
# ---- Replay: sequential episodes ----
print_stage("Replay: Sequential")
steps_seq = replayer.run(
    dataset="test_collector-v0",
    max_episodes=3,
    sample_flag=False,
    print_interval=10,
)
print(f"sequential_steps={steps_seq}")

# ---- Replay: sampled episodes ----
print_stage("Replay: Sampled")
steps_sample = replayer.run(
    dataset="test_collector-v0",
    max_episodes=3,
    sample_flag=True,
    seed=42,
    print_interval=10,
)
print(f"sampled_steps={steps_sample}")
