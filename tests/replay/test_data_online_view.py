from ice_offline.strategy import data
from ice_offline.tools import print_stage


# ====================
# Script Main
# ====================
print_stage("Replay: Online")
steps_online = data.online_view(
    dataset="test_convert_fullobs-v0",
    max_episodes=300,
    seed=123,
    random_episode=False,
    render_interval=1,
    print_interval=10,
)
print(f"online_steps={steps_online}")
