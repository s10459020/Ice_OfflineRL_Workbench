import d3rlpy
import minari

print('imports ok')

# 1) 載入資料集（首次會下載）
dataset = minari.load_dataset("D4RL/antmaze/umaze-v1", download=True)

# 2) 取回對應執行環境（這裡就會是 AntMaze_UMaze-v4）
env = dataset.recover_environment()

obs, info = env.reset()
for _ in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()