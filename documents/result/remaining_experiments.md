# Remaining Experiments

更新時間：2026-07-24

這份清單依 `documents/result/agent_dataset_versions.csv` 與目前 `scripts/trains.py`、`scripts/test_all.py` 的任務狀態整理。速查表已重新生成；`scripts/test_all.py` 也已移除已完成的 refresh 任務。

## 可直接 Test

目前 `scripts/test_all.py` 只保留已具備 train-min checkpoint、但 noise test 尚未完成的非 ASPL/SCASPL 任務。

- `noise_init`：`scas_n` on `walker2d_replay_medium`，noise = `5e-2`, `1e-1`, `5e-1`, `1e0`
- `noise_action`：`scas_n` on `walker2d_replay_medium`，noise = `5e-2`, `1e-1`, `5e-1`, `1e0`
- `noise_state`：`scas_n` on `walker2d_replay_medium`，noise = `5e-4`, `1e-3`, `5e-3`, `1e-2`

## 仍需 Train

這些任務在速查表內還沒有完整 train eval，或尚未達到指定訓練步數。

- `base / scas_n`：`hopper_d4rl_expert`, `hopper_d4rl_hybrid`, `hopper_replay_medium`, `hopper_replay_expert`
- `base / scas_n`：`halfcheetah_d4rl_medium`, `halfcheetah_d4rl_expert`, `halfcheetah_d4rl_hybrid`, `halfcheetah_replay_medium`, `halfcheetah_replay_expert`
- `base / scas_n`：`walker2d_d4rl_medium`, `walker2d_d4rl_expert`, `walker2d_d4rl_hybrid`, `walker2d_replay_medium` 目前只有 `200000 / 500000`
- `hybrid_random / scas_n`：`walker2d_random_expert_1`, `walker2d_random_expert_3`, `walker2d_random_expert_5`, `walker2d_random_expert_7`, `walker2d_random_expert_9`

## 仍需 Train-Min

這些任務在 `scripts/trains.py` 已列為 train-min，部分會被 agent checkpoint 阻塞到 train 完成後才能跑。

- `base_train_min / scas_n`：hopper + halfcheetah 的 10 個資料集，其中 `hopper_d4rl_medium` 可直接跑，其餘多數需等待新的 `500000` checkpoint
- `hybrid_random_train_min / scas_n`：5 個 hybrid random 資料集，需等待對應 train 完成
- `hybrid_random_train_min / scc_n`：5 個 hybrid random 資料集，目前缺 train-min

## 速查表缺口但未排入任務

這些仍在 `documents/result/*.csv` 內顯示缺口，但目前沒有排入 `scripts/trains.py` 或 `scripts/test_all.py`。

- `stability_scaspl / scaspl_n / walker2d_replay_medium`：目前只有 `200000 / 500000`
- `base / scaspl_n / walker2d_replay_medium`：目前只有 `200000 / 500000`
- `hybrid_random / scaspl_n / walker2d_random_expert_9`：缺 eval
- `noise_init / scaspl_n`：6 個 noise test 尚缺，但 SCASPL 系列目前未排入統整 test
- `noise_state / aspl_c`：7 個 noise test 尚缺，但 ASPL 系列目前未排入統整 test

## 可信度警告

速查表內仍有不少 `!` 標記，代表目前 agent 檔案更新時間晚於採用的 eval。這些分數可以作為舊結果參考，但若要放入正式論文表格，仍需按目前最新 agent 定義重新完成 train / train-min / test。

