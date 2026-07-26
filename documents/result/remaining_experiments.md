# Remaining Experiments

更新時間：2026-07-24

這份清單依 `documents/result/agent_dataset_versions.csv` 與目前 `scripts/trains.py`、`scripts/test_all.py` 的任務狀態整理。八個代表模型中的 ASPL 代表已改為 `aspl`，不再使用 `aspl_c`。

## 可直接 Test

目前 `scripts/test_all.py` 只保留已具備 train-min checkpoint、但 noise test 尚未完成的 ASPL 任務。

- `noise_init / aspl`：12 個 walker2d noise test
- `noise_action / aspl`：12 個 walker2d noise test
- `noise_state / aspl`：12 個 walker2d noise test

## 仍需 Train

目前 `scripts/trains.py` 已排入缺少正式 train checkpoint 的 ASPL 任務。

- `base_train / aspl`：5 個 halfcheetah 資料集，訓練到 `200000`
- `hybrid_random_train / aspl`：5 個 hybrid random 資料集，訓練到 `200000`

## 仍需 Train-Min

目前 `scripts/trains.py` 已排入可補齊或會在前置 train 完成後補齊的 ASPL train-min 任務。

- `base_train_min / aspl`：5 個 hopper 資料集，補 `200000..220000`
- `base_train_min / aspl`：5 個 halfcheetah 資料集，需先完成對應 `base_train / aspl`
- `hybrid_random_train_min / aspl`：5 個 hybrid random 資料集，需先完成對應 `hybrid_random_train / aspl`

## 暫停 Train

`scas_n` 目前判定仍不成熟，以下速查表缺口暫停排入 `scripts/trains.py`。

- `base / scas_n`：`hopper_d4rl_expert`, `hopper_d4rl_hybrid`, `hopper_replay_medium`, `hopper_replay_expert`
- `base / scas_n`：`halfcheetah_replay_medium`
- `base / scas_n`：`walker2d_d4rl_medium`, `walker2d_d4rl_expert`, `walker2d_d4rl_hybrid`, `walker2d_replay_medium` 目前只有 `200000 / 500000`
- `hybrid_random / scas_n`：5 個 hybrid random 資料集

## 速查表缺口但未排入任務

這些仍在 `documents/result/*.csv` 內顯示缺口，但目前沒有排入 `scripts/trains.py` 或 `scripts/test_all.py`。

- `stability_scaspl / scaspl_n / walker2d_replay_medium`：目前只有 `200000 / 500000`
- `base / scaspl_n / walker2d_replay_medium`：目前只有 `200000 / 500000`
- `hybrid_random / scaspl_n / walker2d_random_expert_9`：缺 eval
- `noise_init / scaspl_n`：6 個 noise test 尚缺，但 SCASPL 系列目前未排入統整 test

## 可信度警告

速查表內仍有不少 `!` 標記，代表目前 agent 檔案更新時間晚於採用的 eval。這些分數可以作為舊結果參考，但若要放入正式論文表格，仍需按目前最新 agent 定義重新完成 train / train-min / test。
