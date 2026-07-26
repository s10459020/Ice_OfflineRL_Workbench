# Remaining Experiments

更新時間：2026-07-25

這份清單依重新產生後的 `documents/result/agent_dataset_versions.csv`、目前 `scripts/trains.py`、`scripts/run_cpu.py`、`scripts/test_all.py` 整理。ASPL、SCAS、SCASPL 系列已視為新定義模型，舊分數不再作為速查表分數使用；新模型的 `200000` 局部資料仍可用 `(L)` 作為預估。

## Result 缺口總覽

- 目前速查表共有 115 個空白 cell。
- SCASPL 相關空白 86 個，原因是 `scaspl` 基底與其變體已被標記為 `stale_agent`。
- ASPL 相關空白 11 個：`stability_aspl / aspl_gp` 5 個、`stability_aspl / aspl_c` 4 個、`hybrid_random / aspl` 2 個。
- SCAS 相關空白 18 個：`base / scas_n` 6 個、`hybrid_random / scas_n` 5 個、`stability_scas / scas_gp` 2 個、`stability_scas / scas_gpn` 5 個。
- 仍有 12 個非空 cell 帶 `!`，這些不屬於 ASPL/SCAS/SCASPL 系列。
- 仍有 23 個 `(L)` cell，代表新模型已有局部資料但未達指定步數。

## SCASPL 需重作

SCASPL 目前不再信任任何舊結果，以下全部視為需要重新 train / train-min / test。

- `stability_scaspl`：30 個 cell，包含 `scaspl`, `scaspl_n`, `scaspl_gp`, `scaspl_c`, `scaspl_nc`, `scaspl_gpc` 對 walker2d 五個資料集。
- `base / scaspl_n`：15 個 cell，三個環境各五個資料集。
- `noise_init / scaspl_n`：12 個 noise test。
- `noise_action / scaspl_n`：12 個 noise test。
- `noise_state / scaspl_n`：12 個 noise test。
- `hybrid_random / scaspl_n`：5 個 hybrid random 資料集。

目前這批 SCASPL 重作尚未排入 `scripts/trains.py` 或 `scripts/test_all.py`，避免在結構尚未定稿時污染統合任務。

## 已排入 trains

`scripts/trains.py` 目前負責 CUDA 端的 ASPL 與 SCAS_N。

- `hybrid_random_train / aspl`：`walker2d_random_expert_7` 目前 `260000 / 500000`，`walker2d_random_expert_9` 尚缺。
- `base_train / scas_n`：11 個任務，其中 6 個 stale、5 個 missing。
- `hybrid_random_train / scas_n`：5 個任務尚缺。
- `base_train_min / aspl`：15 個任務尚缺。
- `hybrid_random_train_min / aspl`：3 個可補、2 個 blocked。
- `base_train_min / scas_n`：4 個可補、11 個 blocked。
- `hybrid_random_train_min / scas_n`：5 個 blocked。

## 已排入 run_cpu

`scripts/run_cpu.py` 目前只負責 stability 中 SCAS 變體的 CPU 後段流程。

- `stability_train_min / scas`：5 個任務尚缺。
- `stability_train_min / scas_gp`：3 個任務尚缺、2 個 blocked。
- `stability_train_min / scas_gpn`：5 個 blocked。
- `stability_test`：15 個任務目前都等待 train-min 完成。

## 已排入 test_all

`scripts/test_all.py` 維持目前待測清單，不額外加入 SCASPL。

- `base / aspl`：15 個 test。
- `base / scas_n`：4 個 halfcheetah test。
- `noise_init / scas_gp`：6 個 test。
- `noise_init / aspl`：12 個 test。
- `noise_action / aspl`：12 個 test。
- `noise_state / aspl`：12 個 test。
- `hybrid_random / scc_n`：5 個 test。

## 後續建議

1. 先讓目前 `trains.py` 與 `run_cpu.py` 的非 SCASPL 任務跑完，避免交叉污染。
2. SCASPL 定稿後，獨立建立一批 SCASPL-only train / train-min / test 清單。
3. SCASPL 重作完成後再重新執行 `documents/result/generate_result_quick_tables.py`，確認 `stability_scaspl.csv` 與代表模型中的 `scaspl_n` 不再空白。
