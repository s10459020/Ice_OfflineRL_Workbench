# Result Quick Tables

這個資料夾保存論文表格用的速查表。速查表不取代正式 `view`，而是用同一套 mean-return normalization 規則，在 `test` 尚未完整時用較新的可用資料做估計。

## 檔案

- `stability_td3bc.csv`
- `stability_aspl.csv`
- `stability_scas.csv`
- `stability_scaspl.csv`
- `stability_scc.csv`
- `base.csv`
- `noise_init.csv`
- `noise_action.csv`
- `noise_state.csv`
- `hybrid_random.csv`
- `agent_dataset_versions.csv`
- `agent_file_versions.csv`

## 分數規則

每個 cell 都是 normalized mean return：

```text
score = (mean_return - lower_mean) / (upper_mean - lower_mean) * 100
```

`lower_mean` 與 `upper_mean` 沿用各 experiment view 裡的 table boundary：

- `base`：三個環境各自使用 random 到 medium/expert 的 boundary。
- `stability_*`：使用 walker2d 五個資料集的 boundary。
- `noise_*`：使用目前 noise view 定義的 walker2d noise 任務 boundary。
- `hybrid_random`：使用 walker2d random 到 d4rl expert 的 boundary。

## 來源優先規則

對每個 `(experiment, dataset, agent)` 只採用新格式路徑：

```text
tmps/evals/{experiment}/{agent}/{dataset}-v0/data/eval_data.hdf5
```

不採用舊式 `{dataset}-{agent}-v0` 路徑。

候選來源：

1. `test`：正式 test eval，符合 `agent_step + 20_000` 才算完整，使用全部 rows。
2. `train_min`：若正式 test 缺失，使用對應 train-min eval 的最後 10 rows，cell 加 `(tm)`。
3. `train`：若只有 train eval，且達到該 agent 的指定訓練步數，使用最後 10 rows，cell 加 `(t)`。
4. 若最新可用來源尚未達到指定步數，但已經有 eval rows 可參考，使用最後 10 rows，cell 加 `(L)`。

如果較早流程的 eval 比較晚流程更新，採用較新的 eval；若最新 eval 沒達到指定步數但已有可用 rows，cell 以 `(L)` 標記，沒有可用 rows 才留空。

## 排序規則

- `stability_*`：agent 欄位順序必須對齊 `scripts/experiment_stability/view_*.py` 內各子 view 的 `AGENTS`。
- 標準資料集列順序固定為 `M, E, H, MR, ER`，也就是 `d4rl_medium`, `d4rl_expert`, `d4rl_hybrid`, `replay_medium`, `replay_expert`。
- `base`：每個環境內都使用 `M, E, H, MR, ER`，環境順序維持 `hopper`, `walker2d`, `halfcheetah`。
- `noise_*`：只包含該 view 定義的 walker2d 任務，任務內資料集順序維持 `M, H, MR`。

## 可信度標記

第一次版本檢查使用 agent 檔案更新時間：

```text
source/ice_offline/agent/{agent}.py
```

若 agent 檔案比採用的 eval data 新，cell 前綴 `!`，代表分數可能已被目前 agent 定義淘汰，需要重新訓練/測試。

同一個資料集列內，若 cell 分數達到該列最高分的 95% 以上，cell 後綴 `*`，代表該方法在同資料集比較中屬於接近最佳區間。

`agent_dataset_versions.csv` 會列出每個 cell 的：

- 使用來源 stage。
- eval data path 與更新時間。
- agent file path 與更新時間。
- agent 是否比 eval data 新。
- raw mean 與 normalized score。

## 更新方式

從專案根目錄執行：

```bash
.venv/bin/python documents/result/generate_result_quick_tables.py
```
