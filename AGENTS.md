環境固定規則：
1. 未來提供任何可點擊的檔案超連結時，路徑一律使用 /E: 開頭（例如 /E:/Ice_OfflineRL_Workbench/AGENTS.md）。
2. 任何與 d3rl 相關的參考查詢，優先且預設只從專案內 .venv 的原始碼搜尋；回覆時需附上對應的 /E: 開頭檔案連結。
3. 本專案定位為 Offline RL 研究專案；所有建議、實作與評估以 Offline RL 工作流與研究可重現性為優先。
4. 所有新建檔案必須使用 UTF-8 編碼；所有檔案讀取與寫入操作預設使用 UTF-8，以避免中文亂碼。
5. 命名與匯出禁止使用別名設計（alias）。所有檔名、import、export 與對外 API 名稱必須使用單一正式名稱，不得以轉發或別名方式維持相容。
6. 全專案禁止使用 `from __future__ import annotations`；不得新增此行。
6. 全專案禁止使用 `from __future__ import annotations`；不得新增此行。

目前專案理解（2026-04-30）：
1. dataset 主來源為 Minari 格式資料集；專案多數流程以 `dataset_id` 配合 `minari.load_dataset(...)` 讀取資料，路徑根目錄由 `source/ice_offline/tools/paths.py` 的 `minari_root()` 管理（預設 `tmps/datasets`）。
2. state 資訊有兩條主路：
   - online：透過 `StateCollector`（`source/ice_offline/dataset/collector_state.py`）在 `reset/step` 蒐集狀態，輸出 `state_data.hdf5`。
   - offline：透過 `convert_fullobs`（`source/ice_offline/dataset/converter_state.py`）把 Minari episode 轉成 `State`，並寫入 `infos["state"]` 建立轉換後資料集。
3. 重播（replay）主路：
   - online inject：`StateInjectWrapper`（`source/ice_offline/dataset/injector_state.py`）依資料集中的 state/obs/action/reward 做 transition 重播。
   - offline loader：`StateLoader`（`source/ice_offline/dataset/loader_state.py`）從 `state_data.hdf5` 讀取 episode 與 step。
4. value 流程目前以 state-action value（Q-like）為核心：
   - 蒐集：`ValueCollector`（`source/ice_offline/dataset/collector_value.py`）輸出 `value_data.hdf5` 的 `episode_x/values`。
   - 載入：`ValueLoader`（`source/ice_offline/dataset/loader_value.py`）讀取 `values` 供後續渲染與檢查。
5. overlay 為 state/value 的可視化核心：
   - 引擎：`OverlayEngine`（`source/ice_offline/env/visualization/overlay_engine.py`）。
   - state trail：`TrailUnit`（`source/ice_offline/env/visualization/unit_trail.py`）。
   - value distribution：`DistributionUnit`（`source/ice_offline/env/visualization/unit_distribution.py`，含 ring/rect 風格）。
6. distribution 目前主要支援 state-action 分布可視化（以現有 values 結構為基礎），尚未抽象成多種 value 指標的通用 schema。
7. API/UI 檢查介面目前由 GUI service + presenter 串接：
   - service：`MinariDatasetService`（`source/ice_offline/env/gui/services/minari_dataset_service.py`）整合 overlay loader 與 units。
   - presenter：`ViewerPresenter`（`source/ice_offline/env/gui/presenters/viewer_presenter.py`）處理 episode/step 與 all-mode 導航。
