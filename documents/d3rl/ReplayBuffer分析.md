# ReplayBuffer
- dataset基礎元件

## attributes
- dataset_info 儲存空間格式
    - action_size: 離散為action數量，連續為action維度
    - action_space: action是否為Continuous
    - action/observation/reward_signature: numpy型別
- episodes 儲存回合資訊
    - episodes[ep_i] 提供 as[st_i]/os[st_i]/rs[st_i]
    - episodes[ep_i].transition_count 指有多少合法step
    - 在truncate發生時，會提供T+1個transition用以訓練，但step = T
    - episodes[ep_i].terminated 代表最後是否為terminal，
    - 實務上會產生terminal[step]，並根據terminated決定最後的數值
- buffer 用於抽樣
    - sample_transition_batch 會自動抽樣任意episodes內的transition
    - batch內可重複(推測)
    - batch間可重複，在還沒有跑完所有的transition前(推測))
- transition_count做為總數量

## transition
Transition被定義為(o,a,r,o',a')，可適用於包含SARSA的各種演算法。
然而，面對兩種情況(done正常結束, truncate被中斷)，資料會有不一樣的呈現方式。
- truncate代表回合被中斷，通常該Step資訊仍會被蒐集(o,a,r)，但僅作為o',a'使用
    - 假設第T+1回合觸發truncate，會記錄T+1個(o,a,r)，但count = T
    - y = r + Q(o', a')僅會運行1~T transition
- done代表回合順利結束，T+1回合的資訊仍被視為有效資訊，但不需要o', a'
    - 假設第T+1回合觸發done，會記錄T+1個(o,a,r)，且count = T+1
    - 當T+1回合時，若有terminal訊號則會塞入dummy o', a'維持程式運行
    - y = r + Q(o', a')*(1-terminal)，該方式會在結束時忽略下一步