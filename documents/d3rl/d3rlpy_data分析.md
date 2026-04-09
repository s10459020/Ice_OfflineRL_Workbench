# d3rlpy data

d3rlpy_data是利用`dataset, env = get_cartpole()`等方式自動下載的資料集，預設會下載到根目錄下的`d3rlpy_data/`內，格式與minari data不同。


## 格式
以`cartpole_replay_v1.1.0.h5`為例子
- 99866 steps
- 1582 episode
	- 1448 truncated
	- 134 done


### 回合資訊 terminals, episode_terminals
terminals對應truncated，代表回合失敗，過程被中斷
episode_terminals對應done，代表回合順利執行結束

當發生done時，該回合會直接結束，目前猜測episode_terminals做為回合結束的正式開關
episode_terminals並不會影響到terminals，每個回合的第199個step觸發
`(episode_terminals[1553]=1, episode_terminals[1752]=1, terminals[1752]=0)`

當發生truncated時，該回合並不會直接結束，而是會在下回合觸發episode_terminals，
因此可以從資料中觀察到terminals=1會使下回合terminals與episode_terminals皆為1。
`(terminals[33]=1, then terminals[34]=1 and episode_terminals[34]=1)`

該資料庫運行了1582個episode
- episode_terminals = 1582次，對應到 134 done + 1448 truncated
- terminals = 2896次，對應到1448次的truncated會連續觸發
- ReplayBuffer.episodes在連續的terminals中會出現「單回合truncated」的資訊
- ReplayBuffer.episodes.length = 3030次，對應 2896 terminals + 134 done
- 目前沒發生「第199回合發生truncated」這種特殊情況


### 轉移資訊 observations, actions, rewards  
這些是transition資料，但是d3rlpy data將所有回合的資訊都「合併」在一起了。
必須透過terminals, episode_terminals去區分回合
目前顯示(o,a,r)共有99866筆，但程式顯示transition_count為99732筆

當terminals發生時，最後一回合會被count計算進去，用做TD訓練的最後一筆資料
(平時為y = r + Q(s',a')，最後一回合為y = r，不需要s'/o'的資訊)

而當只有episode_terminals發生時，最後一回合則不包含，等於該step作廢
(99866-99732 = 134 done)


### 其他資訊
- version: data版本
- discrete_action: action為連續或離散