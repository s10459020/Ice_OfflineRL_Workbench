# MDPDatatset
- 繼承於ReplayBuffer
- 提供舊版的建構方式(os,as,rs,terminals, timeouts)
- 使用EpisodeGenerator切分資料，補齊episodes

# Episode
- 自行定義episodes，並傳入ReplayBuffer
- 提供已切分後的建構方式(ep.os, op.as, op.rs...)