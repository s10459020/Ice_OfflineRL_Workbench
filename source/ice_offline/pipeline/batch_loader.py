import numpy as np

from ice_offline.dataset._spec import BaseDataset


class BatchLoader:
    # ====================
    # Init
    # ====================
    def __init__(self, dataset_spec: BaseDataset, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

        self.dataset_spec = dataset_spec
        self.dataset = dataset_spec.make_dataset()
        self.buffer = self._build_buffer(self.dataset)
        
        self.obs_shape = tuple(int(x) for x in self.buffer["obs"].shape[1:])
        self.act_shape = tuple(int(x) for x in self.buffer["act"].shape[1:]) 
        self.obs_size = int(getattr(self.dataset.spec.observation_space, "n", 0))
        self.act_size = int(getattr(self.dataset.spec.action_space, "n", 0))
        self.count = self.buffer["obs"].shape[0]
        

    # ====================
    # Public API
    # ====================
    def sample_batch(self, batch_size: int) -> dict[str, np.ndarray]:
        idx = self._rng.integers(0, self.count, size=(batch_size,))
        return {
            "obs": self.buffer["obs"][idx],
            "next_obs": self.buffer["next_obs"][idx],
            "act": self.buffer["act"][idx],
            "rew": self.buffer["rew"][idx],
            "done": self.buffer["done"][idx],
        }


    # ====================
    # Internal
    # ====================
    def _build_buffer(self, dataset) -> dict[str, np.ndarray]:
        obs_list: list[np.ndarray] = []
        act_list: list[np.ndarray] = []
        rew_list: list[np.ndarray] = []
        done_list: list[np.ndarray] = []
        next_obs_list: list[np.ndarray] = []

        for episode in dataset.iterate_episodes():
            obs = self.dataset_spec.obs_encode_batch(episode.observations)
            act = self.dataset_spec.act_encode_batch(episode.actions)
            rew = np.asarray(episode.rewards, dtype=np.float32)

            term = np.asarray(episode.terminations, dtype=np.bool)
            trunc = np.asarray(episode.truncations, dtype=np.bool)
            done = np.logical_or(term, trunc).astype(np.float32)

            obs_list.append(obs[0:-1])
            next_obs_list.append(obs[1:])
            act_list.append(act)
            rew_list.append(rew)
            done_list.append(done)

        obs_all = np.concatenate(obs_list, axis=0)
        next_obs_all = np.concatenate(next_obs_list, axis=0)
        act_all = np.concatenate(act_list, axis=0)
        rew_all = np.concatenate(rew_list, axis=0)
        done_all = np.concatenate(done_list, axis=0)

        return {
            "obs": obs_all,
            "next_obs": next_obs_all,
            "act": act_all,
            "rew": rew_all,
            "done": done_all,
        }


