import minari
import numpy as np


class MinariLoader:
    def __init__(self, dataset_id: str) -> None:
        self.dataset_id = dataset_id
        self.minari_dataset = minari.load_dataset(self.dataset_id, download=True)
        self.buffer = self._build_buffer(self.minari_dataset)

    def _build_buffer(self, minari_dataset) -> dict[str, np.ndarray]:
        obs_list: list[np.ndarray] = []
        next_obs_list: list[np.ndarray] = []
        act_list: list[np.ndarray] = []
        rew_list: list[np.ndarray] = []
        term_list: list[np.ndarray] = []
        trunc_list: list[np.ndarray] = []

        for episode in minari_dataset.iterate_episodes():
            obs_all = np.asarray(episode.observations)
            act = np.asarray(episode.actions)
            rew = np.asarray(episode.rewards)
            term = np.asarray(episode.terminations)
            trunc = np.asarray(episode.truncations)
            obs_list.append(obs_all[0:-1])
            next_obs_list.append(obs_all[1:])
            act_list.append(act)
            rew_list.append(rew)
            term_list.append(term)
            trunc_list.append(trunc)

        return {
            "observations": np.concatenate(obs_list, axis=0),
            "next_observations": np.concatenate(next_obs_list, axis=0),
            "actions": np.concatenate(act_list, axis=0),
            "rewards": np.concatenate(rew_list, axis=0),
            "terminations": np.concatenate(term_list, axis=0),
            "truncations": np.concatenate(trunc_list, axis=0),
        }
