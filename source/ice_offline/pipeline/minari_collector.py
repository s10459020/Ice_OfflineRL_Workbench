from typing import Any

import minari


class MinariCollectorWrapper:
    def __init__(
        self,
        env,
        *,
        record_infos: bool = False,
        algorithm_name: str = "unknown",
        author: str = "ice_offline",
        author_email: str = "ice_offline@example.com",
        code_permalink: str = "https://example.com/ice_offline",
        description: str = "",
        eval_env=None,
    ) -> None:
        self._collector = minari.DataCollector(env, record_infos=record_infos)
        self._algorithm_name = algorithm_name
        self._author = author
        self._author_email = author_email
        self._code_permalink = code_permalink
        self._description = description
        self._eval_env = eval_env

    def reset(self, *args: Any, **kwargs: Any):
        return self._collector.reset(*args, **kwargs)

    def step(self, *args: Any, **kwargs: Any):
        return self._collector.step(*args, **kwargs)

    def close(self) -> None:
        self._collector.close()

    def save(self, dataset_id: str):
        try:
            minari.delete_dataset(dataset_id)
        except Exception:
            pass

        return self._collector.create_dataset(
            dataset_id=dataset_id,
            algorithm_name=self._algorithm_name,
            author=self._author,
            author_email=self._author_email,
            code_permalink=self._code_permalink,
            eval_env=self._eval_env,
            description=self._description,
        )
