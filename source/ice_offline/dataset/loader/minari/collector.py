import os
from typing import Any

import minari

from ice_offline.tools.paths import minari_root


class MinariCollectorWrapper(minari.DataCollector):
    def __init__(self, *args, **kwargs) -> None:
        os.environ.setdefault("MINARI_DATASETS_PATH", str(minari_root()))
        super().__init__(*args, **kwargs)

    def reset(self, *args: Any, **kwargs: Any):
        try:
            return super().reset(*args, **kwargs)
        except Exception:
            self.close()
            raise

    def step(self, *args: Any, **kwargs: Any):
        try:
            return super().step(*args, **kwargs)
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        super().close()

    def save(
        self,
        dataset_id: str,
        *,
        algorithm_name: str = "unknown",
        author: str = "ice_offline",
        author_email: str = "ice_offline@example.com",
        code_permalink: str = "https://example.com/ice_offline",
        description: str = "",
        eval_env=None,
    ):
        if eval_env is None:
            eval_env = self.env
        try:
            try:
                minari.delete_dataset(dataset_id)
            except Exception:
                pass

            return self.create_dataset(
                dataset_id=dataset_id,
                algorithm_name=algorithm_name,
                author=author,
                author_email=author_email,
                code_permalink=code_permalink,
                eval_env=eval_env,
                description=description,
            )
        except Exception:
            self.close()
            raise
