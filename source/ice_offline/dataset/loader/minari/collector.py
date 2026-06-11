import json
import os
from pathlib import Path
from typing import Any

import minari

from ice_offline.config.paths import RUNS_ROOT


class MinariCollectorWrapper(minari.DataCollector):
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
        path: Path,
        *,
        id: str,
        agent_id: str,
        eval_env=None,
        algorithm_name: str = "unknown",
        author: str = "ice_offline",
        author_email: str = "ice_offline@example.com",
        code_permalink: str = "https://example.com/ice_offline",
        description: str = "",
    ):
        dataset_id = path.relative_to(RUNS_ROOT).parent.parent.as_posix()
        os.environ["MINARI_DATASETS_PATH"] = str(RUNS_ROOT)
        if eval_env is None:
            eval_env = self.env
        try:
            try:
                minari.delete_dataset(dataset_id)
            except Exception:
                pass

            dataset = self.create_dataset(
                dataset_id=dataset_id,
                algorithm_name=algorithm_name,
                author=author,
                author_email=author_email,
                code_permalink=code_permalink,
                eval_env=eval_env,
                description=description,
            )
            metadata_path = path.parent / "metadata.json"
            with metadata_path.open("r", encoding="utf-8") as file:
                metadata = json.load(file)
            env_spec = metadata.get("env_spec", {})
            if isinstance(env_spec, str):
                env_spec = json.loads(env_spec)
            metadata["id"] = id
            metadata["agent_id"] = agent_id
            metadata["env_id"] = env_spec.get("id", "")
            with metadata_path.open("w", encoding="utf-8", newline="\n") as file:
                json.dump(metadata, file, ensure_ascii=False)
                file.write("\n")
            return dataset
        except Exception:
            self.close()
            raise
