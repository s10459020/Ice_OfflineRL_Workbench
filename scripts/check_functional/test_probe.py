from pathlib import Path
import tempfile

import gymnasium as gym
import numpy as np

from ice_offline.store.probe.action_axis_probe import ActionAxisProbe
from ice_offline.store.probe.op_collector import ProbeCollectWrapper


ENV_ID = "Hopper-v5"
SAMPLE_COUNT = 100
STEP_COUNT = 5
ACTION_DIM = 3


def eval_fn(observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
    return np.sum(actions, axis=1, dtype=np.float32)


def main() -> None:
    env = gym.make(ENV_ID)
    wrapper = ProbeCollectWrapper(env, ActionAxisProbe(SAMPLE_COUNT), eval_fn)
    try:
        wrapper.reset(seed=0)
        for _ in range(STEP_COUNT):
            action = wrapper.action_space.sample()
            _, _, terminated, truncated, _ = wrapper.step(action)
            if terminated or truncated:
                break

        output_root = Path(tempfile.mkdtemp(prefix="ice_probe_check_"))
        dataset = wrapper.save(output_root / "data" / "main_data.hdf5")
        try:
            step = dataset.read_step(0, 0)
            probe_count = SAMPLE_COUNT * ACTION_DIM
            _check_shape("observations", step["observations"], (probe_count, 11))
            _check_shape("actions", step["actions"], (probe_count, 3))
            _check_shape("values", step["values"], (probe_count,))
            _check_ood_action_segments(step["actions"])
            print(f"saved={dataset.path}")
            print(f"episode_count={dataset.episode_count}")
            print(f"step_counts={dataset.step_counts}")
            print(f"observations_shape={step['observations'].shape}")
            print(f"actions_shape={step['actions'].shape}")
            print(f"values_shape={step['values'].shape}")
        finally:
            dataset.close()
    finally:
        wrapper.close()


def _check_shape(name: str, value: np.ndarray, expected: tuple[int, ...]) -> None:
    actual = tuple(value.shape)
    if actual != expected:
        raise RuntimeError(f"{name} shape expected={expected} actual={actual}")


def _check_ood_action_segments(actions: np.ndarray) -> None:
    for action_index in range(ACTION_DIM):
        start = action_index * SAMPLE_COUNT
        stop = start + SAMPLE_COUNT
        segment = actions[start:stop]
        changing = segment[:, action_index]
        if float(changing[0]) != -1.0 or float(changing[-1]) != 1.0:
            raise RuntimeError(f"action dimension {action_index} does not scan [-1, 1]")


if __name__ == "__main__":
    main()
