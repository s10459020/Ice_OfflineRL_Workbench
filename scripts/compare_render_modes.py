import time
from collections.abc import Callable
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401  # Ensure MiniGrid environments are registered.

from visualization.minigrid import RenderDelayWrapper, RenderOverlayWrapper, TrailWrapper


RUN_STEPS = 10000
RUN_REPEATS = 10
ENV_ID = "MiniGrid-FourRooms-v0"
FPS_LIST = (1, 3, 5)
LOG_PATH = Path("../tmps/compare_render_modes.log")

_LOG_FILE = None


def _log(message: str) -> None:
    print(message)
    if _LOG_FILE is not None:
        _LOG_FILE.write(message + "\n")
        _LOG_FILE.flush()


def _setup_log_file() -> None:
    global _LOG_FILE
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _LOG_FILE = LOG_PATH.open("w", encoding="utf-8")


def _close_log_file() -> None:
    global _LOG_FILE
    if _LOG_FILE is not None:
        _LOG_FILE.close()
        _LOG_FILE = None


def _run_none(steps: int) -> float:
    env = gym.make(ENV_ID)
    env.reset()
    start = time.perf_counter()
    for _ in range(steps):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset()
    elapsed = time.perf_counter() - start
    env.close()
    return elapsed


def _run_text_render(steps: int) -> float:
    env = gym.make(ENV_ID)
    env.reset()
    start = time.perf_counter()
    for step in range(1, steps + 1):
        action = env.action_space.sample()
        _, reward, terminated, truncated, _ = env.step(action)
        print(
            f"text_render step={step} action={action} reward={float(reward):.3f} "
            f"terminated={terminated} truncated={truncated}"
        )
        if terminated or truncated:
            env.reset()
    elapsed = time.perf_counter() - start
    env.close()
    return elapsed


def _run_delay(steps: int, fps: int) -> float:
    env = gym.make(ENV_ID, render_mode="human")
    env = RenderDelayWrapper(env, fps=fps, render_on_done=False, render_on_reset=False)
    env.reset()
    start = time.perf_counter()
    for _ in range(steps):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        env.render()
        if terminated or truncated:
            env.reset()
    elapsed = time.perf_counter() - start
    env.close()
    return elapsed


def _run_trail(steps: int, fps: int) -> float:
    env = gym.make(ENV_ID, render_mode="human")
    env = RenderDelayWrapper(env, fps=fps, render_on_done=False, render_on_reset=False)
    env = RenderOverlayWrapper(env)
    env = TrailWrapper(env)
    env.reset()
    start = time.perf_counter()
    for _ in range(steps):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        env.render()
        if terminated or truncated:
            env.reset()
    elapsed = time.perf_counter() - start
    env.close()
    return elapsed


def _benchmark(name: str, runner: Callable[[int], float], steps: int, repeats: int) -> None:
    times: list[float] = []
    for i in range(1, repeats + 1):
        elapsed = runner(steps)
        times.append(elapsed)
        sps = steps / elapsed if elapsed > 0 else float("inf")
        _log(f"{name:>12} | run={i:02d}/{repeats} | time={elapsed:.3f}s | sps={sps:.2f}")

    avg_time = sum(times) / len(times)
    avg_sps = steps / avg_time if avg_time > 0 else float("inf")
    _log(f"{name:>12} | average       | time={avg_time:.3f}s | sps={avg_sps:.2f}")
    _log("-" * 72)


def compare_render_modes(
    steps: int = RUN_STEPS,
    repeats: int = RUN_REPEATS,
    fps_list: tuple[int, ...] = FPS_LIST,
) -> None:
    _setup_log_file()
    try:
        _log(f"Compare render modes | steps={steps} | repeats={repeats} | fps={list(fps_list)}")
        _log("-" * 72)

        _benchmark("text", _run_text_render, steps, repeats)
        _benchmark("none", _run_none, steps, repeats)
        for fps in fps_list:
            _benchmark(f"delay@{fps}", lambda s, f=fps: _run_delay(s, f), steps, repeats)
            _benchmark(f"trail@{fps}", lambda s, f=fps: _run_trail(s, f), steps, repeats)
    finally:
        _close_log_file()


if __name__ == "__main__":
    compare_render_modes()

