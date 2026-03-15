from __future__ import annotations

import time

import gymnasium as gym


class RenderDelayWrapper(gym.Wrapper):
    """Render-throttling wrapper that draws according to target FPS."""

    def __init__(
        self,
        env: gym.Env,
        fps: float = 0.0,
        render_on_done: bool = False,
        render_on_reset: bool = False,
    ) -> None:
        super().__init__(env)
        self._interval = 0.0 if fps <= 0.0 else 1.0 / float(fps)
        self._render_on_done = bool(render_on_done)
        self._render_on_reset = bool(render_on_reset)
        self._next_render_time = 0.0

    def _render_now(self):
        if self._interval > 0.0:
            self._next_render_time = time.perf_counter() + self._interval
        return self.env.render()

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if self._render_on_reset:
            self._render_now()
        return result

    def render(self):
        if time.perf_counter() < self._next_render_time:
            return None

        return self._render_now()

    def step(self, action):
        out = self.env.step(action)
        _, _, terminated, truncated, _ = out
        if self._render_on_done and (terminated or truncated):
            self._render_now()
        return out
