import gymnasium as gym


class RenderQuiteWrapper(gym.Wrapper):
    """Suppress implicit render calls triggered inside env.reset()/env.step()."""

    def _run_without_render(self, fn, *args, **kwargs):
        base_env = self.env.unwrapped
        original_render_mode = getattr(base_env, "render_mode", None)
        if hasattr(base_env, "render_mode"):
            base_env.render_mode = None
        try:
            return fn(*args, **kwargs)
        finally:
            if hasattr(base_env, "render_mode"):
                base_env.render_mode = original_render_mode

    def reset(self, **kwargs):
        return self._run_without_render(self.env.reset, **kwargs)

    def step(self, action):
        return self._run_without_render(self.env.step, action)
