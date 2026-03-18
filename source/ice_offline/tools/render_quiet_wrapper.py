import gymnasium as gym


class RenderQuietWrapper(gym.Wrapper):
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


def ensure_render_quiet(env: gym.Env) -> gym.Env:
    """Ensure RenderQuietWrapper exists; insert closest to base env when possible."""
    if not isinstance(env, gym.Wrapper):
        return RenderQuietWrapper(env)

    current = env
    while isinstance(current, gym.Wrapper) and isinstance(current.env, gym.Wrapper):
        current = current.env

    if not isinstance(current.env, RenderQuietWrapper):
        current.env = RenderQuietWrapper(current.env)
    return env
