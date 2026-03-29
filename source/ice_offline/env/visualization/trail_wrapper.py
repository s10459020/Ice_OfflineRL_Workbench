from .trail_unit import TrailUnit


class TrailWrapper(TrailUnit):
    """Render a trail overlay using recent (position, direction) states."""

    def __init__(
        self,
        max_trails: int = 64,
        trail_mode: str = "rollout",
    ) -> None:
        super().__init__(max_trails=max_trails, trail_mode=trail_mode)

