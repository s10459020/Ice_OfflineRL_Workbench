from dataclasses import dataclass


@dataclass(frozen=True)
class TrailPoint:
    pos: tuple[int, int]
    direction: int


class Trail:
    """Trail storage + indexing."""

    def __init__(self, *, max_trails: int = 8) -> None:
        if max_trails < 1:
            raise ValueError("max_trails must be >= 1")
        self.max_trails = int(max_trails)
        self.points: list[tuple[tuple[int, int], int]] = []
        self.points_by_cell: dict[tuple[int, int], list[tuple[int, int]]] = {}
        self._view_end: int = 0
        self._cells_dirty = True

    def reset(self) -> None:
        self.points.clear()
        self.points_by_cell.clear()
        self._view_end = 0
        self._cells_dirty = True

    def push(self, pos: tuple[int, int], direction: int) -> None:
        self.points.append((pos, int(direction)))
        self._view_end = len(self.points)
        self._cells_dirty = True

    def set_max_trails(self, max_trails: int) -> None:
        if max_trails < 1:
            raise ValueError("max_trails must be >= 1")
        self.max_trails = int(max_trails)
        self._cells_dirty = True

    def set_view_end(self, view_end: int | None) -> None:
        if view_end is None:
            self._view_end = len(self.points)
        else:
            self._view_end = max(0, min(int(view_end), len(self.points)))
        self._cells_dirty = True

    def get_cell(self, i: int, j: int) -> list[tuple[int, int]]:
        if self._cells_dirty:
            self._build_cells()
        return self.points_by_cell.get((int(i), int(j)), [])

    def _build_cells(self) -> None:
        end = max(0, min(self._view_end, len(self.points)))
        start = max(0, end - self.max_trails)
        visible_points = self.points[start:end]

        points_by_cell_raw: dict[tuple[int, int], dict[int, tuple[int, int, int]]] = {}
        total = len(visible_points)
        for idx, (pos, trail_dir) in enumerate(visible_points):
            # Uniform recency bucketing (1..4): newer points use darker buckets.
            # This avoids one-off special cases while keeping monotonic behavior.
            bucket = ((idx + 1) * 4 + total - 1) // total
            by_dir = points_by_cell_raw.setdefault(pos, {})
            by_dir[int(trail_dir)] = (idx, bucket, int(trail_dir))

        points_by_cell: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for pos, by_dir in points_by_cell_raw.items():
            ordered = sorted(by_dir.values(), key=lambda item: item[0])
            points_by_cell[pos] = [(bucket, trail_dir) for _, bucket, trail_dir in ordered]
        self.points_by_cell = points_by_cell
        self._cells_dirty = False
