from collections.abc import Iterable


class Trail:
    """Trail storage + indexing."""

    def __init__(self, *, max_trails: int = 64) -> None:
        if max_trails < 1:
            raise ValueError("max_trails must be >= 1")
        self.max_trails = int(max_trails)
        self.points: list[tuple[tuple[int, int], int]] = []
        self.points_by_cell: dict[tuple[int, int], list[tuple[int, int]]] = {}
        self._cells_dirty = True

    def reset(self) -> None:
        self.points.clear()
        self.points_by_cell.clear()
        self._cells_dirty = True

    def push(self, pos: tuple[int, int], direction: int) -> None:
        self.points.append((pos, int(direction)))
        overflow = len(self.points) - self.max_trails
        if overflow > 0:
            del self.points[:overflow]
        self._cells_dirty = True

    def extend(self, points: Iterable[tuple[tuple[int, int], int]]) -> None:
        for pos, direction in points:
            self.push(pos, direction)

    def _build_cells(self) -> None:
        if not self._cells_dirty:
            return

        points_by_cell_raw: dict[tuple[int, int], dict[int, tuple[int, int, int]]] = {}
        total = len(self.points)
        den = max(1, total - 1)
        for idx, (pos, trail_dir) in enumerate(self.points):
            ratio = idx / den
            if ratio <= 0.25:
                bucket = 1
            elif ratio <= 0.50:
                bucket = 2
            elif ratio <= 0.75:
                bucket = 3
            else:
                bucket = 4
            by_dir = points_by_cell_raw.setdefault(pos, {})
            by_dir[int(trail_dir)] = (idx, bucket, int(trail_dir))

        points_by_cell: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for pos, by_dir in points_by_cell_raw.items():
            ordered = sorted(by_dir.values(), key=lambda item: item[0])
            points_by_cell[pos] = [(bucket, trail_dir) for _, bucket, trail_dir in ordered]
        self.points_by_cell = points_by_cell
        self._cells_dirty = False

    def get_cell(self, i: int, j: int) -> list[tuple[int, int]]:
        self._build_cells()
        return self.points_by_cell.get((int(i), int(j)), [])

    def get_cell_hash(self, i: int, j: int) -> int:
        entries = self.get_cell(i, j)
        h = 0
        for level, trail_dir in entries:
            token = (((int(level) & 0x7) << 2) | (int(trail_dir) & 0x3)) + 1
            h = (h * 33) + token
        return h

    def count(self) -> int:
        return len(self.points)
