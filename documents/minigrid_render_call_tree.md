# MiniGrid Render Call Tree (Top-Down)

## Color legend
- `🔵` initialized / mostly stable after init-reset
- `🟠` runtime render-internal state
- `🟢` step-varying state

## Variable summary used in render path
- `🔵 self.render_mode`
- `🔵 self.highlight`
- `🔵 self.tile_size`
- `🔵 self.agent_pov`
- `🔵 self.screen_size`
- `🔵 self.width`
- `🔵 self.height`
- `🔵 self.agent_view_size`
- `🔵 self.see_through_walls`
- `🔵 self.metadata["render_fps"]`
- `🟠 self.render_size`
- `🟠 self.window`
- `🟠 self.clock`
- `🟢 self.mission`
- `🟢 self.agent_pos`
- `🟢 self.agent_dir`
- `🟢 self.grid`
- `🟢 self.carrying`


## Call order from `env.render()`
1. `def MiniGridEnv.render`
2. `def MiniGridEnv.get_frame`
3. Branch A: `def MiniGridEnv.get_full_render`
4. Branch A: `def MiniGridEnv.gen_obs_grid`
5. Branch A: `def MiniGridEnv.get_view_exts`
6. Shared: `def Grid.render`
7. Shared: `def Grid.render_tile`
8. Shared: `def highlight_img` (if highlight)
9. Branch B: `def MiniGridEnv.get_pov_render`
10. Branch B reuses: `gen_obs_grid -> get_view_exts -> Grid.render -> Grid.render_tile -> highlight_img`

---

## def MiniGridEnv.render
```python
def render(self):
    img = self.get_frame(🔵self.highlight, 🔵self.tile_size, 🔵self.agent_pov)

    if 🔵self.render_mode == "human":
        img = np.transpose(img, axes=(1, 0, 2))
        if 🟠self.render_size is None:
            🟠self.render_size = img.shape[:2]
        if 🟠self.window is None:
            pygame.init()
            pygame.display.init()
            🟠self.window = pygame.display.set_mode(
                (🔵self.screen_size, 🔵self.screen_size)
            )
            pygame.display.set_caption("minigrid")
        if 🟠self.clock is None:
            🟠self.clock = pygame.time.Clock()
        surf = pygame.surfarray.make_surface(img)

        offset = surf.get_size()[0] * 0.1
        bg = pygame.Surface(
            (int(surf.get_size()[0] + offset), int(surf.get_size()[1] + offset))
        )
        bg.convert()
        bg.fill((255, 255, 255))
        bg.blit(surf, (offset / 2, 0))

        bg = pygame.transform.smoothscale(bg, (🔵self.screen_size, 🔵self.screen_size))

        font_size = 22
        text = 🟢self.mission
        font = pygame.freetype.SysFont(pygame.font.get_default_font(), font_size)
        text_rect = font.get_rect(text, size=font_size)
        text_rect.center = bg.get_rect().center
        text_rect.y = bg.get_height() - font_size * 1.5
        font.render_to(bg, text_rect, text, size=font_size)

        self.window.blit(bg, (0, 0))
        pygame.event.pump()
        self.clock.tick(🔵self.metadata["render_fps"])
        pygame.display.flip()

    elif 🔵self.render_mode == "rgb_array":
        return img
```

## def MiniGridEnv.get_frame
```python
def get_frame(
    self,
    highlight: bool = True,
    tile_size: int = TILE_PIXELS,
    agent_pov: bool = False,
):
    if agent_pov:
        return self.get_pov_render(tile_size)
    else:
        return self.get_full_render(highlight, tile_size)
```

---

## Branch A - def MiniGridEnv.get_full_render
```python
def get_full_render(self, highlight, tile_size):
    _, vis_mask = self.gen_obs_grid()

    f_vec = self.dir_vec
    r_vec = self.right_vec
    top_left = (
        🟢self.agent_pos
        + f_vec * (🔵self.agent_view_size - 1)
        - r_vec * (🔵self.agent_view_size // 2)
    )

    highlight_mask = np.zeros(shape=(🔵self.width, 🔵self.height), dtype=bool)

    for vis_j in range(0, 🔵self.agent_view_size):
        for vis_i in range(0, 🔵self.agent_view_size):
            if not vis_mask[vis_i, vis_j]:
                continue

            abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

            if abs_i < 0 or abs_i >= 🔵self.width:
                continue
            if abs_j < 0 or abs_j >= 🔵self.height:
                continue

            highlight_mask[abs_i, abs_j] = True

    img = self.grid.render(
        tile_size,
        🟢self.agent_pos,
        🟢self.agent_dir,
        highlight_mask=highlight_mask if highlight else None,
    )

    return img
```

## Branch A/B shared - def MiniGridEnv.gen_obs_grid
```python
def gen_obs_grid(self, agent_view_size=None):
    topX, topY, botX, botY = self.get_view_exts(agent_view_size)

    agent_view_size = agent_view_size or 🔵self.agent_view_size

    grid = self.grid.slice(topX, topY, agent_view_size, agent_view_size)

    for i in range(🟢self.agent_dir + 1):
        grid = grid.rotate_left()

    if not 🔵self.see_through_walls:
        vis_mask = grid.process_vis(
            agent_pos=(agent_view_size // 2, agent_view_size - 1)
        )
    else:
        vis_mask = np.ones(shape=(grid.width, grid.height), dtype=bool)

    agent_pos = grid.width // 2, grid.height - 1
    if 🟢self.carrying:
        grid.set(*agent_pos, 🟢self.carrying)
    else:
        grid.set(*agent_pos, None)

    return grid, vis_mask
```

## Branch A/B shared - def MiniGridEnv.get_view_exts
```python
def get_view_exts(self, agent_view_size=None):
    agent_view_size = agent_view_size or 🔵self.agent_view_size

    if 🟢self.agent_dir == 0:
        topX = 🟢self.agent_pos[0]
        topY = 🟢self.agent_pos[1] - agent_view_size // 2
    elif 🟢self.agent_dir == 1:
        topX = 🟢self.agent_pos[0] - agent_view_size // 2
        topY = 🟢self.agent_pos[1]
    elif 🟢self.agent_dir == 2:
        topX = 🟢self.agent_pos[0] - agent_view_size + 1
        topY = 🟢self.agent_pos[1] - agent_view_size // 2
    elif 🟢self.agent_dir == 3:
        topX = 🟢self.agent_pos[0] - agent_view_size // 2
        topY = 🟢self.agent_pos[1] - agent_view_size + 1
    else:
        assert False, "invalid agent direction"

    botX = topX + agent_view_size
    botY = topY + agent_view_size
    return topX, topY, botX, botY
```

---

## Shared lower-level - def Grid.render
```python
def render(
    self,
    tile_size: int,
    agent_pos: tuple[int, int],
    agent_dir: int | None = None,
    highlight_mask: np.ndarray | None = None,
) -> np.ndarray:
    if highlight_mask is None:
        highlight_mask = np.zeros(shape=(🔵self.width, 🔵self.height), dtype=bool)

    width_px = 🔵self.width * tile_size
    height_px = 🔵self.height * tile_size
    img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

    for j in range(0, 🔵self.height):
        for i in range(0, 🔵self.width):
            cell = self.get(i, j)

            agent_here = np.array_equal(agent_pos, (i, j))
            tile_img = Grid.render_tile(
                cell,
                agent_dir=agent_dir if agent_here else None,
                highlight=highlight_mask[i, j],
                tile_size=tile_size,
            )

            ymin = j * tile_size
            ymax = (j + 1) * tile_size
            xmin = i * tile_size
            xmax = (i + 1) * tile_size
            img[ymin:ymax, xmin:xmax, :] = tile_img

    return img
```

## Shared lower-level - def Grid.render_tile
```python
@classmethod
def render_tile(
    cls,
    obj: WorldObj | None,
    agent_dir: int | None = None,
    highlight: bool = False,
    tile_size: int = TILE_PIXELS,
    subdivs: int = 3,
) -> np.ndarray:
    key: tuple[Any, ...] = (agent_dir, highlight, tile_size)
    key = obj.encode() + key if obj else key

    if key in cls.tile_cache:
        return cls.tile_cache[key]

    img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)

    fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
    fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

    if obj is not None:
        obj.render(img)

    if agent_dir is not None:
        tri_fn = point_in_triangle((0.12, 0.19), (0.87, 0.50), (0.12, 0.81))
        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * agent_dir)
        fill_coords(img, tri_fn, (255, 0, 0))

    if highlight:
        highlight_img(img)

    img = downsample(img, subdivs)
    cls.tile_cache[key] = img
    return img
```

## Shared lower-level - def highlight_img
```python
def highlight_img(img, color=(255, 255, 255), alpha=0.30):
    blend_img = img + alpha * (np.array(color, dtype=np.uint8) - img)
    blend_img = blend_img.clip(0, 255).astype(np.uint8)
    img[:, :, :] = blend_img
```

---

## Branch B - def MiniGridEnv.get_pov_render
```python
def get_pov_render(self, tile_size):
    grid, vis_mask = self.gen_obs_grid()

    img = grid.render(
        tile_size,
        agent_pos=(🔵self.agent_view_size // 2, 🔵self.agent_view_size - 1),
        agent_dir=3,
        highlight_mask=vis_mask,
    )
    return img
```










