# MiniGrid Step Call Tree (Top-Down)

## Color legend
- `🔵` initialized / mostly stable after init-reset
- `🟠` runtime env-internal state
- `🟢` step-varying state

## Variable summary used in step path
- `🔵 self.actions`
- `🔵 self.max_steps`
- `🔵 self.render_mode`
- `🔵 self.agent_view_size`
- `🔵 self.see_through_walls`
- `🟢 self.step_count`
- `🟢 self.agent_pos`
- `🟢 self.agent_dir`
- `🟢 self.grid`
- `🟢 self.carrying`
- `🟢 self.mission`

## Call order from `env.step(action)`
1. `def MiniGridEnv.step`
2. `property MiniGridEnv.front_pos`
3. `property MiniGridEnv.dir_vec`
4. `def Grid.get` (front cell)
5. Branch update (`left/right/forward/pickup/drop/toggle/done`)
6. Optional `def MiniGridEnv._reward` (goal reached)
7. Optional `def MiniGridEnv.render` (if `render_mode == "human"`)
8. `def MiniGridEnv.gen_obs`
9. `def MiniGridEnv.gen_obs_grid`
10. `def MiniGridEnv.get_view_exts`

---

## def MiniGridEnv.step
```python
def step(self, action):
    🟢self.step_count += 1

    reward = 0
    terminated = False
    truncated = False

    fwd_pos = 🟢self.front_pos
    fwd_cell = self.grid.get(*fwd_pos)

    if action == 🔵self.actions.left:
        🟢self.agent_dir -= 1
        if 🟢self.agent_dir < 0:
            🟢self.agent_dir += 4

    elif action == 🔵self.actions.right:
        🟢self.agent_dir = (🟢self.agent_dir + 1) % 4

    elif action == 🔵self.actions.forward:
        if fwd_cell is None or fwd_cell.can_overlap():
            🟢self.agent_pos = tuple(fwd_pos)
        if fwd_cell is not None and fwd_cell.type == "goal":
            terminated = True
            reward = self._reward()
        if fwd_cell is not None and fwd_cell.type == "lava":
            terminated = True

    elif action == 🔵self.actions.pickup:
        if fwd_cell and fwd_cell.can_pickup():
            if 🟢self.carrying is None:
                🟢self.carrying = fwd_cell
                🟢self.carrying.cur_pos = np.array([-1, -1])
                self.grid.set(fwd_pos[0], fwd_pos[1], None)

    elif action == 🔵self.actions.drop:
        if not fwd_cell and 🟢self.carrying:
            self.grid.set(fwd_pos[0], fwd_pos[1], 🟢self.carrying)
            🟢self.carrying.cur_pos = fwd_pos
            🟢self.carrying = None

    elif action == 🔵self.actions.toggle:
        if fwd_cell:
            fwd_cell.toggle(self, fwd_pos)

    elif action == 🔵self.actions.done:
        pass

    else:
        raise ValueError(...)

    if 🟢self.step_count >= 🔵self.max_steps:
        truncated = True

    if 🔵self.render_mode == "human":
        self.render()

    obs = self.gen_obs()
    return obs, reward, terminated, truncated, {}
```

---

## Shared in step path - property MiniGridEnv.front_pos
```python
@property
def front_pos(self):
    return 🟢self.agent_pos + self.dir_vec
```

## Shared in step path - property MiniGridEnv.dir_vec
```python
@property
def dir_vec(self):
    assert 0 <= 🟢self.agent_dir < 4
    return DIR_TO_VEC[🟢self.agent_dir]
```

---

## Shared in step path - def MiniGridEnv.gen_obs
```python
def gen_obs(self):
    grid, vis_mask = self.gen_obs_grid()
    image = grid.encode(vis_mask)
    obs = {
        "image": image,
        "direction": 🟢self.agent_dir,
        "mission": 🟢self.mission,
    }
    return obs
```

## Shared in step path - def MiniGridEnv.gen_obs_grid
```python
def gen_obs_grid(self, agent_view_size=None):
    topX, topY, botX, botY = self.get_view_exts(agent_view_size)

    agent_view_size = agent_view_size or 🔵self.agent_view_size
    grid = self.grid.slice(topX, topY, agent_view_size, agent_view_size)

    for _ in range(🟢self.agent_dir + 1):
        grid = grid.rotate_left()

    if not 🔵self.see_through_walls:
        vis_mask = grid.process_vis(agent_pos=(agent_view_size // 2, agent_view_size - 1))
    else:
        vis_mask = np.ones(shape=(grid.width, grid.height), dtype=bool)

    agent_pos = (grid.width // 2, grid.height - 1)
    if 🟢self.carrying:
        grid.set(*agent_pos, 🟢self.carrying)
    else:
        grid.set(*agent_pos, None)

    return grid, vis_mask
```

## Shared in step path - def MiniGridEnv.get_view_exts
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
        assert False

    botX = topX + agent_view_size
    botY = topY + agent_view_size
    return topX, topY, botX, botY
```
