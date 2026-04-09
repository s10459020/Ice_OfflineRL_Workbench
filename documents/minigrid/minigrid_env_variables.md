# MiniGrid Env Variables (Blue/Green)

Source: `minigrid_render_call_tree.md` + `minigrid_step_call_tree.md`

## Color legend
- `🔵` initialized / mostly stable after init-reset
- `🟢` step-varying state

## 🔵 Blue variables (initialized / mostly stable)

| Variable | kwargs-related mapping (direct or indirect) |
|---|---|
| `self.render_mode` | `render_mode` |
| `self.highlight` | `highlight` |
| `self.tile_size` | `tile_size` |
| `self.agent_pov` | `agent_pov` |
| `self.screen_size` | `screen_size` |
| `self.width` | `width` / `grid_size`; some BabyAI envs are indirectly influenced by `room_size` |
| `self.height` | `height` / `grid_size`; some BabyAI envs are indirectly influenced by `room_size` |
| `self.agent_view_size` | `agent_view_size` |
| `self.see_through_walls` | `see_through_walls` |
| `self.max_steps` | `max_steps` |
| `self.actions` | No common kwargs mapping (usually fixed `Actions` enum) |
| `self.metadata["render_fps"]` | No kwargs mapping (class static metadata) |

## 🟢 Green variables (step-varying state)

| Variable | Notes |
|---|---|
| `self.mission` | Mission text changes per episode/environment generation |
| `self.agent_pos` | Agent position changes during reset/step |
| `self.agent_dir` | Agent direction changes during reset/step |
| `self.grid` | World grid state changes during reset/step |
| `self.carrying` | Carried object state changes on pickup/drop |
| `self.step_count` | Current episode step counter increments each step |

## Remaining kwargs (not directly mapped above)

| kwargs | Description |
|---|---|
| `debug` | Used in some BabyAI tasks to enable stricter verifier behavior; wrong interactions may directly trigger failure. |
| `first_color` | In two-door ordering tasks, fixes the first target door color (otherwise sampled). |
| `second_color` | In two-door ordering tasks, fixes the second target door color (otherwise sampled). |
| `num_doors` | Controls how many doors are generated in door-order tasks. |
| `num_objs` | Controls how many objects/distractors are generated in some put-next local tasks. |
| `select_by` | Controls how target selection is described in instruction generation (e.g., by `color` or by `loc`). |
| `strict` | Enables strict instruction verification; certain incorrect intermediate actions become immediate failure. |
