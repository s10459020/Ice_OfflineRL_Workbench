import argparse
import bisect

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

from ice_offline.pipeline import OldValueLoader


DATASET_ID = "check/visualization-q-step-v0"


def _build_canvas(values_xyda: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inner_w, inner_h, _, _ = values_xyda.shape
    values_xyd = np.max(values_xyda, axis=3)

    canvas = np.zeros((inner_h * 2, inner_w * 2), dtype=np.float32)
    mask = np.zeros_like(canvas, dtype=bool)

    for x in range(inner_w):
        for y in range(inner_h):
            d_vals = values_xyd[x, y]
            row = y * 2
            col = x * 2
            canvas[row, col] = d_vals[3]      # up
            canvas[row, col + 1] = d_vals[0]  # right
            canvas[row + 1, col] = d_vals[2]  # left
            canvas[row + 1, col + 1] = d_vals[1]  # down
            mask[row : row + 2, col : col + 2] = True

    return canvas, mask


def main(dataset_id: str, episode: int, step: int) -> None:
    loader = OldValueLoader(dataset_id)
    try:
        episode_count = loader.get_episode_count()
        episode_steps = [len(loader.load_episode(ep)) for ep in range(episode_count)]
        episode_offsets = [0]
        for count in episode_steps:
            episode_offsets.append(episode_offsets[-1] + count)
        total_steps = episode_offsets[-1]

        initial_episode = max(0, min(episode, episode_count - 1))
        initial_step = max(0, min(step, episode_steps[initial_episode] - 1))
        initial_global_step = episode_offsets[initial_episode] + initial_step
        values_xyda = loader.load_step(initial_episode, initial_step)
        canvas, mask = _build_canvas(values_xyda)

        fig, ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.12)
        image = ax.imshow(canvas, cmap="viridis")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

        rows, cols = canvas.shape
        text_artists = []

        def draw_labels(current_canvas: np.ndarray) -> None:
            nonlocal text_artists
            for text in text_artists:
                text.remove()
            text_artists = []
            for r in range(rows):
                for c in range(cols):
                    if mask[r, c]:
                        text = ax.text(
                            c,
                            r,
                            f"{current_canvas[r, c]:.1f}",
                            ha="center",
                            va="center",
                            color="white",
                            fontsize=8,
                        )
                        text_artists.append(text)

        draw_labels(canvas)

        inner_w, inner_h, _, _ = values_xyda.shape
        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.6, alpha=0.45)

        ax.set_xticks(np.arange(-0.5, cols, 2))
        ax.set_yticks(np.arange(-0.5, rows, 2))
        ax.grid(which="major", color="white", linewidth=1.6, alpha=0.9)
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(rows - 0.5, -0.5)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax_step = fig.add_axes([0.15, 0.04, 0.65, 0.03])
        step_slider = Slider(ax_step, "global_step", 0, total_steps - 1, valinit=initial_global_step, valstep=1)

        def render_current() -> None:
            global_step = int(step_slider.val)
            current_episode = bisect.bisect_right(episode_offsets, global_step) - 1
            current_step = global_step - episode_offsets[current_episode]
            current_values = loader.load_step(current_episode, current_step)
            current_canvas, _ = _build_canvas(current_values)
            image.set_data(current_canvas)
            draw_labels(current_canvas)
            ax.set_title(
                f"Q Step View - dataset={dataset_id} "
                f"episode={current_episode} step={current_step} global_step={global_step} "
                f"shape=({inner_w},{inner_h},4,a)"
            )
            fig.canvas.draw_idle()

        def on_slider_change(_val: float) -> None:
            render_current()

        def on_key_press(event) -> None:
            if event.key == "right":
                step_slider.set_val(min(total_steps - 1, int(step_slider.val) + 1))
            elif event.key == "left":
                step_slider.set_val(max(0, int(step_slider.val) - 1))

        step_slider.on_changed(on_slider_change)
        fig.canvas.mpl_connect("key_press_event", on_key_press)
        render_current()
        plt.show()
    finally:
        loader.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", type=str, default=DATASET_ID)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--step", type=int, default=1)
    args = parser.parse_args()
    main(args.dataset_id, args.episode, args.step)
