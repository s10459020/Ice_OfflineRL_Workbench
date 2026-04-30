import minigrid  # noqa: F401

from ice_offline.env.visualization import BasicUnit, OverlayLoader
from ice_offline.env.visualization.overlay_engine import RenderLayer
from ice_offline.tools import Timer


def main(dataset: str = "test_collect_state-v0") -> None:
    loader = OverlayLoader(dataset=dataset, units=[BasicUnit()], render_mode="human")
    try:
        total_episodes = loader.get_episode_count()
        for ep in range(total_episodes):
            loader.load(ep)
            steps = 0
            t = 1
            while True:
                seek_ms, out = Timer.record("overlay_loader_human.seek", lambda: loader.seek(t))
                _, _, _, terminated, truncated, _ = out
                render_ms, _ = Timer.record("overlay_loader_human.render", loader.render)
                layer_text = " ".join(
                    f"{layer.name.lower()}={Timer.get(f'overlay.layer.{layer.name.lower()}'):.3f}ms"
                    for layer in RenderLayer
                )
                print(f"episode={ep} seek={t:03d} seek_ms={seek_ms:.3f} render_ms={render_ms:.3f} {layer_text}")
                steps += 1
                if terminated or truncated:
                    break
                t += 1
            print(f"loader episode={ep} steps={steps}")
    finally:
        loader.close()


if __name__ == "__main__":
    main()
