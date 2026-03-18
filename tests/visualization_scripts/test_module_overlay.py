import threading
import time

import gymnasium as gym
import minigrid  # noqa: F401  # Ensure MiniGrid environments are registered.
import numpy as np
import pygame
from minigrid.wrappers import FullyObsWrapper

from ice_offline.visualization.minigrid import RenderOverlayWrapper


ENV_ID = "BabyAI-OneRoomS8-v0"
SEED = 42
DISPLAY_FPS = 30


def make_native_env():
    return FullyObsWrapper(gym.make(ENV_ID, render_mode="rgb_array"))


def make_overlay_env():
    return RenderOverlayWrapper(FullyObsWrapper(gym.make(ENV_ID, render_mode="rgb_array")))


def worker_loop(make_env, state, stop_event):
    env = make_env()
    rng = np.random.default_rng(SEED)
    action_n = min(4, int(env.action_space.n))
    lock = state["lock"]
    try:
        env.reset(seed=SEED)
        while not stop_event.is_set():
            t0 = time.perf_counter()
            action = int(rng.integers(0, action_n))
            _, _, term, trunc, _ = env.step(action)
            if term or trunc:
                env.reset(seed=SEED)
            frame = env.render()
            dt = time.perf_counter() - t0
            with lock:
                state["frame"] = frame
                state["frames"] += 1
                state["steps"] += 1
                state["total_s"] += dt
    finally:
        env.close()


def main():
    state_n = {"lock": threading.Lock(), "frame": None, "frames": 0, "steps": 0, "total_s": 0.0}
    state_o = {"lock": threading.Lock(), "frame": None, "frames": 0, "steps": 0, "total_s": 0.0}
    stop_event = threading.Event()

    th_n = threading.Thread(target=worker_loop, args=(make_native_env, state_n, stop_event), daemon=True)
    th_o = threading.Thread(target=worker_loop, args=(make_overlay_env, state_o, stop_event), daemon=True)
    th_n.start()
    th_o.start()

    while True:
        with state_n["lock"]:
            frame_n = state_n["frame"]
        with state_o["lock"]:
            frame_o = state_o["frame"]
        if frame_n is not None and frame_o is not None:
            break
        time.sleep(0.01)

    gap = np.full((frame_n.shape[0], 8, 3), 30, dtype=np.uint8)
    left_w = int(frame_n.shape[1])
    gap_w = int(gap.shape[1])
    panel_h = int(frame_n.shape[0])

    pygame.init()
    screen = pygame.display.set_mode((left_w + gap_w + int(frame_o.shape[1]), panel_h))
    pygame.display.set_caption("Left: Native | Right: Overlay")
    font = pygame.font.Font(None, 22)
    clock = pygame.time.Clock()
    started = time.perf_counter()

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            with state_n["lock"]:
                frame_n = state_n["frame"]
                frames_n = state_n["frames"]
                steps_n = state_n["steps"]
                total_n = state_n["total_s"]
            with state_o["lock"]:
                frame_o = state_o["frame"]
                frames_o = state_o["frames"]
                steps_o = state_o["steps"]
                total_o = state_o["total_s"]

            combo = np.concatenate((frame_n, gap, frame_o), axis=1)
            surface = pygame.surfarray.make_surface(np.transpose(combo, (1, 0, 2)))
            screen.blit(surface, (0, 0))

            elapsed = max(1e-9, time.perf_counter() - started)
            fps_n = frames_n / elapsed
            fps_o = frames_o / elapsed
            ms_n = 1000.0 * total_n / max(1, steps_n)
            ms_o = 1000.0 * total_o / max(1, steps_o)
            txt_n = font.render(f"fps={fps_n:.1f} ms={ms_n:.2f}", True, (255, 255, 255))
            txt_o = font.render(f"fps={fps_o:.1f} ms={ms_o:.2f}", True, (255, 255, 255))
            lbl_n = font.render("Left: Native", True, (255, 255, 255))
            lbl_o = font.render("Right: Overlay", True, (255, 255, 255))
            screen.blit(lbl_n, (6, 6))
            screen.blit(lbl_o, (left_w + gap_w + 6, 6))
            y = panel_h - txt_n.get_height() - 6
            screen.blit(txt_n, (6, y))
            screen.blit(txt_o, (left_w + gap_w + 6, y))

            pygame.display.flip()
            clock.tick(DISPLAY_FPS)
    finally:
        stop_event.set()
        th_n.join(timeout=2.0)
        th_o.join(timeout=2.0)
        pygame.quit()


if __name__ == "__main__":
    main()
