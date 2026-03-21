import gymnasium as gym

from ice_offline.replay import StateReplayWrapper
from ice_offline.tools.types import State, Transition
from ice_offline.tools import insert_render_quiet_innermost


def run(
    env: gym.Env,
    max_episodes: int = 3,
    state_sequences: list[list[State]] | None = None,
    trajectories: list[list[Transition]] | None = None,
    *,
    render_interval: int | None = None,
    print_interval: int | None = None,
) -> int:
    env = insert_render_quiet_innermost(env)
    replay_count = min(int(max_episodes), len(state_sequences), len(trajectories))
    replay_env = StateReplayWrapper(
        env=env,
        state_sequences=state_sequences[:replay_count],
        trajectories=trajectories[:replay_count],
        random_episode=False,
    )

    step = 0
    try:
        for episode_index in range(replay_count):
            _obs, info = replay_env.reset(options={"episode_index": episode_index})
            if print_interval is not None:
                print_message = f"episode={episode_index} episode_step=0"
                state_index = info.get("state_index")
                state_sequence_length = info.get("state_sequence_length")
                trajectory_length = info.get("trajectory_length")
                state = info.get("state")
                if state_index is not None:
                    print_message += f" state_index={state_index}"
                if state_sequence_length is not None:
                    print_message += f" state_sequence_length={state_sequence_length}"
                if trajectory_length is not None:
                    print_message += f" trajectory_length={trajectory_length}"
                if state is not None:
                    print_message += f" pos={state.agent_pos} dir={state.agent_dir}"
                print(print_message)
            if render_interval == 1:
                replay_env.render()

            episode_step = 0
            while True:
                action = replay_env.act()
                _next_obs, reward, terminated, truncated, info = replay_env.step(action)
                episode_step += 1
                step += 1

                if print_interval is not None and step % print_interval == 0:
                    print_message = (
                        f"step={step} episode={episode_index} episode_step={episode_step} "
                        f"action={int(action)} reward={float(reward):.3f} "
                        f"terminated={terminated} truncated={truncated}"
                    )
                    state_index = info.get("state_index")
                    state = info.get("state")
                    if state_index is not None:
                        print_message += f" state_index={state_index}"
                    if state is not None:
                        print_message += f" pos={state.agent_pos} dir={state.agent_dir}"
                    print(print_message)
                if render_interval is not None and step % render_interval == 0:
                    replay_env.render()

                if terminated or truncated:
                    break

                _obs = _next_obs
    finally:
        replay_env.close()

    return step
