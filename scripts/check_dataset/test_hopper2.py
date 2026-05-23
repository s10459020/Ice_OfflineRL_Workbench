import numpy as np
import gymnasium as gym


ENV_ID = "Hopper-v5"


# Parameter index mapping (directly tied to MujocoEnv.set_state(qpos, qvel))
# 0-5: qpos[0:6], 6-11: qvel[0:6]
PARAMETERS = [
    "qpos[0] rootx",
    "qpos[1] rootz",
    "qpos[2] rooty",
    "qpos[3] thigh_joint",
    "qpos[4] leg_joint",
    "qpos[5] foot_joint",
    "qvel[0] rootx_dot",
    "qvel[1] rootz_dot",
    "qvel[2] rooty_dot",
    "qvel[3] thigh_joint_dot",
    "qvel[4] leg_joint_dot",
    "qvel[5] foot_joint_dot",
]


def get_state(env: gym.Env) -> tuple[np.ndarray, np.ndarray]:
    base = env.unwrapped
    qpos = base.data.qpos.copy()
    qvel = base.data.qvel.copy()
    return qpos, qvel


def set_parameter(env: gym.Env, param_index: int, param_value: float) -> None:
    qpos, qvel = get_state(env)
    if param_index < 6:
        qpos[param_index] = param_value
    else:
        qvel[param_index - 6] = param_value
    env.unwrapped.set_state(qpos, qvel)


def print_parameter_list() -> None:
    print("=== Adjustable Parameters ===")
    for idx, name in enumerate(PARAMETERS):
        print(f"{idx}: {name}")


def print_state(env: gym.Env) -> None:
    qpos, qvel = get_state(env)
    print(f"qpos={np.array2string(qpos, precision=6)}")
    print(f"qvel={np.array2string(qvel, precision=6)}")


def main() -> None:
    env = gym.make(ENV_ID, render_mode="human")
    try:
        observation, info = env.reset()
        print("reset done")
        print(f"obs_shape={observation.shape}")
        print(f"reset_info={info}")
        print_parameter_list()
        print_state(env)

        zero_action = np.zeros(env.action_space.shape, dtype=np.float32)

        while True:
            raw = input("input: [empty]=step-zero-action, or index,value, or q to quit > ")
            if raw == "q":
                break

            if raw != "":
                index_text, value_text = raw.split(",")
                param_index = int(index_text.strip())
                param_value = float(value_text.strip())
                set_parameter(env, param_index, param_value)

            observation, reward, terminated, truncated, step_info = env.step(zero_action)
            print(f"reward={reward:.6f} terminated={terminated} truncated={truncated}")
            print(f"step_info={step_info}")
            print(f"obs={np.array2string(observation, precision=6)}")
            print_state(env)

            if terminated or truncated:
                observation, info = env.reset()
                print("auto reset")
                print(f"reset_info={info}")
                print_state(env)
    finally:
        env.close()


if __name__ == "__main__":
    main()
