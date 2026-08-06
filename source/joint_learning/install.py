import os
from pathlib import Path
import subprocess
import sys


JOINT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = JOINT_ROOT.parent.parent
VENV_PYTHON = JOINT_ROOT / ".venv" / "bin" / "python"
MUJOCO_LIBRARY_PATH = Path.home() / ".mujoco" / "mujoco210" / "bin"
NVIDIA_LIBRARY_PATH = Path("/usr/lib/nvidia")


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    subprocess.run(command, check=True, cwd=PROJECT_ROOT, env=env)


def main() -> None:
    run([sys.executable, str(JOINT_ROOT / "create_venv.py")])
    run(["bash", str(JOINT_ROOT / "install_mujoco.sh")])

    run([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    run([str(VENV_PYTHON), "-m", "pip", "install", "Cython<3", "numpy<2"])
    run([
        str(VENV_PYTHON),
        "-m",
        "pip",
        "install",
        "torch>=2.5",
        "gym[mujoco_py]==0.26.2",
        "h5py",
        "matplotlib",
        "scipy",
    ])

    env = os.environ.copy()
    library_paths = [str(MUJOCO_LIBRARY_PATH), str(NVIDIA_LIBRARY_PATH)]
    if env.get("LD_LIBRARY_PATH"):
        library_paths.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(library_paths)
    env["PYTHONPATH"] = str(PROJECT_ROOT / "source")

    run([
        str(VENV_PYTHON),
        "-c",
        "import gym, mujoco_py; env = gym.make('Hopper-v2'); env.close(); "
        "print(f'gym={gym.__version__} mujoco_py={mujoco_py.__version__}')",
    ], env=env)

    print(f"Run: source {JOINT_ROOT / '.venv' / 'bin' / 'activate'}")


if __name__ == "__main__":
    main()
