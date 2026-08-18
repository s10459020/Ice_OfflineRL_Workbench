from pathlib import Path
import subprocess
import sys
import venv


JOINT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = JOINT_ROOT.parent.parent
VENV_PATH = JOINT_ROOT / ".venv"
VENV_PYTHON = JOINT_ROOT / ".venv" / "bin" / "python"
ACTIVATE_PATH = VENV_PATH / "bin" / "activate"
ACTIVATE_LINES = (
    'export PYTHONPATH="$VIRTUAL_ENV/../..${PYTHONPATH:+:$PYTHONPATH}"',
)


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    subprocess.run(command, check=True, cwd=PROJECT_ROOT, env=env)


def create_venv() -> None:
    if not VENV_PATH.exists():
        venv.create(VENV_PATH, with_pip=True)

    activate_text = ACTIVATE_PATH.read_text(encoding="utf-8")
    activate_lines = [
        line
        for line in activate_text.splitlines()
        if not line.startswith("export PYTHONPATH=")
    ]
    activate_lines.extend(ACTIVATE_LINES)
    ACTIVATE_PATH.write_text("\n".join(activate_lines) + "\n", encoding="utf-8")

    print(f"Joint venv is ready at {VENV_PATH}")


def install_dependencies() -> None:
    run([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    run([
        str(VENV_PYTHON),
        "-m",
        "pip",
        "install",
        "--editable",
        str(JOINT_ROOT),
    ])


def verify_installation() -> None:
    run([
        str(VENV_PYTHON),
        "-c",
        "import gymnasium as gym, mujoco; env = gym.make('Hopper-v5'); env.close(); "
        "print(f'gymnasium={gym.__version__} mujoco={mujoco.__version__}')",
    ])


def main() -> None:
    create_venv()
    install_dependencies()
    verify_installation()
    print(f"Run: source {JOINT_ROOT / '.venv' / 'bin' / 'activate'}")


if __name__ == "__main__":
    main()
