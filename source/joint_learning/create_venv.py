from pathlib import Path
import venv


JOINT_ROOT = Path(__file__).resolve().parent
VENV_PATH = JOINT_ROOT / ".venv"
ACTIVATE_PATH = VENV_PATH / "bin" / "activate"
ACTIVATE_LINES = (
    'export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"',
    'export PYTHONPATH="$VIRTUAL_ENV/../..${PYTHONPATH:+:$PYTHONPATH}"',
)


def main() -> None:
    if not VENV_PATH.exists():
        venv.create(VENV_PATH, with_pip=True)

    activate_text = ACTIVATE_PATH.read_text(encoding="utf-8")
    activate_lines = [
        line
        for line in activate_text.splitlines()
        if not line.startswith("export LD_LIBRARY_PATH=")
        and not line.startswith("export PYTHONPATH=")
    ]
    activate_lines.extend(ACTIVATE_LINES)
    ACTIVATE_PATH.write_text("\n".join(activate_lines) + "\n", encoding="utf-8")

    print(f"Joint venv is ready at {VENV_PATH}")


if __name__ == "__main__":
    main()
