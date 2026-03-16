import inspect
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np


class EnvDumperWrapper(gym.Wrapper):
    """Generic env dumper wrapper for inspecting attributes and callables."""

    def _format_space_atom(self, space: gym.Space) -> str:
        if isinstance(space, gym.spaces.Box):
            return (
                f"Box(shape={list(space.shape)}, dtype={space.dtype}, "
                f"low_shape={list(np.array(space.low).shape)}, "
                f"high_shape={list(np.array(space.high).shape)})"
            )
        if isinstance(space, gym.spaces.Discrete):
            return f"Discrete(n={int(space.n)})"
        if isinstance(space, gym.spaces.MultiDiscrete):
            return f"MultiDiscrete(nvec={np.array(space.nvec).tolist()})"
        if isinstance(space, gym.spaces.MultiBinary):
            return f"MultiBinary(n={int(space.n)})"
        return str(space)

    def _format_space_inline(self, space: gym.Space) -> str:
        if isinstance(space, gym.spaces.Dict):
            parts = [
                f"{key}: {self._format_space_inline(subspace)}"
                for key, subspace in space.spaces.items()
            ]
            return "Dict{" + ", ".join(parts) + "}"
        if isinstance(space, gym.spaces.Tuple):
            parts = [self._format_space_inline(subspace) for subspace in space.spaces]
            return "Tuple[" + ", ".join(parts) + "]"
        return self._format_space_atom(space)

    def _format_space(self, space: gym.Space, indent: int = 0) -> str:
        pad = " " * indent

        if isinstance(space, gym.spaces.Dict):
            lines = [f"{pad}Dict{{"]
            for key, subspace in space.spaces.items():
                lines.append(
                    f"{pad}  {key}: {self._format_space_inline(subspace)}"
                )
            lines.append(f"{pad}}}")
            return "\n".join(lines)

        if isinstance(space, gym.spaces.Tuple):
            lines = [f"{pad}Tuple["]
            for idx, subspace in enumerate(space.spaces):
                lines.append(f"{pad}  [{idx}]:")
                lines.append(self._format_space(subspace, indent + 4))
            lines.append(f"{pad}]")
            return "\n".join(lines)
        return f"{pad}{self._format_space_atom(space)}"

    def _format_shallow(self, value: Any) -> str:
        if isinstance(value, np.ndarray):
            return f"ndarray(shape={value.shape}, dtype={value.dtype})"
        return repr(value)

    def _format_dict_one_level(self, value: dict[Any, Any]) -> str:
        lines = ["{"]
        keys = sorted(value.keys(), key=lambda x: str(x))
        for idx, key in enumerate(keys):
            suffix = "," if idx < len(keys) - 1 else ""
            lines.append(f"  {repr(key)}: {self._format_shallow(value[key])}{suffix}")
        lines.append("}")
        return "\n".join(lines)

    def _format_object_one_level(self, obj: Any) -> str:
        if not hasattr(obj, "__dict__"):
            return repr(obj)
        try:
            data = vars(obj)
        except Exception:
            return repr(obj)
        if not data:
            return f"{type(obj).__name__}{{}}"
        lines = [f"{type(obj).__name__}{{"]
        keys = sorted(data.keys(), key=lambda x: str(x))
        for idx, key in enumerate(keys):
            suffix = "," if idx < len(keys) - 1 else ""
            lines.append(f"  {key}: {self._format_shallow(data[key])}{suffix}")
        lines.append("}")
        return "\n".join(lines)

    def _format_value(self, value: Any) -> str:
        if isinstance(value, np.ndarray):
            return f"ndarray(shape={value.shape}, dtype={value.dtype})"
        if isinstance(value, (str, int, float, bool, type(None))):
            return repr(value)
        if isinstance(value, dict):
            return self._format_dict_one_level(value)
        if isinstance(value, (list, tuple, set)):
            return repr(value)
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            return f"{type(value).__name__}(shape={value.shape}, dtype={value.dtype})"
        if hasattr(value, "__dict__"):
            return self._format_object_one_level(value)
        return f"{type(value).__name__}: {repr(value)}"

    def _safe_signature(self, fn: Any) -> str:
        try:
            return str(inspect.signature(fn))
        except Exception:
            return "(signature unavailable)"

    def _wrapper_chain(self) -> list[str]:
        chain = []
        current = self
        while True:
            chain.append(type(current).__name__)
            if not hasattr(current, "env"):
                break
            next_env = getattr(current, "env")
            if next_env is current:
                break
            current = next_env
        return chain

    def _emit_bullet(self, lines: list[str], key: str, text: str) -> None:
        if "\n" in text:
            split = text.splitlines()
            lines.append(f"  - {key}: {split[0]}")
            lines.extend(f"    {line}" for line in split[1:])
        else:
            lines.append(f"  - {key}: {text}")

    def _append_bullet(self, lines: list[str], key: str, value: Any, format_space: bool = True) -> None:
        text = (
            self._format_space(value)
            if format_space and isinstance(value, gym.Space)
            else self._format_value(value)
        )
        self._emit_bullet(lines, key, text)

    def _dump_instance_attributes(self, obj: Any) -> list[str]:
        if not hasattr(obj, "__dict__"):
            return ["  (no __dict__)"]

        lines: list[str] = []
        for key in sorted(obj.__dict__.keys()):
            self._append_bullet(lines, key, obj.__dict__[key], format_space=False)
        return lines

    def _dump_methods(self, obj: Any) -> list[str]:
        lines: list[str] = []
        for name, member in inspect.getmembers(obj):
            if name.startswith("__"):
                continue
            if callable(member):
                lines.append(f"  - {name}{self._safe_signature(member)}")
        return lines

    def _dump_properties(self, cls: type) -> list[str]:
        lines: list[str] = []
        for name, member in inspect.getmembers(cls):
            if name.startswith("__"):
                continue
            if isinstance(member, property):
                lines.append(f"  - {name} (property)")
        return lines

    def dump_report(self, env_id: str, render_mode: str | None) -> str:
        base = self.env.unwrapped
        lines: list[str] = []

        lines.append("=== ENV INFO ===")
        basic_info = {
            "env_id": env_id,
            "render_mode": render_mode,
            "wrapper_chain": self._wrapper_chain(),
            "observation_space": self.observation_space,
            "action_space": self.action_space,
        }
        for key, value in basic_info.items():
            self._append_bullet(lines, key, value)

        lines.append("\n=== INSTANCE ATTRIBUTES (env.unwrapped.__dict__) ===")
        lines.extend(self._dump_instance_attributes(base))

        lines.append("\n=== METHODS (env.unwrapped callable members) ===")
        lines.extend(self._dump_methods(base))

        lines.append("\n=== PROPERTIES (class-level) ===")
        lines.extend(self._dump_properties(type(base)))

        lines.append("\n=== CLASS MRO ===")
        for cls in type(base).mro():
            lines.append(f"  - {cls.__module__}.{cls.__name__}")

        return "\n".join(lines)


def run_env_dump(
    env_id: str,
    render_mode: str | None = None,
    output: str | None = None,
) -> str:
    env = gym.make(env_id, render_mode=render_mode)
    env = EnvDumperWrapper(env)

    env.reset()
    text = env.dump_report(env_id=env_id, render_mode=render_mode)

    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Dump written to: {output}")
    else:
        print(text)

    env.close()
    return text


if __name__ == "__main__":
    # Demo only: FourRooms
    import minigrid  # noqa: F401

    run_env_dump(
        env_id="MiniGrid-FourRooms-v0",
        render_mode=None,
        output=str(Path(__file__).resolve().with_name("fourrooms_dump.txt")),
    )


# Backward compatibility alias
EnvDumper = EnvDumperWrapper
