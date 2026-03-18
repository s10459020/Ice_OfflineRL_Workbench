from __future__ import annotations


def print_banner(title: str) -> None:
    bar = "#" * 72
    print("\n" + bar)
    print(f"# {title.upper():^68} #")
    print(bar)

