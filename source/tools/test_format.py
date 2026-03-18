from __future__ import annotations


def print_banner(title: str) -> None:
    bar = "#" * 72
    print("\n" + bar)
    print(f"# {title.upper():^68} #")
    print(bar)


def stage(title: str) -> None:
    print(f"\n# STAGE: {str(title).upper()}")
    print_banner(title)
