
def print_banner(title: str) -> None:
    bar = "#" * 72
    print("\n" + bar)
    print(f"# {title.upper():^68} #")
    print(bar)


def print_stage(title: str) -> None:
    print(f"\n# STAGE: {str(title).upper()}")
    print_banner(title)


def format_transition(
    *,
    step: int,
    episode: int,
    episode_step: int,
    action: int,
    reward: float,
    terminated: bool,
    truncated: bool,
) -> str:
    return (
        f"step={int(step)} episode={int(episode)} episode_step={int(episode_step)} "
        f"action={int(action)} reward={float(reward):.3f} "
        f"terminated={bool(terminated)} truncated={bool(truncated)}"
    )


def print_transition(
    *,
    step: int,
    episode: int,
    episode_step: int,
    action: int,
    reward: float,
    terminated: bool,
    truncated: bool,
) -> None:
    print(
        format_transition(
            step=step,
            episode=episode,
            episode_step=episode_step,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )
    )


# Backward-compatible alias.
stage = print_stage
