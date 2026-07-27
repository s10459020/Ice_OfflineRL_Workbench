STANDARD_AGENT_SPECS = [
    ("bc", None, 50_000),
    ("td3bc_n", None, 100_000),
    ("iql", None, 200_000),
    ("cql", None, 500_000),
    ("aspl_c", None, 500_000),
    ("scas_n", 100_000, 500_000),
    ("scaspl_pn", 500_000, 500_000),
    ("scc_n", 100_000, 500_000),
]

AGENT_DISPLAY_NAMES = {
    "bc": "bc",
    "td3bc_n": "td3bc",
    "iql": "iql",
    "cql": "cql",
    "aspl_c": "aspl_c",
    "scas_n": "scas_n",
    "scaspl_pn": "scaspl_n",
    "scc_n": "scc_n",
}


def agent_display_name(agent_id: str) -> str:
    return AGENT_DISPLAY_NAMES.get(agent_id, agent_id)


def agent_display_names(agent_ids: list[str]) -> list[str]:
    return [agent_display_name(agent_id) for agent_id in agent_ids]


def standard_agent_ids() -> list[str]:
    return [agent_id for agent_id, _, _ in STANDARD_AGENT_SPECS]
