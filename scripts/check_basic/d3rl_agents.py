import d3rlpy
from ice_offline.dataset.loader.d3rl.converter import to_buffer

# mute logging
d3rlpy.logging.LOG.info = lambda *args, **kwargs: None
d3rlpy.logging.LOG.debug = lambda *args, **kwargs: None

DATASET_ID = "minigrid/BabyAI-OneRoomS8/optimal-fullobs-v0"
DEVICE = "cuda:0"
N_STEPS = 200
AGENTS = [
    d3rlpy.algos.AWACConfig,
    d3rlpy.algos.BCConfig,
    d3rlpy.algos.BCQConfig,
    d3rlpy.algos.BEARConfig,
    d3rlpy.algos.CQLConfig,
    d3rlpy.algos.CRRConfig,
    d3rlpy.algos.CalQLConfig,
    d3rlpy.algos.DDPGConfig,
    d3rlpy.algos.DQNConfig,
    d3rlpy.algos.DecisionTransformerConfig,
    d3rlpy.algos.DiscreteBCConfig,
    d3rlpy.algos.DiscreteBCQConfig,
    d3rlpy.algos.DiscreteCQLConfig,
    d3rlpy.algos.DiscreteDecisionTransformerConfig,
    d3rlpy.algos.DiscreteRandomPolicyConfig,
    d3rlpy.algos.DiscreteSACConfig,
    d3rlpy.algos.DoubleDQNConfig,
    d3rlpy.algos.IQLConfig,
    d3rlpy.algos.NFQConfig,
    d3rlpy.algos.PLASConfig,
    d3rlpy.algos.PLASWithPerturbationConfig,
    d3rlpy.algos.PRDCConfig,
    d3rlpy.algos.RandomPolicyConfig,
    d3rlpy.algos.ReBRACConfig,
    d3rlpy.algos.SACConfig,
    d3rlpy.algos.TACRConfig,
    d3rlpy.algos.TD3Config,
    d3rlpy.algos.TD3PlusBCConfig,
]

def main() -> None:
    dataset = to_buffer(DATASET_ID, mode="flatten")
 
    results: list[tuple[str, bool, str]] = []
    print("\n=== fitting ===")
    for config_cls in AGENTS:
        config_name = config_cls.__name__

        try:
            algo = config_cls().create(device=DEVICE)
            algo.fit(
                dataset,
                n_steps=N_STEPS,
                show_progress=False,
            )
            print(f"{config_name}: OK")
            results.append((config_name, True, ""))
        except Exception as e:
            print(f"{config_name}: FAIL")
            results.append((config_name, False, f"{type(e).__name__}: {e}"))

    print("\n=== summary ===")
    done = [name for name, success, _ in results if success]
    fail = [(name, msg) for name, success, msg in results if not success]
    print("success:", done)
    if fail:
        print("failed:")
        for name, msg in fail:
            brief = msg.splitlines()[0] if msg else ""
            print(f"- {name}: {brief}")


if __name__ == "__main__":
    main()

