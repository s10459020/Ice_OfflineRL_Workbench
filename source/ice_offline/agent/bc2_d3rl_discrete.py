import d3rlpy

from ice_offline.dataset import d3rl_convertor
from ice_offline.tools.printer import print_stage


if __name__ == "__main__":
    # mute d3rlpy logs
    d3rlpy.logging.LOG.info = lambda *args, **kwargs: None
    d3rlpy.logging.LOG.debug = lambda *args, **kwargs: None

    # ====================
    # Train config
    # ====================
    dataset_id = "minigrid/BabyAI-OneRoomS8/optimal-fullobs-v0"
    n_steps = 200

    # ====================
    # Step 1: choose runtime device
    # ====================
    device = "cpu"

    # ====================
    # Step 2: create default config
    # default includes:
    # - batch_size=100
    # - learning_rate=1e-3
    # - beta=0.5
    # - vector encoder hidden_units=[256, 256]
    # ====================
    config = d3rlpy.algos.DiscreteBCConfig()

    # ====================
    # Step 3: create algo object from config
    # ====================
    algo = config.create(device=device)

    # ====================
    # Step 4: build default model weights/optimizer
    # use observation shape + discrete action size
    # ====================
    observation_shape = (8,)
    action_size = 4
    algo.create_impl(
        observation_shape=observation_shape,
        action_size=action_size,
    )

    # ====================
    # Step 5: check built modules
    # ====================
    assert algo.impl is not None
    policy = algo.impl.modules.imitator
    optimizer = algo.impl.modules.optim.optim

    print("DiscreteBC default model is built.")
    print(f"device={device}")
    print(f"observation_shape={observation_shape}")
    print(f"action_size={action_size}")
    print(f"policy_type={type(policy).__name__}")
    print(f"optimizer_type={type(optimizer).__name__}")

    # ====================
    # Step 6: load dataset
    # ====================
    print_stage("Dataset")
    dataset = d3rl_convertor.to_buffer(dataset_id, mode="flatten")
    print(f"dataset_id={dataset_id}")
    print(f"episode_count={len(dataset.episodes)}")
    print(f"transition_count={dataset.transition_count}")

    # ====================
    # Step 7: train
    # ====================
    print_stage("Train")
    algo.fit(
        dataset,
        n_steps=n_steps,
        show_progress=False,
    )
    print(f"train_steps={n_steps}")
    print("DiscreteBC training finished.")
