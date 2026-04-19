import d3rlpy
from ice_offline.dataset import d3rl_convertor


if __name__ == "__main__":
    # mute d3rlpy logs
    d3rlpy.logging.LOG.info = lambda *args, **kwargs: None
    d3rlpy.logging.LOG.debug = lambda *args, **kwargs: None

    device = "cpu"
    observation_shape = (8,)
    action_size = 4

    config = d3rlpy.algos.DiscreteBCConfig()
    algo = config.create(device=device)
    algo.create_impl(
        observation_shape=observation_shape,
        action_size=action_size,
    )

    n_steps = 200
    dataset_id = "minigrid/BabyAI-OneRoomS8/optimal-fullobs-v0"
    dataset = d3rl_convertor.to_buffer(dataset_id, mode="flatten")
    algo.fit(
        dataset,
        n_steps=n_steps,
        show_progress=False,
    )
