from ice_offline.dataset._lookup import get_dataset
from ice_offline.run import test_aspl, test_bc, test_cql, test_iql, test_random, test_scas


EPISODES = 100

if __name__ == "__main__":
    simple = "hopper_simple"
    medium = "hopper_medium"
    expert = "hopper_expert"
    medium_replay_d4rl = "hopper_medium_replay_d4rl"
    medium_expert_d4rl = "hopper_medium_expert_d4rl"
    #test_random.collect(get_dataset(simple), task_id=f"{simple}_random-v0", episodes=EPISODES)
    #test_bc.collect(get_dataset(simple), task_id=f"{simple}_bc-v0", episodes=EPISODES)
    #test_cql.collect(get_dataset(simple), task_id=f"{simple}_cql-v0", episodes=EPISODES)
    #test_iql.collect(get_dataset(simple), task_id=f"{simple}_iql-v0", episodes=EPISODES)
    #test_scas.collect(get_dataset(simple), task_id=f"{simple}_scas-v0", episodes=EPISODES)
    #test_aspl.collect(get_dataset(simple), task_id=f"{simple}_aspl-v0",   episodes=EPISODES)
    #test_random.collect(get_dataset(medium), task_id=f"{medium}_random-v0", episodes=EPISODES)
    #test_bc.collect(get_dataset(medium), task_id=f"{medium}_bc-v0", episodes=EPISODES)
    #test_cql.collect(get_dataset(medium), task_id=f"{medium}_cql-v0", episodes=EPISODES)
    #test_iql.collect(get_dataset(medium), task_id=f"{medium}_iql-v0", episodes=EPISODES)
    #test_scas.collect(get_dataset(medium), task_id=f"{medium}_scas-v0", episodes=EPISODES)
    #test_aspl.collect(get_dataset(medium), task_id=f"{medium}_aspl-v0",   episodes=EPISODES)
    #test_random.collect(get_dataset(expert), task_id=f"{expert}_random-v0", episodes=EPISODES)
    #test_bc.collect(get_dataset(expert), task_id=f"{expert}_bc-v0", episodes=EPISODES)
    #test_cql.collect(get_dataset(expert), task_id=f"{expert}_cql-v0", episodes=EPISODES)
    #test_iql.collect(get_dataset(expert), task_id=f"{expert}_iql-v0", episodes=EPISODES)
    #test_scas.collect(get_dataset(expert), task_id=f"{expert}_scas-v0", episodes=EPISODES)
    #test_aspl.collect(get_dataset(expert), task_id=f"{expert}_aspl-v0",   episodes=EPISODES)
    test_random.collect(get_dataset(medium_replay_d4rl), task_id=f"{medium_replay_d4rl}_random-v0", episodes=EPISODES)
    test_bc.collect(get_dataset(medium_replay_d4rl), task_id=f"{medium_replay_d4rl}_bc-v0", episodes=EPISODES)
    test_cql.collect(get_dataset(medium_replay_d4rl), task_id=f"{medium_replay_d4rl}_cql-v0", episodes=EPISODES)
    test_iql.collect(get_dataset(medium_replay_d4rl), task_id=f"{medium_replay_d4rl}_iql-v0", episodes=EPISODES)
    test_scas.collect(get_dataset(medium_replay_d4rl), task_id=f"{medium_replay_d4rl}_scas-v0", episodes=EPISODES)
    test_aspl.collect(get_dataset(medium_replay_d4rl), task_id=f"{medium_replay_d4rl}_aspl-v0",   episodes=EPISODES)
    test_random.collect(get_dataset(medium_expert_d4rl), task_id=f"{medium_expert_d4rl}_random-v0", episodes=EPISODES)
    test_bc.collect(get_dataset(medium_expert_d4rl), task_id=f"{medium_expert_d4rl}_bc-v0", episodes=EPISODES)
    test_cql.collect(get_dataset(medium_expert_d4rl), task_id=f"{medium_expert_d4rl}_cql-v0", episodes=EPISODES)
    test_iql.collect(get_dataset(medium_expert_d4rl), task_id=f"{medium_expert_d4rl}_iql-v0", episodes=EPISODES)
    test_scas.collect(get_dataset(medium_expert_d4rl), task_id=f"{medium_expert_d4rl}_scas-v0", episodes=EPISODES)
    test_aspl.collect(get_dataset(medium_expert_d4rl), task_id=f"{medium_expert_d4rl}_aspl-v0",   episodes=EPISODES)