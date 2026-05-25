from ice_offline.dataset._lookup import get_dataset
import test_simple_aspl
import test_simple_bc
import test_simple_cql
import test_simple_iql
import test_simple_random
import test_simple_scas
import train_simple_aspl
import train_simple_bc
import train_simple_cql
import train_simple_iql
import train_simple_scas


ENV_ID = "Hopper-v5"
TRAIN_STEP = 200_000
TEST_EPISODES = 200
SAVE_INTERVAL=20_000
MODEL_STEP=200_000


if __name__ == "__main__":
    dataset = get_dataset("hopper_simple")
    train_simple_bc.collect(dataset=dataset, env_id=ENV_ID, steps=TRAIN_STEP, save_interval=SAVE_INTERVAL)
    train_simple_cql.collect(dataset=dataset, env_id=ENV_ID, steps=TRAIN_STEP, save_interval=SAVE_INTERVAL)
    train_simple_iql.collect(dataset=dataset, env_id=ENV_ID, steps=TRAIN_STEP, save_interval=SAVE_INTERVAL)
    train_simple_scas.collect(dataset=dataset, env_id=ENV_ID, dynamics_steps=TRAIN_STEP, agent_steps=TRAIN_STEP, save_interval=SAVE_INTERVAL)
    train_simple_aspl.collect(dataset=dataset, env_id=ENV_ID, steps=TRAIN_STEP, save_interval=SAVE_INTERVAL)

    test_simple_bc.collect(env_id=ENV_ID, test_episodes=TEST_EPISODES, model_step=MODEL_STEP)
    test_simple_cql.collect(env_id=ENV_ID, test_episodes=TEST_EPISODES, model_step=MODEL_STEP)
    test_simple_iql.collect(env_id=ENV_ID, test_episodes=TEST_EPISODES, model_step=MODEL_STEP)
    test_simple_scas.collect(env_id=ENV_ID, test_episodes=TEST_EPISODES, model_step=MODEL_STEP)
    test_simple_aspl.collect(env_id=ENV_ID, test_episodes=TEST_EPISODES, model_step=MODEL_STEP)
    test_simple_random.collect(env_id=ENV_ID, test_episodes=TEST_EPISODES)
