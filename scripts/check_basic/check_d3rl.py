from d3rlpy.datasets import get_cartpole 
from d3rlpy.algos import DQNConfig
from d3rlpy.metrics import TDErrorEvaluator
from d3rlpy.metrics import EnvironmentEvaluator
import numpy as np

def main():
    ## Load environment and agent
    dataset, env = get_cartpole()
    dqn = DQNConfig().create(device="cpu:0")
    dqn.build_with_dataset(dataset)

    ## Load evaluator 
    td_error_evaluator = TDErrorEvaluator(episodes=dataset.episodes)
    env_evaluator = EnvironmentEvaluator(env)

    ## train offline
    dqn.fit(
        dataset,
        n_steps=100000,
        evaluators={
            'td_error': td_error_evaluator,
            'environment': env_evaluator,
        },
    )

    ## evaluate can also be called manually 
    td_errors = td_error_evaluator(dqn, dataset)
    print("TD errors:", td_errors)
    rewards = env_evaluator(dqn, dataset=None)
    print("rewards:", rewards)

    # test prediction
    observation, _ = env.reset()
    action = dqn.predict(np.expand_dims(observation, axis=0))
    value = dqn.predict_value(np.expand_dims(observation, axis=0), action)
    print("Predicted action:", action)
    print("Predicted value:", value)

if __name__ == "__main__":  
    main()
