from ice_offline.dataset._lookup import get_dataset
import ice_offline.run.train_aspl as train_aspl
import ice_offline.run.train_bc as train_bc
import ice_offline.run.train_cql as train_cql
import ice_offline.run.train_iql as train_iql
import ice_offline.run.train_scas as train_scas


DATASET_KEY = "hopper_expert"
TRAIN_STEPS = 200_000
SAVE_INTERVAL = 20_000
EVAL_INTERVAL = 2_000


if __name__ == "__main__":
    dataset = get_dataset(DATASET_KEY)
    train_bc.collect(
        dataset=dataset,
        task_id=f"{DATASET_KEY}_bc-v0",
        steps=TRAIN_STEPS,
        save_interval=SAVE_INTERVAL,
        eval_interval=EVAL_INTERVAL,
    )
    train_cql.collect(
        dataset=dataset,
        task_id=f"{DATASET_KEY}_cql-v0",
        steps=TRAIN_STEPS,
        save_interval=SAVE_INTERVAL,
        eval_interval=EVAL_INTERVAL,
    )
    train_iql.collect(
        dataset=dataset,
        task_id=f"{DATASET_KEY}_iql-v0",
        steps=TRAIN_STEPS,
        save_interval=SAVE_INTERVAL,
        eval_interval=EVAL_INTERVAL,
    )
    train_scas.collect(
        dataset=dataset,
        task_id=f"{DATASET_KEY}_scas-v0",
        dynamics_steps=TRAIN_STEPS,
        agent_steps=TRAIN_STEPS,
        save_interval=SAVE_INTERVAL,
        eval_interval=EVAL_INTERVAL,
    )
    train_aspl.collect(
        dataset=dataset,
        task_id=f"{DATASET_KEY}_aspl-v0",
        steps=TRAIN_STEPS,
        save_interval=SAVE_INTERVAL,
        eval_interval=EVAL_INTERVAL,
    )
