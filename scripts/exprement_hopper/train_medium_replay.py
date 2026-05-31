from ice_offline.dataset.hopper_medium_replay import HopperMediumReplayDataset
import ice_offline.run.train_aspl as train_aspl
import ice_offline.run.train_bc as train_bc
import ice_offline.run.train_cql as train_cql
import ice_offline.run.train_iql as train_iql
import ice_offline.run.train_scas as train_scas
import torch


DATASET_CLASS = HopperMediumReplayDataset
TRAIN_STEPS = 200_000
SAVE_INTERVAL = 20_000
EVAL_INTERVAL = 2_000
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    dataset = DATASET_CLASS(device=DEVICE).load()
    train_bc.collect(
        dataset=dataset,
        task_id=f"{dataset.id}_bc-v0",
        steps=TRAIN_STEPS,
        save_interval=SAVE_INTERVAL,
        eval_interval=EVAL_INTERVAL,
        device=DEVICE,
    )
    train_cql.collect(
        dataset=dataset,
        task_id=f"{dataset.id}_cql-v0",
        steps=TRAIN_STEPS,
        save_interval=SAVE_INTERVAL,
        eval_interval=EVAL_INTERVAL,
        device=DEVICE,
    )
    train_iql.collect(
        dataset=dataset,
        task_id=f"{dataset.id}_iql-v0",
        steps=TRAIN_STEPS,
        save_interval=SAVE_INTERVAL,
        eval_interval=EVAL_INTERVAL,
        device=DEVICE,
    )
    train_scas.collect(
        dataset=dataset,
        task_id=f"{dataset.id}_scas-v0",
        dynamics_steps=TRAIN_STEPS,
        agent_steps=TRAIN_STEPS,
        save_interval=SAVE_INTERVAL,
        eval_interval=EVAL_INTERVAL,
        device=DEVICE,
    )
    train_aspl.collect(
        dataset=dataset,
        task_id=f"{dataset.id}_aspl-v0",
        steps=TRAIN_STEPS,
        save_interval=SAVE_INTERVAL,
        eval_interval=EVAL_INTERVAL,
        device=DEVICE,
    )
