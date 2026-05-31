from ice_offline.dataset.hopper_expert import HopperExpertDataset
from ice_offline.dataset.hopper_medium import HopperMediumDataset
from ice_offline.dataset.hopper_medium_d4rl import HopperMediumD4rlDataset
from ice_offline.dataset.hopper_medium_expert import HopperMediumExpertDataset
from ice_offline.dataset.hopper_medium_replay import HopperMediumReplayDataset
from ice_offline.dataset.hopper_simple import HopperSimpleDataset
from train_agent import train_aspl, train_bc, train_cql, train_iql, train_scas
import torch


TRAIN_STEPS = 200_000
SAVE_INTERVAL = 20_000
EVAL_INTERVAL = 2_000
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    dataset = HopperSimpleDataset(device=DEVICE).load()
    train_bc.collect(dataset=dataset, task_id=f"{dataset.id}_bc-v0", steps=TRAIN_STEPS, save_interval=SAVE_INTERVAL, eval_interval=EVAL_INTERVAL, device=DEVICE)
    # train_cql.collect(dataset=dataset, task_id=f"{dataset.id}_cql-v0", steps=TRAIN_STEPS, save_interval=SAVE_INTERVAL, eval_interval=EVAL_INTERVAL, device=DEVICE)
    # train_iql.collect(dataset=dataset, task_id=f"{dataset.id}_iql-v0", steps=TRAIN_STEPS, save_interval=SAVE_INTERVAL, eval_interval=EVAL_INTERVAL, device=DEVICE)
    # train_scas.collect(dataset=dataset, task_id=f"{dataset.id}_scas-v0", dynamics_steps=TRAIN_STEPS, agent_steps=TRAIN_STEPS, save_interval=SAVE_INTERVAL, eval_interval=EVAL_INTERVAL, device=DEVICE)
    # train_aspl.collect(dataset=dataset, task_id=f"{dataset.id}_aspl-v0", steps=TRAIN_STEPS, save_interval=SAVE_INTERVAL, eval_interval=EVAL_INTERVAL, device=DEVICE)

    # dataset = HopperMediumDataset(device=DEVICE).load()
    # train_bc.collect(dataset=dataset, task_id=f"{dataset.id}_bc-v0", steps=TRAIN_STEPS, save_interval=SAVE_INTERVAL, eval_interval=EVAL_INTERVAL, device=DEVICE)
    # train_cql.collect(dataset=dataset, task_id=f"{dataset.id}_cql-v0", steps=TRAIN_STEPS, save_interval=SAVE_INTERVAL, eval_interval=EVAL_INTERVAL, device=DEVICE)
    # train_iql.collect(dataset=dataset, task_id=f"{dataset.id}_iql-v0", steps=TRAIN_STEPS, save_interval=SAVE_INTERVAL, eval_interval=EVAL_INTERVAL, device=DEVICE)
    # train_scas.collect(dataset=dataset, task_id=f"{dataset.id}_scas-v0", dynamics_steps=TRAIN_STEPS, agent_steps=TRAIN_STEPS, save_interval=SAVE_INTERVAL, eval_interval=EVAL_INTERVAL, device=DEVICE)
    # train_aspl.collect(dataset=dataset, task_id=f"{dataset.id}_aspl-v0", steps=TRAIN_STEPS, save_interval=SAVE_INTERVAL, eval_interval=EVAL_INTERVAL, device=DEVICE)

    # dataset = HopperExpertDataset(device=DEVICE).load()
    # train_bc.collect(dataset=dataset, task_id=f"{dataset.id}_bc-v0", steps=TRAIN_STEPS, save_interval=SAVE_INTERVAL, eval_interval=EVAL_INTERVAL, device=DEVICE)
    # train_cql.collect(dataset=dataset, task_id=f"{dataset.id}_cql-v0", steps=TRAIN_STEPS, save_interval=SAVE_INTERVAL, eval_interval=EVAL_INTERVAL, device=DEVICE)
    # train_iql.collect(dataset=dataset, task_id=f"{dataset.id}_iql-v0", steps=TRAIN_STEPS, save_interval=SAVE_INTERVAL, eval_interval=EVAL_INTERVAL, device=DEVICE)
    # train_scas.collect(dataset=dataset, task_id=f"{dataset.id}_scas-v0", dynamics_steps=TRAIN_STEPS, agent_steps=TRAIN_STEPS, save_interval=SAVE_INTERVAL, eval_interval=EVAL_INTERVAL, device=DEVICE)
    # train_aspl.collect(dataset=dataset, task_id=f"{dataset.id}_aspl-v0", steps=TRAIN_STEPS, save_interval=SAVE_INTERVAL, eval_interval=EVAL_INTERVAL, device=DEVICE)

    # dataset = HopperMediumD4rlDataset(device=DEVICE).load()
    # train_bc.collect(dataset=dataset, task_id=f"{dataset.id}_bc-v0", steps=TRAIN_STEPS, save_interval=SAVE_INTERVAL, eval_interval=EVAL_INTERVAL, device=DEVICE)
    # train_cql.collect(dataset=dataset, task_id=f"{dataset.id}_cql-v0", steps=TRAIN_STEPS, save_interval=SAVE_INTERVAL, eval_interval=EVAL_INTERVAL, device=DEVICE)
    # train_iql.collect(dataset=dataset, task_id=f"{dataset.id}_iql-v0", steps=TRAIN_STEPS, save_interval=SAVE_INTERVAL, eval_interval=EVAL_INTERVAL, device=DEVICE)
    # train_scas.collect(dataset=dataset, task_id=f"{dataset.id}_scas-v0", dynamics_steps=TRAIN_STEPS, agent_steps=TRAIN_STEPS, save_interval=SAVE_INTERVAL, eval_interval=EVAL_INTERVAL, device=DEVICE)
    # train_aspl.collect(dataset=dataset, task_id=f"{dataset.id}_aspl-v0", steps=TRAIN_STEPS, save_interval=SAVE_INTERVAL, eval_interval=EVAL_INTERVAL, device=DEVICE)

    # dataset = HopperMediumReplayDataset(device=DEVICE).load()
    # train_bc.collect(dataset=dataset, task_id=f"{dataset.id}_bc-v0", steps=TRAIN_STEPS, save_interval=SAVE_INTERVAL, eval_interval=EVAL_INTERVAL, device=DEVICE)
    # train_cql.collect(dataset=dataset, task_id=f"{dataset.id}_cql-v0", steps=TRAIN_STEPS, save_interval=SAVE_INTERVAL, eval_interval=EVAL_INTERVAL, device=DEVICE)
    # train_iql.collect(dataset=dataset, task_id=f"{dataset.id}_iql-v0", steps=TRAIN_STEPS, save_interval=SAVE_INTERVAL, eval_interval=EVAL_INTERVAL, device=DEVICE)
    # train_scas.collect(dataset=dataset, task_id=f"{dataset.id}_scas-v0", dynamics_steps=TRAIN_STEPS, agent_steps=TRAIN_STEPS, save_interval=SAVE_INTERVAL, eval_interval=EVAL_INTERVAL, device=DEVICE)
    # train_aspl.collect(dataset=dataset, task_id=f"{dataset.id}_aspl-v0", steps=TRAIN_STEPS, save_interval=SAVE_INTERVAL, eval_interval=EVAL_INTERVAL, device=DEVICE)

    # dataset = HopperMediumExpertDataset(device=DEVICE).load()
    # train_bc.collect(dataset=dataset, task_id=f"{dataset.id}_bc-v0", steps=TRAIN_STEPS, save_interval=SAVE_INTERVAL, eval_interval=EVAL_INTERVAL, device=DEVICE)
    # train_cql.collect(dataset=dataset, task_id=f"{dataset.id}_cql-v0", steps=TRAIN_STEPS, save_interval=SAVE_INTERVAL, eval_interval=EVAL_INTERVAL, device=DEVICE)
    # train_iql.collect(dataset=dataset, task_id=f"{dataset.id}_iql-v0", steps=TRAIN_STEPS, save_interval=SAVE_INTERVAL, eval_interval=EVAL_INTERVAL, device=DEVICE)
    # train_scas.collect(dataset=dataset, task_id=f"{dataset.id}_scas-v0", dynamics_steps=TRAIN_STEPS, agent_steps=TRAIN_STEPS, save_interval=SAVE_INTERVAL, eval_interval=EVAL_INTERVAL, device=DEVICE)
    # train_aspl.collect(dataset=dataset, task_id=f"{dataset.id}_aspl-v0", steps=TRAIN_STEPS, save_interval=SAVE_INTERVAL, eval_interval=EVAL_INTERVAL, device=DEVICE)
