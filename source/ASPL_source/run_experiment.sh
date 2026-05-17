#!/bin/bash

# Experiment configuration
ENV="hopper-medium-expert-v2"

SEED=3

python main.py --env ${ENV} --seed ${SEED}