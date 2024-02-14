#!/bin/bash

ARG_DEVICE=--device=${1:-'gpu'}

ARG_COMPILE=--${2:-'no-compile'}

ARG_BACKEND=--backend=${3:-'inductor'}

# Experiment 1
python main.py --factorization_weight 0.99 --regression_weight 0.01 --logdir run/shared=True_LF=0.99_LR=0.01 $ARG_DEVICE $ARG_COMPILE $ARG_BACKEND 

# Experiment 2
python main.py --factorization_weight 0.5 --regression_weight 0.5 --logdir run/shared=True_LF=0.5_LR=0.5 $ARG_DEVICE $ARG_COMPILE $ARG_BACKEND

# Experiment 3
python main.py --no-shared_embeddings --factorization_weight 0.5 --regression_weight 0.5 --logdir run/shared=False_LF=0.5_LR=0.5 $ARG_DEVICE $ARG_COMPILE $ARG_BACKEND

# Experiment 4
python main.py --no-shared_embeddings --factorization_weight 0.99 --regression_weight 0.01 --logdir run/shared=False_LF=0.99_LR=0.01 $ARG_DEVICE $ARG_COMPILE $ARG_BACKEND