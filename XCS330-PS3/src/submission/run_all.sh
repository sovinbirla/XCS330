#!/bin/bash

ARG_DEVICE=--device=${1:-'cpu'}
ARG_BATCH_SIZE=--batch_size=${2:-16}
ARG_COMPILE=--${3:-'no-compile'}
ARG_BACKEND=--backend=${4:-'inductor'}

# Protonet experiments

# Experiment 1: takes around 35 minutes on the provided VM with a Tesla K80 GPU
python protonet.py --num_support 5 $ARG_DEVICE $ARG_BATCH_SIZE $ARG_COMPILE $ARG_BACKEND

# Experiment 2: takes around 30 minutes on the provided VM with a Tesla K80 GPU
python protonet.py $ARG_DEVICE $ARG_BATCH_SIZE $ARG_COMPILE $ARG_BACKEND

# Maml experiments

# Experiment 1: takes around 2h35 minutes on the provided VM with a Tesla K80 GPU
python maml.py $ARG_DEVICE $ARG_BATCH_SIZE

# Experiment 2: takes around 2h35 minutes on the provided VM with a Tesla K80 GPU
python maml.py --inner_lr 0.04 $ARG_DEVICE $ARG_BATCH_SIZE

# Experiment 3: takes around 7h35 minutes on the provided VM with a Tesla K80 GPU
python maml.py --inner_lr 0.04 --num_inner_steps 5 $ARG_DEVICE $ARG_BATCH_SIZE

# Experiment 4: takes around 2h35 minutes on the provided VM with a Tesla K80 GPU
python maml.py --learn_inner_lrs $ARG_DEVICE $ARG_BATCH_SIZE