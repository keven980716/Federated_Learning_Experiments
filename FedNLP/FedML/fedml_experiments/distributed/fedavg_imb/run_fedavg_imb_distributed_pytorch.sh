#!/usr/bin/env bash

CLIENT_NUM=$1
WORKER_NUM=$2
MODEL=$3
DISTRIBUTION=$4
ROUND=$5
EPOCH=$6
BATCH_SIZE=$7
LR=$8
DATASET=$9
DATA_DIR=${10}
CLIENT_OPTIMIZER=${11}
CI=${12}
SLR=${13}

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 ./main_fedavg.py \
  --gpu_mapping_file "gpu_mapping.yaml" \
  --gpu_mapping_key mapping_config1_11 \
  --model $MODEL \
  --dataset $DATASET \
  --data_dir $DATA_DIR \
  --partition_method $DISTRIBUTION --partition_alpha 1.0 \
  --client_num_in_total $CLIENT_NUM \
  --client_num_per_round $WORKER_NUM \
  --comm_round $ROUND \
  --epochs $EPOCH \
  --client_optimizer $CLIENT_OPTIMIZER --client_momentum 0.9  \
  --local_loss CE \
  --need_server_auxiliary_data --server_auxiliary_data_per_class 32 \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --ci $CI \
  --server_lr $SLR \
  --init_lr_approx_clients 0 --use_var_adjust 0 \
  --scale_server_lr 0 --warmup_steps 0 \
  --var_adjust_begin_round 0 --only_adjusted_layer group \
  --lr_bound_factor 0.02 --client_sampling_strategy uniform \
  --imbalanced_ratio 100.0 \
  --imbalance_version exp_long_tail \
  --decoupled_training --ba_lambda 0.1 --aggregate_ba \
  --ba_local_balanced_data_num_per_class 8 \
  # --use_climb --climb_dual_lr 0.1 --climb_eps 0.1
  # --decoupled_training --ba_lambda 0.1 --aggregate_ba \
  # --ba_local_balanced_data_num_per_class 2 \
  # --save_global_model_during_training_rounds 500 \
  # --save_global_model_directory femnist/ba \


