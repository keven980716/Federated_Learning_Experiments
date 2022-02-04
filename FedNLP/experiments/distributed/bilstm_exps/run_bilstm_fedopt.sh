CLIENT_NUM=100
WORKER_NUM=10
SERVER_NUM=1
GPU_NUM_PER_SERVER=4
CI=0

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM
HOST_FILE=mpi_host_file
hostname > $HOST_FILE

mpirun -np $PROCESS_NUM -hostfile $HOST_FILE \
python -m main_fedopt \
    --gpu_num_per_server $GPU_NUM_PER_SERVER \
    --gpu_server_num $SERVER_NUM \
    --gpu_mapping_file gpu_mapping.yaml \
    --gpu_mapping_key mapping_config1_11 \
    --partition_method niid_label_clients=100_alpha=0.1 \
    --dataset 20news \
    --data_file ~/fednlp_data/data_files/20news_data.h5 \
    --partition_file ~/fednlp_data/partition_files/20news_partition.h5 \
    --embedding_file ~/fednlp_data/glove/glove.6B.300d.txt \
    --client_num_in_total $CLIENT_NUM \
    --client_num_per_round $WORKER_NUM \
    --comm_round 100 \
    --epochs 1 \
    --batch_size 32 \
    --lr 0.001 \
    --server_lr 0.5 \
    --client_optimizer adam --server_optimizer sgd \
    --wd 0.0001 \
    --lstm_dropout 0.2 \
    --embedding_dropout 0 \
    --max_seq_len 128 \
    --do_remove_stop_words True \
    --do_remove_low_freq_words 5 \
    --output_dir "tmp/fedopt_${DATA_NAME}_output/" \
    --ci $CI \
    --init_lr_approx_clients 0 --use_var_adjust 0 \
    --scale_server_lr 0 --warmup_steps 0 \
    --var_adjust_begin_round 0 --only_adjusted_layer group \
    --use_reweight 0 \
    --lr_bound_factor 0.02 --manual_seed 0