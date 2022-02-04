

# Our updates

+ Besides the original code for FedAvg, we add the code for FedOPT to make it applicable on distributed BiLSTM experiments.

+ We only make the updates to allow conducting experiments on 20NewsGroups and AGNews. To successfuly conduct experiments on other datasets, please modify the code of other datasets in ``FedNLP/data/raw_data_loader/obsolete/`` by following the example we have made for 20NewsGroups and AGNews in ``FedNLP/data/raw_data_loader/obsolete/news_20/data_loader.py`` and ``FedNLP/data/raw_data_loader/obsolete/AGNews/data_loader.py`` .

+ We add the code for FedGLAD. There are several important arguments when using FedGLAD:

  (1) **--use_var_adjust**: value chosen from {0, 1}. Setting 1 means using FedGLAD, and setting 0 represents using the original 
  baseline without server learning rate adaptation.

  (2) **--only_adjusted_layer**: value chosen from {'group', 'none'}. Setting 'group' means using the parameter group–wise
  adaptation, and setting 'none' represents the universal adaptation.

  (3) **--lr_bound_factor**: the value of the bounding factor gamma. Default is 0.02.

  (4) **--client_sampling_strategy**: the choice of the client sampling strategy, can be chosen from {'uniform', 'MD', 'AdaFL'}.

  



To run experiments, we provide a bash script, and examples can be

```bash
CUDA_VISIBLE_DEVICES=0,1 sh run_bilstm_fedavg.sh
```

```bash
CUDA_VISIBLE_DEVICES=0,1 sh run_bilstm_fedopt.sh
```



# Following content is from the original README.md

## Installation

http://doc.fedml.ai/#/installation-distributed-computing

## Experimental Tracking
```
pip install --upgrade wandb
wandb login ee0b5f53d949c84cee7decbe7a629e63fb2f8408
```

## How to Run
### Example

```bash
CLIENT_NUM=10
WORKER_NUM=10
SERVER_NUM=1
GPU_NUM_PER_SERVER=4
CI=0

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM
HOST_FILE=experiments/distributed/bilstm_exps/mpi_host_file
hostname > $HOST_FILE

mpirun -np $PROCESS_NUM -hostfile $HOST_FILE \
python -m experiments.distributed.bilstm_exps.main_fedopt \
    --gpu_num_per_server $GPU_NUM_PER_SERVER \
    --gpu_server_num $SERVER_NUM \
    --dataset 20news \
    --data_file data/data_files/20news_data.h5 \
    --partition_file data/partition_files/20news_partition.h5 \
    --embedding_file data/pretrained/glove.6B.300d.txt \
    --client_num_in_total $CLIENT_NUM \
    --client_num_per_round $WORKER_NUM \
    --comm_round 100 \
    --epochs 1 \
    --batch_size 32 \
    --lr 0.005 \
    --server_lr 0.1 \
    --optimizer adam \
    --server_optimizer sgd \
    --wd 0.0001 \
    --lstm_dropout 0.5 \
    --embedding_dropout 0 \
    --max_seq_len 512 \
    --do_remove_stop_words True \
    --do_remove_low_freq_words 5 \
    --ci $CI
```

## Experiment Parameters
| Dataset | Model | Embedding | comm_round(fedavg) | comm_round(fedopt) | batch_size | lr | wd | lstm_dropout | embedding_dropout | max_seq_len | do_remove_stop_words | do_remove_low_freq_words | server_lr(fedopt) |
| ------- | ------ | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| 20news | BiLSTM | glove.6b.300d | 100 | 500 | 32 | 0.005 | 0.0001 | 0.5 | 0 | 512 | True | 5 | 0.1 |
| agnews | BiLSTM | glove.6b.300d | 60 | 500 | 128 | 0.005 | 0.0001 | 0.1 | 0 | 128 | False | 0 | 0.1 |
| semeval_2010_task8 | BiLSTM | glove.840b.300d | 60 | 400 | 10 | 0.005 | 0.0001 | 0.5 | 0.3 | 100 | False | 0 | 0.1 |
| sentiment140 | BiLSTM | glove.840b.300d | 50 | 100 | 256 | 0.001 | 0.0005 | 0.5 | 0.3 | 32 | False | 5 | 0.1 |
| sst_2 | BiLSTM | glove.840b.300d | 100 | 32 | 400 | 0.005 | 0.0001 | 0.5 | 0.3 | 32 | False | 0 | 0.1 |

## Experiment Results
| Dataset | Model | Embedding | Metrics | Centralized | FedAvg | FedOpt | FedAvg Time(s) |
| ------- | ------ | ------- | ------- | ------- | ------- | ------- | ------- |
| 20news | BiLSTM | glove.6b.300d | Accuracy| 78% | 78% | 77.4% | 1853 |
| agnews | BiLSTM | glove.6b.300d | Accuracy| 91.5% | 91.5% | 91.5% | 727 |
| semeval_2010_task8 | BiLSTM | glove.840b.300d | Accuracy | 74% | 74% | 74% | 346 |
| sentiment140 | BiLSTM | glove.840b.300d | Accuracy| 84.5% | 84.5% | 84% | 2285 |
| sst_2 | BiLSTM | glove.840b.300d | Accuracy | 85.5% | 85.5% | 842..5% | 361 |