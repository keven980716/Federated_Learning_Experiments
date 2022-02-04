# Our updates

Compared with the original FedNLP, we only make some updates in the file ``main_tc.py`` and ``initializer.py``.

## Run Experiments
Here are some examples to run experiments:
### 20NewsGroups
```python
DATA_NAME=20news
CUDA_VISIBLE_DEVICES=1 python -m main_tc \
    --dataset ${DATA_NAME} \
    --data_file ~/fednlp_data/data_files/${DATA_NAME}_data.h5 \
    --partition_file ~/fednlp_data/partition_files/${DATA_NAME}_partition.h5 \
    --partition_method uniform \
    --model_type distilbert \
    --model_name distilbert-base-uncased  \
    --do_lower_case True \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --max_seq_length 128 \
    --learning_rate 5e-5 --client_optimizer adam \
    --epochs 20 \
    --evaluate_during_training_steps 200 \
    --output_dir /tmp/${DATA_NAME}_fed/ \
    --n_gpu 1
```

### AGNews
```python
DATA_NAME=agnews
CUDA_VISIBLE_DEVICES=0 python -m main_tc \
    --dataset ${DATA_NAME} \
    --data_file ~/fednlp_data/data_files/${DATA_NAME}_data.h5 \
    --partition_file ~/fednlp_data/partition_files/${DATA_NAME}_partition.h5 \
    --partition_method uniform \
    --model_type distilbert \
    --model_name distilbert-base-uncased  \
    --do_lower_case True \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --max_seq_length 128 \
    --learning_rate 2e-5 --client_optimizer adam \
    --epochs 10 \
    --evaluate_during_training_steps 200 \
    --output_dir /tmp/${DATA_NAME}_fed/ \
    --n_gpu 1
```

