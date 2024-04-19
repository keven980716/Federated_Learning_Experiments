# Instructions to conduct federated long-tailed learning experiments

## Introduction
This file is to help run the experiments included in the paper [Decentralized Decoupled Training for Federated Long-Tailed Learning](https://openreview.net/forum?id=hw7inQwRxB).

Specifically, this paper includes 5 federated long-tailed learning methods: (1) [Fed-Focal Loss](https://arxiv.org/abs/2011.06283); (2) [Ratio Loss](https://ojs.aaai.org/index.php/AAAI/article/view/17219); (3) [CLIMB](https://openreview.net/forum?id=Xo0lbDt975); (4) [CReFF](https://www.ijcai.org/proceedings/2022/0308.pdf); (5) [RedGrape](https://openreview.net/forum?id=hw7inQwRxB). Moreover, it conducts experiment on 4 federated long-tailed learning benchmarks: MNIST-LT, CIFAR-10/100-LT, FEMNIST.

## Details
There are some main arguments in ``main_fedavg.py`` for each of the above methods:

### For Fed-Focal Loss

(1) **--local_loss**: set 'Focal' to use Fed-Focal Loss

### For Ratio Loss

(1) **--local_loss**: set 'Ratio' to use Ratio Loss

### For CLIMB
(1) **--use_climb**: set True to use CLIMB

(2) **--climb_dual_lr**: dual lr for updating CLIMB lambdas

(3) **--climb_eps**: tolerance constant in CLIMB

### For CReFF
(1) **--use_creff**: set True to use CReFF

(2) **--number_of_federated_features_per_class**: number of federated features per class for CReFF

(3) **--federated_features_optimization_steps**: number of optimization steps on federated features

(4) **--federated_features_optimization_lr**: LR during optimization on federated features

(5) **--creff_classifier_retraining_epochs**: number of epochs of re-training classifier

(6) **--creff_classifier_retraining_lr**: LR during CReFF classifier re-training

### For RedGrape

(1) **--decoupled_training**: set True to use RedGrape

(2) **--aggregate_ba**: True for also aggregating the newly introduced supplementary classifier on the server side in each round


(3) **---ba_lambda**: the re-balancing factor $\lambda$

(4) **--ba_local_balanced_data_num_per_class**: the threshold $T$ of the data size per class to construct the local balanced dataset


### Other basic settings
(1) **--imbalance_version**: value chosen from {'exp_long_tail', 'binary'} for two different long-tailed data settings

(2) **--imbalanced_ratio**: the imbalance ratio between the maximum sample number across all classes and the minimum sample number across all classes

(3) **--minority_classes**: manually specified minority classes (separated by '_', e.g., 0_1_2), can be used in the 'binary' imbalanced data setting

(4) **--need_server_auxiliary_data**: True for methods (e.g., Ratio Loss) that require auxiliary balanced dataset

(5) **--server_auxiliary_data_per_class**: the data size per class required in the auxiliary balanced dataset



Other arguments follow the original FedML logics.

## Example
An example to run the experiment of RedGrape on CIFAR-10-LT under the partial client participation and exp_long_tail long-tailed data setting with imbalance ratio 100 can be

```bash
CUDA_VISIBLE_DEVICES=0,1 sh run_fedavg_distributed_pytorch.sh 50 10 resnet56 hetero 1000 5 64 0.1 cifar10 "./../../../data/cifar10" sgd 0 1.0
```

based on the current commands in `` run_fedavg_imb_distributed_pytorch.sh``.
