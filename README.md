# A Research Platform for Federated Learning Experiments 

## Introduction

This is a research platform for conducting federated (machine) learning experiments. This code is mainly based on the two open-sourced federated learning platforms **[FedML](https://github.com/FedML-AI/FedML)** and **[FedNLP](https://github.com/FedML-AI/FedNLP)**, while we will continue to make updates for the broader research (e.g., add more federated optimization methods). **If you find this code helpful for your research, please also do not forget to cite the papers of above two platforms**.

## Overview

The code structure of this repo is consistent with that of FedNLP. The directory ``FedML`` is in the ``FedNLP``. Based on the FedML and FedNLP, we make the following important updates:

+ **[2022.1]**
  - Besides the default client sampling procedure (i.e., *uniform sampling without replacement*), we implement another two sampling strategies in our experiments: (1) [Multinomial Distribution-based Sampling (MD)](https://arxiv.org/abs/2107.12211), and (2) [Dynamic Attention-based Sampling (AdaFL)](https://arxiv.org/abs/2108.05765).
+ **[2021.12]**
  - Fix bugs about distributed experiments on BiLSTM, and allow to perform federated learning experiments on BiLSTM.
  - Add the code for the new federated optimization [SCAFFOLD](https://arxiv.org/abs/1910.06378).
+ **[2021.11]**
  - Adopt the code about the non-i.i.d. label-skewed partitioning procedure from ``FedNLP/data/advanced_partition/niid_label.py``  for image classification datasets (only CIFAR-10 and MNIST now). The original partitioning code for CIFAR-10 and MNIST also consider the quantity-skewed cases (i.e., clients have different numbers of training samples), while in our experiments we only consider the label shift problem. You can also use the original code by following the comments in Line 449-455 in ``FedNLP/FedML/fedml_api/data_preprocessing/cifar10/data_loader.py`` (take CIFAR-10 as an example, the code for MNIST is in ``FedNLP/FedML/fedml_api/data_preprocessing/MNIST/data_loader_new.py`` ).
  - Make updates to allow clients to use any of SGD, SGDM and Adam as the local optimizer for the local training.
  - Implement the code for FedGLAD. The core code is mainly in the files ``FedAVGAggregator.py``, ``FedProxAggregator.py`` and ``FedOptAggregator.py``.
+ **[2021.10]**
  - For text classification tasks with DistilBERT, fix bugs about building local (client) optimizers in the ``FedNLP/training/tc_transformer_trainer.py``.
+ **[2021.9]**
  - Fix bugs about the label-skewed partitioning commands for text classification datasets in the directory ``FedNLP/data/advanced_partition``.
  - For text classification tasks, allow each client to randomly shuffle its local data before local training begins. That is because after data partitioning, each client's local data is arranged in a label order and is not shuffled. The corresponding code is in the ``FedNLP/data_manager/base_data_manager.py`` and ``FedNLP/data_manager/text_classification_manager.py``.

## Usage

### Installation

You can first follow the instructation in [FedNLP](https://github.com/FedML-AI/FedNLP) to install FedNLP and FedML at the same time, then copy our code into the downloaded directory.

Also, after `git clone`-ing this repository, you run the following command for installation:

```bash
conda create -n fednlp python=3.7
conda activate fednlp
# conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch -n fednlp
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt 
pip uninstall transformers
pip install -e transformers/
# cd FedML; git submodule init; git submodule update; cd ../;
```

#### Problem when pip install mpi4py

If there is an error when installing mpi4py, please run
```bash
sudo apt install libopenmpi-dev
pip install mpi4py
```

#### Login wandb

We follow the FedML and FedNLP to use [wandb](https://wandb.ai/site) for experiment tracking, so please login your wandb account, and modify the argument **--entity** in ``wandb.init()`` to your user name. Specifically,

(1) First, go to [https://wandb.ai/authorize](https://wandb.ai/authorize) to create an account (with Github account).

(2) Then, use the command line

```bash
wandb login
```

(3) Paste the api key to login.

After login the wandb, you may still get the error like:
*'wandb.errors.error.CommError: Permission denied, ask the project owner to grant you access'*.
Take the text classification task as an example, go to ``FedNLP/experiments/centralized/transformer_exps/main_tc.py``, in **Line 47**, modify the argument `entity` to your user name of wandb.

### Code Structure

The code structure is consistent with that of FedNLP, so you can follow the instructions in FedNLP to use our code.

### Usage of FedGLAD

Based on the existing federated optimization methods, we implement the code of FedGLAD. There are several important arguments when using FedGLAD:

(1) **--use_var_adjust**: value chosen from {0, 1}. Setting 1 means using FedGLAD, and setting 0 represents using the original 
baseline without server learning rate adaptation.

(2) **--only_adjusted_layer**: value chosen from {'group', 'none'}. Setting 'group' means using the parameter group–wise
adaptation, and setting 'none' represents the universal adaptation.

(3) **--lr_bound_factor**: the value of the bounding factor gamma. Default is 0.02.

(4) **--client_sampling_strategy**: the choice of the client sampling strategy, can be chosen from {'uniform', 'MD', 'AdaFL'}.

We provide some bash scripts for examples to help to conduct quick experiments (e.g., ``FedNLP/experiments/distributed/transformer_exps/run_tc_exps/run_text_classification.sh``). 

## Citation

If you find this code helpful for your research, please cite our work as:

```

```

Also, don't forget to cite FedNLP and FedML as

```
@inproceedings{fednlp2021,
  title={FedNLP: A Research Platform for Federated Learning in Natural Language Processing},
  author={Bill Yuchen Lin and Chaoyang He and ZiHang Zeng and Hulin Wang and Yufen Huang and M. Soltanolkotabi and Xiang Ren and S. Avestimehr},
  year={2021},
  booktitle={arXiv cs.CL 2104.08815},
  url={https://arxiv.org/abs/2104.08815}
}
```

```
@article{chaoyanghe2020fedml,
  Author = {He, Chaoyang and Li, Songze and So, Jinhyun and Zhang, Mi and Wang, Hongyi and Wang, Xiaoyang and Vepakomma, Praneeth and Singh, Abhishek and Qiu, Hang and Shen, Li and Zhao, Peilin and Kang, Yan and Liu, Yang and Raskar, Ramesh and Yang, Qiang and Annavaram, Murali and Avestimehr, Salman},
  Journal = {arXiv preprint arXiv:2007.13518},
  Title = {FedML: A Research Library and Benchmark for Federated Machine Learning},
  Year = {2020}
}
```
