

# Our updates

+ We update the main file ``fedavg_main_tc.py``, this file can be used for FedAvg, FedProx, SCAFFOLD and FedOPT.

+ Please check the arguments in ``fedavg_main_tc.py`` to adjust the experimental settings in ``run_text_classification.sh``.

+ We add the code for FedGLAD. There are several important arguments when using FedGLAD:

  (1) **--use_var_adjust**: value chosen from {0, 1}. Setting 1 means using FedGLAD, and setting 0 represents using the original 
  baseline without server learning rate adaptation.

  (2) **--only_adjusted_layer**: value chosen from {'group', 'none'}. Setting 'group' means using the parameter group–wise
  adaptation, and setting 'none' represents the universal adaptation.

  (3) **--lr_bound_factor**: the value of the bounding factor gamma. Default is 0.02.

  (4) **--client_sampling_strategy**: the choice of the client sampling strategy, can be chosen from {'uniform', 'MD', 'AdaFL'}.

  

To run experiments, we provide a bash script, and an example on 20NewsGroups (with 2*NVIDIA TITAN RTX) can be

```bash
CUDA_VISIBLE_DEVICES=0,1 sh run_text_classification.sh FedAvg "niid_label_clients=100_alpha=0.1" 5e-5 7.0 100 10
```

```bash
CUDA_VISIBLE_DEVICES=0,1 sh run_text_classification.sh FedProx "niid_label_clients=100_alpha=0.1" 5e-5 7.0 100 10
```

