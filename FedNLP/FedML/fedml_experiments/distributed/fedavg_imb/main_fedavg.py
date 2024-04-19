import argparse
import logging
import os
import random
import socket
import sys
import traceback

import numpy as np
import psutil
import setproctitle
import torch
import wandb
from mpi4py import MPI

# add the FedML root directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../../../")))

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))
from fedml_api.distributed.utils.gpu_mapping import mapping_processes_to_gpu_device_from_yaml_file
from fedml_api.data_preprocessing.FederatedEMNIST.data_loader_imb import load_partition_data_federated_emnist
from fedml_api.data_preprocessing.fed_cifar100.data_loader import load_partition_data_federated_cifar100
from fedml_api.data_preprocessing.fed_shakespeare.data_loader import load_partition_data_federated_shakespeare
from fedml_api.data_preprocessing.shakespeare.data_loader import load_partition_data_shakespeare
from fedml_api.data_preprocessing.stackoverflow_lr.data_loader import load_partition_data_federated_stackoverflow_lr
from fedml_api.data_preprocessing.stackoverflow_nwp.data_loader import load_partition_data_federated_stackoverflow_nwp
from fedml_api.data_preprocessing.MNIST.data_loader_imb import load_partition_data_mnist
from fedml_api.data_preprocessing.ImageNet.data_loader import load_partition_data_ImageNet
from fedml_api.data_preprocessing.Landmarks.data_loader import load_partition_data_landmarks

from fedml_api.data_preprocessing.cifar10.data_loader_imb import load_partition_data_cifar10
from fedml_api.data_preprocessing.cifar100.data_loader_imb import load_partition_data_cifar100
from fedml_api.data_preprocessing.cinic10.data_loader import load_partition_data_cinic10

from fedml_api.model.cv.cnn import CNN_DropOut, CNN_DropOut_ba
from fedml_api.model.cv.resnet_gn import resnet18
from fedml_api.model.cv.mobilenet import mobilenet
from fedml_api.model.cv.resnet import resnet56, resnet56_ba
from fedml_api.model.nlp.rnn import RNN_OriginalFedAvg, RNN_StackOverFlow
from fedml_api.model.linear.lr import LogisticRegression
from fedml_api.model.cv.mobilenet_v3 import MobileNetV3
from fedml_api.model.cv.efficientnet import EfficientNet

from fedml_api.distributed.fedavg_imb.FedAvgAPI import FedML_init, FedML_FedAvg_distributed


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--fl_algorithm', type=str, default="FedProx",
                        help='FedAvg')

    parser.add_argument('--model', type=str, default='mobilenet', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='./../../../data/cifar10',
                        help='data directory')

    parser.add_argument('--need_server_auxiliary_data', action='store_true', default=False,
                        help='whether need auxiliary data in the server')

    parser.add_argument('--server_auxiliary_data_per_class', type=int, default=32,
                        help='server auxiliary data per class')
    # decoupled for ba
    parser.add_argument('--decoupled_training', action='store_true', default=False,
                        help='decouple balanced classifier learning and bias attractor learning')

    parser.add_argument('--aggregate_ba', action='store_true', default=False,
                        help='whether aggregate the bias attractor in the server')

    parser.add_argument('--ba_lambda', type=float, default=0.01,
                        help='trade-off between normal gradients and balanced gradients')

    parser.add_argument('--ba_penalty', type=float, default=0.0,
                        help='penalty coefficient of biased classifier')

    parser.add_argument('--ba_local_balanced_data_num_per_class', type=int, default=1,
                        help='data number per class to construct the local balanced dataset')

    # cRT
    parser.add_argument('--retrain_server_classifier', action='store_true', default=False,
                        help='whether retrain the classifier in the server')

    parser.add_argument('--retrain_classifier_epochs', type=int, default=5,
                        help='epochs of retraining classifier in the server')

    parser.add_argument('--retrain_classifier_lr', type=float, default=0.1,
                        help='learning rate of retraining classifier in the server')

    parser.add_argument('--replace_classifier_prototypes', action='store_true', default=False,
                        help='whether replace the classifier weights with the prototypes')

    # for CLIMB
    parser.add_argument('--use_climb', action='store_true', default=False,
                        help='whether ruse CLIMB algorithm')

    parser.add_argument('--climb_dual_lr', type=float, default=0.1,
                        help='dual lr for updating CLIMB lambdas, 0.1 for CIFAR-10, 2 for MNIST')

    parser.add_argument('--climb_eps', type=float, default=0.1,
                        help='tolerance constant for CLIMB, 0.1 for CIFAR-10, 0.01 for MNIST')

    # for CReFF
    parser.add_argument('--use_creff', action='store_true', default=False,
                        help='whether use CReFF algorithm')

    parser.add_argument('--number_of_federated_features_per_class', type=int, default=100,
                        help='number of federated features per class for CReFF')

    parser.add_argument('--federated_features_optimization_steps', type=int, default=100,
                        help='number of optimization steps on federated features')

    parser.add_argument('--federated_features_optimization_lr', type=float, default=0.1,
                        help='LR during optimization on federated features')

    parser.add_argument('--creff_classifier_retraining_epochs', type=int, default=300,
                        help='number of epochs of re-training classifier')

    parser.add_argument('--creff_classifier_retraining_lr', type=float, default=0.1,
                        help='LR during CReFF classifier re-training')

    # partition args
    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')

    parser.add_argument('--imbalance_version', type=str, default='binary',
                        help='the global imbalance version: binary, exp_long_tail')

    parser.add_argument('--imbalanced_ratio', type=float, default=10.0, metavar='PA',
                        help='global imbalanced ratio (default: 10.0)')

    parser.add_argument('--minority_classes', type=str, default="0", metavar='PA',
                        help="minority classes, separated by '_' (e.g., 0_1_2)")

    parser.add_argument('--client_num_in_total', type=int, default=1000, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=4, metavar='NN',
                        help='number of workers')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')

    parser.add_argument('--backend', type=str, default="MPI",
                        help='Backend for Server and Client')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--local_loss', type=str, default='CE', metavar='LL',
                        help='local training loss (CE, Focal, Ratio)')

    parser.add_argument('--client_momentum', type=float, default=0.9,
                        help='client momentum (default: 0)')

    parser.add_argument('--server_lr', type=float, default=1.0,
                        help='server learning rate (default: 1.0)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--comm_round', type=int, default=10,
                        help='how many round of communications we shoud use')

    parser.add_argument('--is_mobile', type=int, default=1,
                        help='whether the program is running on the FedML-Mobile server side')

    parser.add_argument('--frequency_of_the_test', type=int, default=1,
                        help='the frequency of the algorithms')

    parser.add_argument('--gpu_server_num', type=int, default=1,
                        help='gpu_server_num')

    parser.add_argument('--gpu_num_per_server', type=int, default=4,
                        help='gpu_num_per_server')

    parser.add_argument('--gpu_mapping_file', type=str, default="gpu_mapping.yaml",
                        help='the gpu utilization file for servers and clients. If there is no \
                        gpu_util_file, gpu will not be used.')

    parser.add_argument('--gpu_mapping_key', type=str, default="mapping_default",
                        help='the key in gpu utilization file')

    parser.add_argument('--grpc_ipconfig_path', type=str, default="grpc_ipconfig.csv",
                        help='config table containing ipv4 address of grpc server')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')
    
    parser.add_argument('--fedprox_mu', type=float, default=0.1,
                        help='mu (default: 0.1)')

    # number of rounds to estimate initial optimal LR
    parser.add_argument('--init_lr_approx_clients', type=int, default=0, metavar='N',
                        help='number of clients to estimate initial optimal LR')

    # whether to use variance term to adjust lr
    parser.add_argument('--use_var_adjust', type=int, default=0, metavar='N',
                        help='whether to use variance term to adjust lr')

    # whether to enlarge initial lr due to larger global batch size
    parser.add_argument('--scale_server_lr', type=int, default=0, metavar='N',
                        help='whether to scale server lr due to larger global batch size')

    # whether use warm-up at the beginning
    parser.add_argument('--warmup_steps', type=int, default=0, metavar='N',
                        help='lr warm-up steps (i.e. rounds)')

    # the round idx when lr adjustment begins
    parser.add_argument('--var_adjust_begin_round', type=int, default=0, metavar='N',
                        help='the round idx when lr adjustment begins')

    # whether only adjust the lr for specific layers
    parser.add_argument('--only_adjusted_layer', type=str, default=None, metavar='N',
                        help='adjust the lr for specific layers (classifier or encoder)')

    # whether use re-weight technique
    parser.add_argument('--lr_bound_factor', type=float, default=0.0, metavar='N',
                        help='decides the bound of adjusted lr')

    # client sampling strategy
    parser.add_argument('--client_sampling_strategy', type=str, default='uniform', metavar='N',
                        help='client sampling strategy: uniform  / MD / AdaFL')

    # save model
    parser.add_argument('--save_global_model_during_training_rounds', type=int, default=0, metavar='N',
                        help='the number of rounds after which the global model saved')

    parser.add_argument('--save_global_model_directory', type=str, default='model_dir', metavar='N',
                        help='the directory in the which the global model saved')

    args = parser.parse_args()
    return args


def load_data(args, dataset_name):
    auxiliary_data_server = None
    if dataset_name == "mnist":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        # client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        #         # train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        #         # class_num = load_partition_data_mnist(args.batch_size)
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num, auxiliary_data_server = load_partition_data_mnist(args.dataset, args.data_dir, args.partition_method,
                                                                     args.partition_alpha, args.imbalance_version,
                                                                     args.imbalanced_ratio, args.minority_classes,
                                                                     args.client_num_in_total, args.batch_size,
                                                                     need_server_auxiliary_data=args.need_server_auxiliary_data,
                                                                     server_auxiliary_data_per_class=args.server_auxiliary_data_per_class,
                                                                     retrain_server_classifier=args.retrain_server_classifier)
        """
        For shallow NN or linear models, 
        we uniformly sample a fraction of clients each round (as the original FedAvg paper)
        """
        #args.client_num_in_total = client_num

    elif dataset_name == "femnist":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num, auxiliary_data_server = load_partition_data_federated_emnist(args.dataset, args.data_dir,
                                                                                need_server_auxiliary_data=args.need_server_auxiliary_data,
                                                                                server_auxiliary_data_per_class=args.server_auxiliary_data_per_class,
                                                                                retrain_server_classifier=args.retrain_server_classifier)
        args.client_num_in_total = client_num

    elif dataset_name == "shakespeare":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_shakespeare(args.batch_size)
        args.client_num_in_total = client_num

    elif dataset_name == "fed_shakespeare":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_shakespeare(args.dataset, args.data_dir)
        args.client_num_in_total = client_num

    elif dataset_name == "fed_cifar100":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_cifar100(args.dataset, args.data_dir)
        args.client_num_in_total = client_num
    elif dataset_name == "stackoverflow_lr":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_stackoverflow_lr(args.dataset, args.data_dir)
        args.client_num_in_total = client_num
    elif dataset_name == "stackoverflow_nwp":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_stackoverflow_nwp(args.dataset, args.data_dir)
        args.client_num_in_total = client_num
    elif dataset_name == "ILSVRC2012":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_ImageNet(dataset=dataset_name, data_dir=args.data_dir,
                                                 partition_method=None, partition_alpha=None,
                                                 client_number=args.client_num_in_total, batch_size=args.batch_size)

    elif dataset_name == "gld23k":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        args.client_num_in_total = 233
        fed_train_map_file = os.path.join(args.data_dir, 'mini_gld_train_split.csv')
        fed_test_map_file = os.path.join(args.data_dir, 'mini_gld_test.csv')
        args.data_dir = os.path.join(args.data_dir, 'images')

        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_landmarks(dataset=dataset_name, data_dir=args.data_dir,
                                                  fed_train_map_file=fed_train_map_file,
                                                  fed_test_map_file=fed_test_map_file,
                                                  partition_method=None, partition_alpha=None,
                                                  client_number=args.client_num_in_total, batch_size=args.batch_size)

    elif dataset_name == "gld160k":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        args.client_num_in_total = 1262
        fed_train_map_file = os.path.join(args.data_dir, 'federated_train.csv')
        fed_test_map_file = os.path.join(args.data_dir, 'test.csv')
        args.data_dir = os.path.join(args.data_dir, 'images')

        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_landmarks(dataset=dataset_name, data_dir=args.data_dir,
                                                  fed_train_map_file=fed_train_map_file,
                                                  fed_test_map_file=fed_test_map_file,
                                                  partition_method=None, partition_alpha=None,
                                                  client_number=args.client_num_in_total, batch_size=args.batch_size)


    else:
        if dataset_name == "cifar10":
            data_loader = load_partition_data_cifar10
        elif dataset_name == "cifar100":
            data_loader = load_partition_data_cifar100
        elif dataset_name == "cinic10":
            data_loader = load_partition_data_cinic10
        else:
            data_loader = load_partition_data_cifar10

        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num, auxiliary_data_server = data_loader(args.dataset, args.data_dir, args.partition_method,
                                                       args.partition_alpha, args.imbalance_version,
                                                       args.imbalanced_ratio, args.minority_classes,
                                                       args.client_num_in_total, args.batch_size,
                                                       need_server_auxiliary_data=args.need_server_auxiliary_data,
                                                       server_auxiliary_data_per_class=args.server_auxiliary_data_per_class,
                                                       retrain_server_classifier=args.retrain_server_classifier)
    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, auxiliary_data_server]
    return dataset


def create_model(args, model_name, output_dim):
    logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
    model = None
    if model_name == "lr" and args.dataset == "mnist":
        logging.info("LogisticRegression + MNIST")
        model = LogisticRegression(28 * 28, output_dim)
    elif model_name == "rnn" and args.dataset == "shakespeare":
        logging.info("RNN + shakespeare")
        model = RNN_OriginalFedAvg()
    elif model_name == "cnn" and args.dataset == "femnist":
        logging.info("CNN + FederatedEMNIST")
        # model = CNN_DropOut(False)
        if args.decoupled_training or args.use_creff:
            model = CNN_DropOut_ba(False, return_final_hidden_states=(args.decoupled_training or args.use_creff))
        else:
            model = CNN_DropOut(False, return_final_hidden_states=args.decoupled_training)
    elif model_name == "cnn" and args.dataset == "mnist":
        logging.info("CNN + MNIST")
        if args.decoupled_training or args.use_creff:
            model = CNN_DropOut_ba(True, return_final_hidden_states=(args.decoupled_training or args.use_creff))
        else:
            model = CNN_DropOut(True, return_final_hidden_states=args.decoupled_training)
    elif model_name == "resnet18_gn" and args.dataset == "fed_cifar100":
        logging.info("ResNet18_GN + Federated_CIFAR100")
        model = resnet18()
    elif model_name == "rnn" and args.dataset == "fed_shakespeare":
        logging.info("RNN + fed_shakespeare")
        model = RNN_OriginalFedAvg()
    elif model_name == "lr" and args.dataset == "stackoverflow_lr":
        logging.info("lr + stackoverflow_lr")
        model = LogisticRegression(10004, output_dim)
    elif model_name == "rnn" and args.dataset == "stackoverflow_nwp":
        logging.info("CNN + stackoverflow_nwp")
        model = RNN_StackOverFlow()
    elif model_name == "resnet56":
        if args.decoupled_training or args.use_creff:
            model = resnet56_ba(class_num=output_dim, return_final_hidden_states=(args.decoupled_training or args.use_creff))
        else:
            model = resnet56(class_num=output_dim, return_final_hidden_states=args.decoupled_training)
    elif model_name == "mobilenet":
        model = mobilenet(class_num=output_dim)
    # TODO
    elif model_name == 'mobilenet_v3':
        '''model_mode \in {LARGE: 5.15M, SMALL: 2.94M}'''
        model = MobileNetV3(model_mode='LARGE')
    elif model_name == 'efficientnet':
        model = EfficientNet()

    return model


if __name__ == "__main__":
    # quick fix for issue in MacOS environment: https://github.com/openai/spinningup/issues/16
    if sys.platform == 'darwin':
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # initialize distributed computing (MPI)
    comm, process_id, worker_number = FedML_init()

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logging.info(args)

    # customize the process name
    str_process_name = "FedAvg (distributed):" + str(process_id)
    setproctitle.setproctitle(str_process_name)

    # customize the log format
    # logging.basicConfig(level=logging.INFO,
    logging.basicConfig(level=logging.DEBUG,
                        format=str(
                            process_id) + ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')
    hostname = socket.gethostname()
    logging.info("#############process ID = " + str(process_id) +
                 ", host name = " + hostname + "########" +
                 ", process ID = " + str(os.getpid()) +
                 ", process Name = " + str(psutil.Process(os.getpid())))

    # initialize the wandb machine learning experimental tracking platform (https://www.wandb.com/).
    if process_id == 0:
        wandb.init(
            # project="federated_nas",
            project="fedimb",
            entity='YourEntityName',
            name="FedAVG(d)" + " partition=" + args.partition_method + " alpha=" + str(args.partition_alpha) + " S_LR=" + str(args.server_lr) +
                   " C_LR=" + str(args.lr) + " C_OPT=" + str(args.client_optimizer) + " C_MOM=" + str(args.client_momentum) +
                   " local_epochs=" + str(args.epochs) +
                   " use_var_adjust=" + str(args.use_var_adjust) + " scale_server_lr=" + str(args.scale_server_lr) +
                   " approx_clients=" + str(args.init_lr_approx_clients) + " minority_classes=" + str(args.minority_classes) +
                   " imbalanced_ratio=" + str(args.imbalanced_ratio) + " local_loss=" + str(args.local_loss)
                 + " only_adjusted_layer=" + str(args.only_adjusted_layer) +
                   " lr_bound_factor=" + str(args.lr_bound_factor) + " decoupled_training=" + str(args.decoupled_training) +
                   " aggregate_ba=" + str(args.aggregate_ba) + " ba_lambda=" + str(args.ba_lambda) +
                   " ba_penalty=" + str(args.ba_penalty) + " use_CLIMB=" + str(args.use_climb) + " use_CReFF=" + str(args.use_creff),
            config=args
        )

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Please check "GPU_MAPPING.md" to see how to define the topology
    logging.info("process_id = %d, size = %d" % (process_id, worker_number))
    device = mapping_processes_to_gpu_device_from_yaml_file(process_id, worker_number, args.gpu_mapping_file, args.gpu_mapping_key)

    # load data
    # auxiliary_data_server: {class_id: dataloader}
    dataset = load_data(args, args.dataset)
    [train_data_num, test_data_num, train_data_global, test_data_global,
     train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, auxiliary_data_server] = dataset

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    model = create_model(args, model_name=args.model, output_dim=100)

    # try:
        # start "federated averaging (FedAvg)"
    FedML_FedAvg_distributed(process_id, worker_number, device, comm,
                             model, train_data_num, train_data_global, test_data_global,
                             train_data_local_num_dict, train_data_local_dict, test_data_local_dict, args, auxiliary_data_server=auxiliary_data_server)
    # except Exception as e:
    #     print(e)
    #     logging.info('traceback.format_exc():\n%s' % traceback.format_exc())
    #     MPI.COMM_WORLD.Abort()
