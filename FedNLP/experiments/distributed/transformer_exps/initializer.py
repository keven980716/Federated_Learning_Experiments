import random

import numpy as np
import torch
from transformers import (
    BertConfig,
    BertTokenizer,
    BertForTokenClassification,
    BertForQuestionAnswering,
    DistilBertConfig,
    DistilBertTokenizer,
    DistilBertForTokenClassification,
    DistilBertForQuestionAnswering,
    BartConfig, 
    BartForConditionalGeneration, 
    BartTokenizer,
)

from FedML.fedml_api.distributed.fedavg.FedAvgAPI import FedML_FedAvg_distributed
from FedML.fedml_api.distributed.fedopt.FedOptAPI import FedML_FedOpt_distributed
from FedML.fedml_api.distributed.fedprox.FedProxAPI import FedML_FedProx_distributed
from FedML.fedml_api.distributed.scaffold.SCAFFOLDAPI import FedML_SCAFFOLD_distributed
from model.transformer.bert_model import BertForSequenceClassification
from model.transformer.distilbert_model import DistilBertForSequenceClassification


def get_fl_algorithm_initializer(alg_name):
    if alg_name == "FedAvg":
        fl_algorithm = FedML_FedAvg_distributed
    elif alg_name == "FedOPT":
        fl_algorithm = FedML_FedOpt_distributed
    elif alg_name == "FedProx":
        fl_algorithm = FedML_FedProx_distributed
    elif alg_name == 'SCAFFOLD':
        fl_algorithm = FedML_SCAFFOLD_distributed
    else:
        raise Exception("please do sanity check for this algorithm.")

    return fl_algorithm


def create_model(args, formulation="classification"):
    # create model, tokenizer, and model config (HuggingFace style)
    MODEL_CLASSES = {
        "classification": {
            "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
            "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
            # "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
            # "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
        },
        "seq_tagging": {
            "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
            "distilbert": (DistilBertConfig, DistilBertForTokenClassification, DistilBertTokenizer),
        },
        "span_extraction": {
            "bert": (BertConfig, BertForQuestionAnswering, BertTokenizer),
            "distilbert": (DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer),
        },
        "seq2seq": {
            "bart": (BartConfig, BartForConditionalGeneration, BartTokenizer),
        }
    }
    config_class, model_class, tokenizer_class = MODEL_CLASSES[formulation][
        args.model_type]
    # config = config_class.from_pretrained(
    #     args.model_name, num_labels=args.num_labels, **args.config)
    config = config_class.from_pretrained(args.model_name, **args.config)
    model = model_class.from_pretrained(args.model_name, config=config)
    if formulation != "seq2seq":
        tokenizer = tokenizer_class.from_pretrained(
            args.model_name, do_lower_case=args.do_lower_case)
    else:
        tokenizer = [None, None]
        tokenizer[0] = tokenizer_class.from_pretrained(args.model_name)
        tokenizer[1]= tokenizer[0]
    # logging.info(self.model)
    return config, model, tokenizer


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def add_federated_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # # PipeTransformer related
    parser.add_argument("--run_id", type=int, default=0)

    parser.add_argument("--is_debug_mode", default=0, type=int,
                        help="is_debug_mode")

    # Data related
    # TODO: list all dataset names:
    parser.add_argument('--dataset', type=str, default='agnews', metavar='N',
                        help='dataset used for training')

    parser.add_argument(
        '--data_file_path', type=str,
        default='/home/bill/fednlp_data/data_files/agnews_data.h5',
        help='data h5 file path')

    parser.add_argument(
        '--partition_file_path', type=str,
        default='/home/bill/fednlp_data/partition_files/agnews_partition.h5',
        help='partition h5 file path')

    parser.add_argument('--partition_method', type=str, default='uniform',
                        help='partition method')

    # Model related
    parser.add_argument('--model_type', type=str, default='bert', metavar='N',
                        help='transformer model type')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', metavar='N',
                        help='transformer model name')
    parser.add_argument('--do_lower_case', type=bool, default=True, metavar='N',
                        help='transformer model name')

    # Learning related
    parser.add_argument('--train_batch_size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--eval_batch_size', type=int, default=8, metavar='N',
                        help='input batch size for evaluation (default: 8)')

    parser.add_argument('--max_seq_length', type=int, default=128, metavar='N',
                        help='maximum sequence length (default: 128)')

    parser.add_argument('--n_gpu', type=int, default=1, metavar='EP',
                        help='how many gpus will be used ')

    parser.add_argument('--fp16', default=False, action="store_true",
                        help='if enable fp16 for training')
    parser.add_argument('--manual_seed', type=int, default=42, metavar='N',
                        help='random seed')

    # IO related
    parser.add_argument('--output_dir', type=str, default="/tmp/", metavar='N',
                        help='path to save the trained results and ckpts')

    # Federated Learning related
    parser.add_argument('--fl_algorithm', type=str, default="FedAvg",
                        help='Algorithm list: FedAvg; FedOPT; FedProx; SCAFFOLD ')

    parser.add_argument('--backend', type=str, default="MPI",
                        help='Backend for Server and Client')

    parser.add_argument('--comm_round', type=int, default=10,
                        help='how many round of communications we shoud use')

    parser.add_argument('--is_mobile', type=int, default=1,
                        help='whether the program is running on the FedML-Mobile server side')

    parser.add_argument('--client_num_in_total', type=int, default=-1, metavar='NN',
                        help='number of clients in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int,
                        default=4, metavar='NN', help='number of workers')

    parser.add_argument('--epochs', type=int, default=3, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, metavar='EP',
                        help='how many steps for accumulate the loss.')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='Optimizer used on the client. This field can be the name of any subclass of the torch Opimizer class.')

    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate on the client (default: 0.001)')

    parser.add_argument('--client_momentum', type=float, default=0,
                        help='client momentum (default: 0)')

    parser.add_argument('--weight_decay', type=float, default=0, metavar='N',
                        help='L2 penalty')

    parser.add_argument('--server_optimizer', type=str, default='sgd',
                        help='Optimizer used on the server. This field can be the name of any subclass of the torch Opimizer class.')

    parser.add_argument('--server_lr', type=float, default=0.1,
                        help='server learning rate (default: 0.001)')

    parser.add_argument('--server_momentum', type=float, default=0,
                        help='server momentum (default: 0)')

    parser.add_argument('--fedprox_mu', type=float, default=1,
                        help='mu (default: 1)')

    parser.add_argument(
        '--evaluate_during_training_steps', type=int, default=100, metavar='EP',
        help='the frequency of the evaluation during training')

    parser.add_argument('--frequency_of_the_test', type=int, default=1,
                        help='the frequency of the algorithms')

    # GPU device management
    parser.add_argument('--gpu_mapping_file', type=str, default="gpu_mapping.yaml",
                        help='the gpu utilization file for servers and clients. If there is no \
                    gpu_util_file, gpu will not be used.')

    parser.add_argument('--gpu_mapping_key', type=str,
                        default="mapping_default",
                        help='the key in gpu utilization file')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')
    
    # cached related
    parser.add_argument('--reprocess_input_data',  action='store_true',
                        help='whether generate features')
    
    # freeze related
    parser.add_argument('--freeze_layers', type=str, default='', metavar='N',
                        help='freeze which layers')

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
    parser.add_argument('--use_reweight', type=int, default=0, metavar='N',
                        help='whether use re-weight')

    # whether use re-weight technique
    parser.add_argument('--lr_bound_factor', type=float, default=0.0, metavar='N',
                        help='decides the bound of adjusted lr')

    # client sampling strategy
    parser.add_argument('--client_sampling_strategy', type=str, default='uniform', metavar='N',
                        help='client sampling strategy: uniform  / MD / AdaFL')
    return parser
