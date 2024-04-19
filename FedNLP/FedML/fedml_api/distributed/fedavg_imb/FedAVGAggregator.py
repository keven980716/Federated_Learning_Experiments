import copy
import logging
import random
import time

import numpy as np
import torch
import wandb
import math
import os
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from .utils import transform_list_to_tensor


class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]


class FedAVGAggregator(object):

    def __init__(self, train_global, test_global, all_train_data_num,
                 train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_num, device,
                 args, model_trainer, auxiliary_data_server=None):
        self.trainer = model_trainer

        self.args = args
        self.train_global = train_global
        self.test_global = test_global
        self.val_global = self._generate_validation_set()
        self.all_train_data_num = all_train_data_num

        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict

        self.worker_num = worker_num
        self.device = device
        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        ###
        self.current_client_index_dict = dict()
        self.update_step = 0
        self.centralized_params_list = []
        self.individual_grad_norm_sum = 0.0
        self.distance_norm_sum = 0.0
        self.centralized_avg_params = None
        self.momentum = None
        self.momentum_beta = 0.9
        self.global_init_lr = 1.0
        self.init_var_term = 1.0
        self.exp_var_term = 0.0
        self.classifier_exp_var_term = 0.0
        self.encoder_exp_var_term = 0.0
        self.exp_var_term_beta = 0.9
        self.exp_history_lr = 0.0
        self.init_lr_approx_clients_list = []
        self.exp_var_term_dict = dict()
        self.lr_bound_factor = self.args.lr_bound_factor
        self.client_sampling_strategy = self.args.client_sampling_strategy
        if self.args.client_sampling_strategy == 'AdaFL':
            self.sampling_weights = [1 / self.args.client_num_in_total for i in range(self.args.client_num_in_total)]
            self.sampling_weights_alpha = 0.9
            self.sampling_weights_distance = dict()
        self.local_training_loss_dict = dict()

        self.auxiliary_data_server = auxiliary_data_server
        self.alpha_list = None  # initialized to be None
        self.ratio_coef_alpha = 1.0
        self.ratio_coef_beta = 0.1
        self.prototypes = None  # initialized to be None
        # for BA
        self.balanced_gradients = None  # initialized to be None, supposed to be a dict {class_id: {param_name: grad}}
        self.balanced_gradients_dict = {}  # for each sampled client {client_idx: balanced_gradients}}
        self.ba_params = {}  # {client_idx: {param_name: param_data}}
        for idx in range(self.args.client_num_in_total):
            self.ba_params[idx] = None
        # for CLIMB
        self.climb_lambda = {}  # {client_idx: value}
        self.local_training_loss_per_client = {}
        for idx in range(self.args.client_num_in_total):
            self.climb_lambda[idx] = 0.0
            self.local_training_loss_per_client[idx] = 0.0
        # for CReFF
        self.creff_retrained_classifier = None
        self.federated_features = None
        self.federated_labels = None
        self.federated_features_opt = None
        ###

    def get_global_model_params(self):
        return self.trainer.get_model_params()

    def set_global_model_params(self, model_parameters):
        self.trainer.set_model_params(model_parameters)

    def add_local_trained_result(self, index, model_params, sample_num, client_index, local_training_loss,
                                 balanced_gradients):
        logging.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        ###
        # str -> int
        self.current_client_index_dict[index] = int(client_index)
        self.local_training_loss_dict[index] = local_training_loss
        ###
        self.balanced_gradients_dict[index] = balanced_gradients  # a dict, key is each label/ or None
        ###
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def aggregate(self):
        start_time = time.time()
        model_list = []
        training_num = 0

        for idx in range(self.worker_num):
            if self.args.is_mobile == 1:
                self.model_dict[idx] = transform_list_to_tensor(self.model_dict[idx])
            #model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
            ###
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx], self.current_client_index_dict[idx],
                               self.local_training_loss_dict[idx], self.balanced_gradients_dict[idx]))
            ###
            training_num += self.sample_num_dict[idx]

        logging.info("len of self.model_dict[idx] = " + str(len(self.model_dict)))
        

        # logging.info("################aggregate: %d" % len(model_list))
        (num0, averaged_params, _, _, _) = model_list[0]
        # (num0, _, _) = model_list[0]
        local_training_loss_sum = 0
        original_param_0 = model_list[0][1].copy()
        # averaged_params = model_list[0][1].copy()
        # aggregate bias attractor
        if self.args.decoupled_training and not self.args.aggregate_ba:
            for i in range(0, len(model_list)):
                _, local_model_params, client_index, _, _ = model_list[i]
                for k in local_model_params.keys():
                    if 'bias_attractor' in k:
                        if self.ba_params[client_index] is None:
                            self.ba_params[client_index] = {}
                        self.ba_params[client_index][k] = copy.deepcopy(local_model_params[k].data.cpu())

        # calculate weights for CLIMB
        if self.args.use_climb:
            avg_climb_lambda = np.mean(list(self.climb_lambda.values()))

        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params, client_index, local_training_loss, _ = model_list[i]
                w = local_sample_number / training_num
                local_training_loss_sum += local_training_loss * w
                if self.args.use_climb:
                    w = w * (1 + self.climb_lambda[client_index] - avg_climb_lambda)
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

        # aggregate local classifier gradients for global balanced gradients, and update climb_lambda
        bal_grad_class_client_num = {}  # {class_id: client_num}
        new_balanced_gradients = {}  # {class_id: {param_name: grad}}
        for i in range(0, len(model_list)):
            local_sample_number, _, client_index, local_training_loss, balanced_gradients_per_class = model_list[i]
            # w = local_sample_number / training_num
            # local_training_loss_sum += local_training_loss * w
            if self.args.use_climb:
                self.climb_lambda[client_index] = max(0.0, self.climb_lambda[client_index] + self.args.climb_dual_lr *
                                                      (local_training_loss - local_training_loss_sum - self.args.climb_eps))
            if (self.args.decoupled_training or self.args.use_creff) and balanced_gradients_per_class is not None:
                if self.balanced_gradients is None:
                    self.balanced_gradients = {}  # {class_id: {param_name: grad}}
                for class_id in balanced_gradients_per_class:
                    if class_id not in new_balanced_gradients:
                        new_balanced_gradients[class_id] = {}  # {param_name: grad}
                        for param_name in balanced_gradients_per_class[class_id]:
                            new_balanced_gradients[class_id][param_name] = balanced_gradients_per_class[class_id][param_name]
                    else:
                        for param_name in balanced_gradients_per_class[class_id]:
                            new_balanced_gradients[class_id][param_name] += balanced_gradients_per_class[class_id][param_name]
                    if class_id in bal_grad_class_client_num:
                        bal_grad_class_client_num[class_id] += 1
                    else:
                        bal_grad_class_client_num[class_id] = 1
        if new_balanced_gradients:
            for class_id in new_balanced_gradients:
                for param_name in new_balanced_gradients[class_id]:
                    new_balanced_gradients[class_id][param_name] /= bal_grad_class_client_num[class_id]
                self.balanced_gradients[class_id] = copy.deepcopy(new_balanced_gradients[class_id])
            # print(self.balanced_gradients)
        wandb.log({"Local Training Loss": local_training_loss_sum})

        # group distances/grad norms
        distance_dict = dict()
        grad_norm_dict = dict()
        adjusted_lr_dict = dict()
        for k in averaged_params.keys():
            if 'weight' in k or 'bias' in k:
                distance_dict[k] = []
                grad_norm_dict[k] = []
                adjusted_lr_dict[k] = 0

        # calculate 2-norm distance between each gradient and average gradient
        distance_list = []
        # classifier_distance_list, encoder_distance_list = [], []
        for i in range(0, len(model_list)):
            dist = 0
            # classifier_distance, encoder_distance = 0, 0
            for k in averaged_params.keys():
                if 'weight' in k or 'bias' in k:
                    if i == 0:
                        dist += torch.norm(averaged_params[k] - original_param_0[k]) ** 2
                        # if 'classifier' in k or 'fc' in k:
                        #     classifier_distance += torch.norm(averaged_params[k] - original_param_0[k]) ** 2
                        # else:
                        #     encoder_distance += torch.norm(averaged_params[k] - original_param_0[k]) ** 2
                        distance_dict[k].append((torch.norm(averaged_params[k] - original_param_0[k]) ** 2).item())
                    else:
                        dist += torch.norm(averaged_params[k] - model_list[i][1][k]) ** 2
                        # if 'classifier' in k or 'fc' in k:
                        #     classifier_distance += torch.norm(averaged_params[k] - model_list[i][1][k]) ** 2
                        # else:
                        #     encoder_distance += torch.norm(averaged_params[k] - model_list[i][1][k]) ** 2
                        distance_dict[k].append((torch.norm(averaged_params[k] - model_list[i][1][k]) ** 2).item())
                else:
                    # print(k)
                    pass
            distance_list.append(dist.item())
            if self.args.client_sampling_strategy == 'AdaFL':
                self.sampling_weights_distance[i] = math.sqrt(dist.item())
            # classifier_distance_list.append(classifier_distance.item())
            # encoder_distance_list.append(encoder_distance.item())

        # calculate each gradient's 2-norm
        grad_norm_list = []
        # classifier_grad_norm_list, encoder_grad_norm_list = [], []
        for i in range(0, len(model_list)):
            grad_norm = 0
            # classifier_grad_norm, encoder_grad_norm = 0, 0
            for k in averaged_params.keys():
                if 'weight' in k or 'bias' in k:
                    if i == 0:
                        grad_norm += torch.norm(original_param_0[k] - self.trainer.model.state_dict()[k].cpu()) ** 2
                        # if 'classifier' in k or 'fc' in k:
                        #     classifier_grad_norm += torch.norm(original_param_0[k] - self.trainer.model.state_dict()[k].cpu()) ** 2
                        # else:
                        #     encoder_grad_norm += torch.norm(original_param_0[k] - self.trainer.model.state_dict()[k].cpu()) ** 2
                        grad_norm_dict[k].append((torch.norm(original_param_0[k] - self.trainer.model.state_dict()[k].cpu()) ** 2).item())
                    else:
                        grad_norm += torch.norm(model_list[i][1][k] - self.trainer.model.state_dict()[k].cpu()) ** 2
                        # if 'classifier' in k or 'fc' in k:
                        #     classifier_grad_norm += torch.norm(model_list[i][1][k] - self.trainer.model.state_dict()[k].cpu()) ** 2
                        # else:
                        #     encoder_grad_norm += torch.norm(model_list[i][1][k] - self.trainer.model.state_dict()[k].cpu()) ** 2
                        grad_norm_dict[k].append((torch.norm(model_list[i][1][k] - self.trainer.model.state_dict()[k].cpu()) ** 2).item())
            grad_norm_list.append(grad_norm.item())
            # classifier_grad_norm_list.append(classifier_grad_norm.item())
            # encoder_grad_norm_list.append(encoder_grad_norm.item())
        
        # calculate var term in the denominator
        var_dict = dict()
        for k in distance_dict.keys():
            var_dict[k] = pow(1 / (1 - np.sum(distance_dict[k]) / (np.sum(grad_norm_dict[k]) + 1e-12)), 1/2)
            #wandb.log({k: var_dict[k]})

        var_term = pow(1 / (1 - np.sum(distance_list) / np.sum(grad_norm_list)), 1/2)
        classifier_var_term = 1 #pow(1 / (1 - np.sum(classifier_distance_list) / np.sum(classifier_grad_norm_list)), 1/2)
        encoder_var_term = 1 #pow(1 / (1 - np.sum(encoder_distance_list) / np.sum(encoder_grad_norm_list)), 1/2)

        if self.update_step == 0:  # self.args.var_adjust_begin_round:
            self.exp_var_term = var_term
            self.classifier_exp_var_term = classifier_var_term
            self.encoder_exp_var_term = encoder_var_term
            for k in var_dict.keys():
                self.exp_var_term_dict[k] = var_dict[k]

        # calculate approximate numerator dot for optimizing the end of training
        # approx_numerator_dot = local_loss_sum
        # wandb.log({'Approx Numerator dot': approx_numerator_dot})
        # adjust global lr
        if self.args.init_lr_approx_clients == 0:
            # whether scale server lr due to larger global batch size
            if not self.args.scale_server_lr:
                adjusted_lr = self.args.server_lr
            else:
                adjusted_lr = self.args.server_lr * self.args.client_num_per_round
            adjusted_classifier_lr = adjusted_lr
            adjusted_encoder_lr = adjusted_lr
            for k in adjusted_lr_dict.keys():
                adjusted_lr_dict[k] = adjusted_lr
            # clip large LR
            up_bound = (1 + self.lr_bound_factor * self.update_step) * adjusted_lr
            low_bound = (1 - self.lr_bound_factor * self.update_step) * adjusted_lr
            # whether use var_based lr adjustment
            if self.args.use_var_adjust:
                if self.update_step > (self.args.var_adjust_begin_round - 1):
                    if self.args.only_adjusted_layer == 'classifier':
                        adjusted_classifier_lr *= (classifier_var_term / self.classifier_exp_var_term)
                    elif self.args.only_adjusted_layer == 'encoder':
                        adjusted_encoder_lr *= (encoder_var_term / self.encoder_exp_var_term)
                    elif self.args.only_adjusted_layer == 'separate':
                        adjusted_classifier_lr *= (classifier_var_term / self.classifier_exp_var_term)
                        adjusted_encoder_lr *= (encoder_var_term / self.encoder_exp_var_term)
                    elif self.args.only_adjusted_layer == 'group':
                        for k in adjusted_lr_dict.keys():
                            adjusted_lr_dict[k] *= (var_dict[k] / self.exp_var_term_dict[k])
                    else:
                        adjusted_lr *= (var_term / self.exp_var_term)
            adjusted_lr = max(min(adjusted_lr, up_bound), low_bound)
            adjusted_classifier_lr = max(min(adjusted_classifier_lr, up_bound), low_bound)
            adjusted_encoder_lr = max(min(adjusted_encoder_lr, up_bound), low_bound)
            #
            for k in adjusted_lr_dict.keys():
                adjusted_lr_dict[k] = max(min(adjusted_lr_dict[k], up_bound), low_bound)

        else:
            pass
        # whether warm-up lr
        if self.args.warmup_steps != 0 and self.update_step < self.args.warmup_steps:
            adjusted_lr = 1.0 + self.update_step * (adjusted_lr - 1.0) / self.args.warmup_steps
            adjusted_classifier_lr = 1.0 + self.update_step * (adjusted_classifier_lr - 1.0) / self.args.warmup_steps
            adjusted_encoder_lr = 1.0 + self.update_step * (adjusted_encoder_lr - 1.0) / self.args.warmup_steps

        for k in averaged_params.keys():
            if 'weight' in k or 'bias' in k:
                if not self.args.use_var_adjust:
                    current_lr = adjusted_lr
                else:
                    if self.args.only_adjusted_layer == 'group':
                        current_lr = adjusted_lr_dict[k]
                    elif self.args.only_adjusted_layer == 'classifier':
                        if 'classifier' in k or 'fc' in k:
                            current_lr = adjusted_classifier_lr
                        else:
                            current_lr = adjusted_lr
                    elif self.args.only_adjusted_layer == 'encoder':
                        if 'classifier' in k or 'fc' in k:
                            current_lr = adjusted_lr
                        else:
                            current_lr = adjusted_encoder_lr
                    elif self.args.only_adjusted_layer == 'separate':
                        if 'classifier' in k or 'fc' in k:
                            current_lr = adjusted_classifier_lr
                        else:
                            current_lr = adjusted_encoder_lr
                    else:
                        current_lr = adjusted_lr
                averaged_params[k] = self.trainer.model.state_dict()[k].cpu() + current_lr * \
                                     (averaged_params[k] - self.trainer.model.state_dict()[k].cpu())

        wandb.log({"Exact Server LR": adjusted_lr})
        wandb.log({"Exact Server Classifier LR": adjusted_classifier_lr})
        wandb.log({"Exact Server Encoder LR": adjusted_encoder_lr})
        #print("Exact Server LR: ", adjusted_lr)
        wandb.log({"Var Term": var_term})
        # wandb.log({"Exp Var Term": self.exp_var_term})
        #print("Var Term: ", var_term)
        """
        for k in averaged_params.keys():
            if 'weight' in k or 'bias' in k:
                averaged_params[k] = self.trainer.model.state_dict()[k].cpu() + self.args.server_lr * \
                                     (averaged_params[k] - self.trainer.model.state_dict()[k].cpu())
        """
        if self.auxiliary_data_server is not None and self.args.local_loss == 'Ratio':
            global_params_last = copy.deepcopy(self.trainer.model.state_dict())
            cc_net = []
            for id in self.auxiliary_data_server.keys():
                cc_w = self.server_update(model=copy.deepcopy(self.trainer.model).to(self.device),
                                          train_data=self.auxiliary_data_server[id], device=self.device)
                cc_net.append(copy.deepcopy(cc_w))
            pos, weight_name = self.outlier_detect(global_params_last, cc_net)

            self.alpha_list = self.whole_determination(pos, global_params_last, cc_net, weight_name)
            print("Reset alpha list to be: ", self.alpha_list)

        # update the global model which is cached at the server side
        self.set_global_model_params(averaged_params)

        # for decoupled local training
        if self.auxiliary_data_server is not None and (self.args.decoupled_training or self.args.replace_classifier_prototypes):
            # self.get_class_prototype()
            # this is only for oracle case
            pass
            # self.get_balanced_gradients()

        # for cRT
        if self.auxiliary_data_server is not None and self.args.retrain_server_classifier:
            averaged_params = self.retrain_classifier()

        # for replace the classifier with the prototypes
        if self.auxiliary_data_server is not None and self.args.replace_classifier_prototypes:
            for k in averaged_params.keys():
                if 'weight' in k and ('classifier' in k or 'fc' in k or 'linear_2' in k):
                    for label in range(len(averaged_params[k].data)):
                        averaged_params[k].data[label] = averaged_params[k].data[label].norm().item() * self.prototypes[label]  # / self.prototypes[label].norm().item()
            self.set_global_model_params(averaged_params)

        # for CReFF
        if self.args.use_creff:
            for k in reversed(averaged_params.keys()):
                if 'weight' in k and ('classifier' in k or 'fc' in k or 'linear_2' in k):
                    num_classes, hidden_size = averaged_params[k].shape
                    break
            if self.creff_retrained_classifier is None:
                self.creff_retrained_classifier = nn.Linear(hidden_size, num_classes, bias=True)
                self.federated_features = torch.randn(size=(num_classes * self.args.number_of_federated_features_per_class,
                                                            hidden_size), dtype=torch.float,
                                                      requires_grad=True, device=self.device)
                self.federated_labels = torch.tensor([np.ones(self.args.number_of_federated_features_per_class) * i for i in range(num_classes)],
                                                     dtype=torch.long, requires_grad=False, device=self.device).view(-1)  # [0,0,0, 1,1,1, ..., 9,9,9]

                self.federated_features_opt = torch.optim.SGD([self.federated_features, ],
                                                              lr=self.args.federated_features_optimization_lr)
            # updating federated features
            self.update_federated_features(averaged_params, self.device, num_classes, hidden_size)
            # re-train classifier
            self.creff_retrain_classifier(num_classes, hidden_size)
        """
        if self.update_step > 0:
            for k in self.momentum.keys():
                self.momentum[k] = self.momentum_beta * self.momentum[k] + (1) * (averaged_params[k] - self.trainer.model.state_dict()[k].cpu())
        """
        if self.update_step > 0:  # self.args.var_adjust_begin_round:
            self.exp_var_term = self.exp_var_term_beta * self.exp_var_term + (1 - self.exp_var_term_beta) * var_term
            self.classifier_exp_var_term = self.exp_var_term_beta * self.classifier_exp_var_term + (1 - self.exp_var_term_beta) * classifier_var_term
            self.encoder_exp_var_term = self.exp_var_term_beta * self.encoder_exp_var_term + (1 - self.exp_var_term_beta) * encoder_var_term
            for k in self.exp_var_term_dict.keys():
                self.exp_var_term_dict[k] = self.exp_var_term_beta * self.exp_var_term_dict[k] + (1 - self.exp_var_term_beta) * var_dict[k]

        # update sampling weights if AdaFL
        if self.args.client_sampling_strategy == 'AdaFL':
            total_dist, total_original_weights = 0, 0
            for i in range(len(self.current_client_index_dict)):
                current_client = self.current_client_index_dict[i]
                total_dist += self.sampling_weights_distance[i]
                total_original_weights += self.sampling_weights[current_client]
            for i in range(len(self.current_client_index_dict)):
                current_client = self.current_client_index_dict[i]
                self.sampling_weights[current_client] = self.sampling_weights_alpha * self.sampling_weights[current_client] + (1 - self.sampling_weights_alpha)\
                                                        * total_original_weights * self.sampling_weights_distance[i] / total_dist
            # after above procedure, the sum of sampling weights is not equal to 1 strictly
            # only keep 8 digits
            for i in range(len(self.sampling_weights)):
                self.sampling_weights[i] = round(self.sampling_weights[i], 8)
            self.sampling_weights[0] += round(1.0 - round(np.sum(self.sampling_weights).item(), 8), 8)
            print(self.sampling_weights)
            # print(np.sum(self.sampling_weights))
            # assert np.sum(self.sampling_weights) == 1.0

        self.update_step += 1

        # whether save the aggregated global model
        if self.args.save_global_model_during_training_rounds > 0 and self.update_step % self.args.save_global_model_during_training_rounds == 0:
            # self.trainer.model.save_pretrained(os.path.join(self.args.output_dir, 'round{}/'.format(str(self.update_step))))  # for nlp model
            save_path = os.path.join(self.args.save_global_model_directory, "round{}.pth".format(str(self.update_step)))
            os.makedirs(self.args.save_global_model_directory, exist_ok=True)
            torch.save(self.trainer.model.state_dict(), save_path)

        # tau-normalization
        # averaged_params = self.tau_normalization(averaged_params)

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return averaged_params

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            if self.client_sampling_strategy == 'uniform':
                client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
            elif self.client_sampling_strategy == 'MD':
                client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=True)
            elif self.client_sampling_strategy == 'AdaFL':
                client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False, p=self.sampling_weights)
            else:
                print("Client sampling strategy is not defined.")
                assert 0 == 1
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    # for ratio loss
    def server_update(self, model, train_data, device):
        model.to(device)
        model.train()

        # train and update
        tmp_criterion = nn.CrossEntropyLoss().to(device)
        # todo: add adam
        tmp_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.lr,
                                        momentum=self.args.client_momentum)
        for batch_idx, (x, labels) in enumerate(train_data):
            x, labels = x.to(device), labels.to(device)
            model.zero_grad()
            log_probs = model(x)
            loss = tmp_criterion(log_probs, labels)
            loss.backward()
            tmp_optimizer.step()
        return model.state_dict()

    def outlier_detect(self, w_global, w_local):
        weight_name = None
        for k in w_global.keys():
            if ('linear' in k and 'weight' in k) or ('fc' in k and 'weight' in k):
                weight_name = k
        if not weight_name:
            assert 0 == 1
        print(weight_name)
        w_global = w_global[weight_name].cpu().numpy()
        w = []
        for i in range(len(w_local)):
            temp = (w_local[i][weight_name].cpu().numpy() - w_global) * self.args.client_num_in_total
            w.append(temp)
        res = self.search_neuron_new(w)
        return res, weight_name

    def search_neuron_new(self, w):
        w = np.array(w)
        class_num, dim = w.shape[1], w.shape[2]
        pos_res = np.zeros((len(w), class_num, dim))
        for i in range(w.shape[1]):
            for j in range(w.shape[2]):
                temp = []
                for p in range(len(w)):
                    temp.append(w[p, i, j])
                max_index = temp.index(max(temp))
                # pos_res[max_index, i, j] = 1

                if w[max_index, i, j] == 0:
                    outlier = np.where(temp == w[max_index, i, j])
                else:
                    outlier = np.where(np.abs(temp) / abs(w[max_index, i, j]) > 0.80)
                if len(outlier[0]) < 2:
                    pos_res[max_index, i, j] = 1
                # pos_res[max_index, i, j] = 1
        return pos_res

    def whole_determination(self, pos, w_glob_last, cc_net, weight_name):
        ratio_res = []
        class_num = pos.shape[1]
        for it in range(class_num):
            cc_class = it
            aux_sum = 0
            aux_other_sum = 0
            for i in range(pos.shape[1]):
                for j in range(pos.shape[2]):
                    if pos[cc_class, i, j] == 1:
                        temp = []
                        last = w_glob_last[weight_name].cpu().numpy()[i, j]
                        cc = cc_net[cc_class][weight_name].cpu().numpy()[i, j]
                        for p in range(len(cc_net)):
                            temp.append(cc_net[p][weight_name].cpu().numpy()[i, j] - last)
                        temp = np.array(temp)
                        temp = np.delete(temp, cc_class)
                        temp_ave = np.sum(temp) / (len(cc_net) - 1)
                        aux_sum += cc - last
                        aux_other_sum += temp_ave
            if aux_other_sum != 0:
                res = abs(aux_sum) / abs(aux_other_sum)
            else:
                res = 10
            print('label {}-----aux_data:{}, aux_other:{}, ratio:{}'.format(it, aux_sum, aux_other_sum, res))
            ratio_res.append(res)

        # normalize the radio alpha
        ratio_min = np.min(ratio_res)
        ratio_max = np.max(ratio_res)
        for i in range(len(ratio_res)):
            # add a upper bound to the ratio
            if ratio_res[i] >= 5000:
                ratio_res[i] = 5000
            ratio_res[i] = 1.0 + 0.1 * ratio_res[i]
            # ratio_res[i] = 1.5 - 0.3  * (ratio_res[i] - ratio_min) / (ratio_max - ratio_min)
        return ratio_res

    # for decoupled training
    def get_class_prototype(self):
        classification_weight = None
        #for k in self.trainer.model.state_dict().keys():
        #    if 'weight' in k and ('linear' in k or 'fc' in k or 'classifier' in k):
        #        classification_weight = self.trainer.model.state_dict()[k].cpu()

        model = copy.deepcopy(self.trainer.model).to(self.device)
        model.to(self.device)
        model.eval()
        if self.prototypes is None:
            self.prototypes = {}

        for id in self.auxiliary_data_server.keys():
            """
            if classification_weight is not None:
                cur_prototype = classification_weight[id]
                self.prototypes[id] = cur_prototype
                continue
            """
            cur_prototype = None
            sample_num = 0
            train_data = self.auxiliary_data_server[id]
            with torch.no_grad():
                for batch_idx, (x, labels) in enumerate(train_data):
                    x, labels = x.to(self.device), labels.to(self.device)
                    sample_num += len(labels)
                    _, final_states = model(x)
                    if cur_prototype is None:
                        cur_prototype = torch.sum(final_states, 0).to('cpu')
                    else:
                        cur_prototype += torch.sum(final_states, 0).to('cpu')
                cur_prototype /= sample_num
                cur_prototype /= cur_prototype.norm().item()
                self.prototypes[id] = cur_prototype
        print(self.prototypes)

    # for cRT
    def retrain_classifier(self):
        model = copy.deepcopy(self.trainer.model)
        for k, v in model.named_parameters():
            #print(k, v.requires_grad)
            if not ('classifier' in k or 'fc' in k or 'linear_2' in k):
                v.requires_grad = False
            #    print(k, v.requires_grad)
        model.to(self.device)
        model.train()
        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=self.args.retrain_classifier_lr,
                                    momentum=0.9)
        print("begin retraining classifier...")
        train_data = self.auxiliary_data_server
        for epoch in range(self.args.retrain_classifier_epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(self.device), labels.to(self.device)
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()
                # to avoid nan loss
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                batch_loss.append(loss.item())
            logging.info('cRT Epoch: {}\tLoss: {:.6f}'.format(epoch, sum(batch_loss) / len(batch_loss)))
        averaged_params = model.cpu().state_dict()
        self.trainer.set_model_params(averaged_params)
        return averaged_params

    # for tau_normalization
    def tau_normalization(self, averaged_params):
        for k in averaged_params.keys():
            if 'weight' in k and ('classifier' in k or 'fc' in k or 'linear_2' in k):
                print(self.trainer.model.state_dict()[k], averaged_params[k])
                per_norm = math.sqrt((averaged_params[k].norm().item() ** 2) / len(averaged_params[k]))
                for i in range(len(averaged_params[k])):
                    averaged_params[k][i] = averaged_params[k][i] * per_norm / averaged_params[k][i].norm().item()
                self.set_global_model_params(averaged_params)
                print(self.trainer.model.state_dict()[k], averaged_params[k])
        return averaged_params

    # for balanced gradients
    def get_balanced_gradients(self):
        model = copy.deepcopy(self.trainer.model).to(self.device)
        model.to(self.device)
        model.eval()
        criterion = nn.CrossEntropyLoss().to(self.device)
        if self.balanced_gradients is None:
            self.balanced_gradients = {}  # {class_id: {param_name: grad}}

        for id in self.auxiliary_data_server.keys():
            cur_grad = None  # {param_name: grad}
            sample_num = 0
            train_data = self.auxiliary_data_server[id]
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(self.device), labels.to(self.device)
                model.zero_grad()
                sample_num += 1
                log_probs, final_states = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()

                if cur_grad is None:
                    cur_grad = {}  # {param_name: grad}
                    for k, v in model.named_parameters():
                        if 'classifier' in k or 'fc' in k or 'linear_2' in k:
                            cur_grad[k] = copy.deepcopy(v.grad.data.cpu())
                else:
                    for k, v in model.named_parameters():
                        if 'classifier' in k or 'fc' in k or 'linear_2' in k:
                            cur_grad[k] += v.grad.data.cpu()
            for param_name in cur_grad:
                cur_grad[param_name] /= sample_num
            self.balanced_gradients[id] = copy.deepcopy(cur_grad)
        # print(self.balanced_gradients)

    # CReFF: update federated features
    def update_federated_features(self, global_params, device, num_classes, hidden_size):
        print("Begin updating federated features for CReFF......")
        creff_criterion = nn.CrossEntropyLoss().to(device)
        # feature_net_params = self.creff_retrained_classifier.state_dict()
        # for name_param in reversed(global_params):
        #     if 'bias' in name_param and ('classifier' in name_param or 'fc' in name_param or 'linear_2' in name_param):
        #         feature_net_params['bias'].data = copy.deepcopy(global_params[name_param].data.to(device))
        #     if 'weight' in name_param and ('classifier' in name_param or 'fc' in name_param or 'linear_2' in name_param):
        #         feature_net_params['weight'].data = copy.deepcopy(global_params[name_param].data.to(device))
        #         break
        # self.creff_retrained_classifier.load_state_dict(feature_net_params)
        self.creff_retrained_classifier.to(device)
        self.creff_retrained_classifier.train()
        net_global_parameters = list(self.creff_retrained_classifier.parameters())
        """
        gw_real_all = {class_index: [] for class_index in range(num_classes)}
        for gradient_one in list_clients_gradient:
            for class_num, gradient in gradient_one.items():
                gw_real_all[class_num].append(gradient)
        """
        gw_real_avg = {class_index: [] for class_index in range(num_classes)}
        # aggregate the real feature gradients
        for i in range(num_classes):
            """
            gw_real_temp = []
            list_one_class_client_gradient = gw_real_all[i]

            if len(list_one_class_client_gradient) != 0:
                weight_temp = 1.0 / len(list_one_class_client_gradient)
                for name_param in range(2):
                    list_values_param = []
                    for one_gradient in list_one_class_client_gradient:
                        list_values_param.append(one_gradient[name_param] * weight_temp)
                    value_global_param = sum(list_values_param)
                    gw_real_temp.append(value_global_param)
                gw_real_avg[i] = gw_real_temp
            """
            if i in self.balanced_gradients:
                gw_real_temp = []
                for name_param in self.balanced_gradients[i]:
                    gw_real_temp.append(copy.deepcopy(self.balanced_gradients[i][name_param]).to(device))
                gw_real_avg[i] = copy.deepcopy(gw_real_temp)
        # update the federated features.
        for ep in range(self.args.federated_features_optimization_steps):
            loss_feature = torch.tensor(0.0).to(device)
            for c in range(num_classes):
                if len(gw_real_avg[c]) != 0:
                    feature_syn = self.federated_features[c * self.args.number_of_federated_features_per_class: (c + 1) *
                                                          self.args.number_of_federated_features_per_class].reshape(
                                                          (self.args.number_of_federated_features_per_class, hidden_size))
                    lab_syn = torch.ones((self.args.number_of_federated_features_per_class,), device=device, dtype=torch.long) * c
                    output_syn = self.creff_retrained_classifier(feature_syn)
                    loss_syn = creff_criterion(output_syn, lab_syn)
                    # compute the federated feature gradients of class c
                    gw_syn = torch.autograd.grad(loss_syn, net_global_parameters, create_graph=True)
                    loss_feature += self.match_loss(gw_syn, gw_real_avg[c])
            self.federated_features_opt.zero_grad()
            loss_feature.backward()
            self.federated_features_opt.step()
            print("Updating federated features, Epoch={}\t Loss={}".format(ep, loss_feature.item()))

    # CReFF: re-train new classifier
    def creff_retrain_classifier(self, num_classes, hidden_size, batch_size_local_training=32):
        print("Begin retraining classifier for CReFF......")
        creff_criterion = nn.CrossEntropyLoss().to(self.device)
        feature_syn_train_ft = copy.deepcopy(self.federated_features.detach())
        label_syn_train_ft = copy.deepcopy(self.federated_labels.detach())
        dst_train_syn_ft = TensorDataset(feature_syn_train_ft, label_syn_train_ft)
        ft_model = nn.Linear(hidden_size, num_classes, bias=True).to(self.device)
        optimizer_ft_net = torch.optim.SGD(ft_model.parameters(),
                                           lr=self.args.creff_classifier_retraining_lr)
        ft_model.train()
        for epoch in range(self.args.creff_classifier_retraining_epochs):
            trainloader_ft = DataLoader(dataset=dst_train_syn_ft,
                                        batch_size=batch_size_local_training,
                                        shuffle=True)
            for data_batch in trainloader_ft:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = ft_model(images)
                loss_net = creff_criterion(outputs, labels)
                optimizer_ft_net.zero_grad()
                loss_net.backward()
                optimizer_ft_net.step()
                print("Retraining CReFF classifier, Epoch={}\t Loss={}".format(epoch, loss_net.item()))
        ft_model.eval()
        self.creff_retrained_classifier = copy.deepcopy(ft_model.to('cpu'))

    # CReFF: match loss
    def match_loss(self, gw_syn, gw_real, dis_metric='cos'):
        dis = torch.tensor(0.0).to(self.device)

        if dis_metric == 'wb':
            for ig in range(len(gw_real)):
                gwr = gw_real[ig]
                gws = gw_syn[ig]
                dis += self.distance_wb(gwr, gws)

        elif dis_metric == 'mse':
            gw_real_vec = []
            gw_syn_vec = []
            for ig in range(len(gw_real)):
                gw_real_vec.append(gw_real[ig].reshape((-1)))
                gw_syn_vec.append(gw_syn[ig].reshape((-1)))
            gw_real_vec = torch.cat(gw_real_vec, dim=0)
            gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
            dis = torch.sum((gw_syn_vec - gw_real_vec) ** 2)

        elif dis_metric == 'cos':
            gw_real_vec = []
            gw_syn_vec = []
            for ig in range(len(gw_real)):
                gw_real_vec.append(gw_real[ig].reshape((-1)))
                gw_syn_vec.append(gw_syn[ig].reshape((-1)))
            gw_real_vec = torch.cat(gw_real_vec, dim=0)
            gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
            dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (
                        torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

        else:
            exit('DC error: unknown distance function')

        return dis

    def distance_wb(self, gwr, gws):
        shape = gwr.shape
        if len(shape) == 4:  # conv, out*in*h*w
            gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
            gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
        elif len(shape) == 3:  # layernorm, C*h*w
            gwr = gwr.reshape(shape[0], shape[1] * shape[2])
            gws = gws.reshape(shape[0], shape[1] * shape[2])
        elif len(shape) == 2:  # linear, out*in
            tmp = 'do nothing'
        elif len(shape) == 1:  # batchnorm/instancenorm, C; groupnorm x, bias
            gwr = gwr.reshape(1, shape[0])
            gws = gws.reshape(1, shape[0])
            # return 0

        dis_weight = torch.sum(
            1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
        dis = dis_weight
        return dis

    def _generate_validation_set(self, num_samples=10000):
        if self.args.dataset.startswith("stackoverflow"):
            test_data_num = len(self.test_global.dataset)
            sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
            subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
            sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
            return sample_testset
        else:
            return self.test_global

    def test_on_server_for_all_clients(self, round_idx):
        if self.trainer.test_on_the_server(self.train_data_local_dict, self.test_data_local_dict, self.device, self.args):
            return

        if round_idx % self.args.frequency_of_the_test == 0 or round_idx == self.args.comm_round - 1:
            logging.info("################test_on_server_for_all_clients : {}".format(round_idx))
            # train_num_samples = []
            # train_tot_corrects = []
            # train_losses = []
            # for client_idx in range(self.args.client_num_in_total):
            #     # train data
            #     if self.args.decoupled_training or self.args.use_creff:
            #         metrics, _ = self.trainer.test(self.train_data_local_dict[client_idx], self.device, self.args, self.creff_retrained_classifier)
            #     else:
            #         metrics, _ = self.trainer.test(self.train_data_local_dict[client_idx], self.device, self.args)
            #     train_tot_correct, train_num_sample, train_loss = metrics['test_correct'], metrics['test_total'], metrics['test_loss']
            #     train_tot_corrects.append(copy.deepcopy(train_tot_correct))
            #     train_num_samples.append(copy.deepcopy(train_num_sample))
            #     train_losses.append(copy.deepcopy(train_loss))

            #     """
            #     Note: CI environment is CPU-based computing. 
            #     The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            #     """
            #     if self.args.ci == 1:
            #         break

            # # test on training dataset
            # train_acc = sum(train_tot_corrects) / sum(train_num_samples)
            # train_loss = sum(train_losses) / sum(train_num_samples)
            # wandb.log({"Train/Acc": train_acc, "round": round_idx})
            # wandb.log({"Train/Loss": train_loss, "round": round_idx})
            # stats = {'training_acc': train_acc, 'training_loss': train_loss}
            # logging.info(stats)

            # test data
            test_num_samples = []
            test_tot_corrects = []
            test_losses = []

            if round_idx == self.args.comm_round - 1:
                if self.args.decoupled_training or self.args.use_creff:
                    metrics, results_per_class = self.trainer.test(self.test_global, self.device, self.args, self.creff_retrained_classifier)
                else:
                    metrics, results_per_class = self.trainer.test(self.test_global, self.device, self.args)
            else:
                if self.args.decoupled_training or self.args.use_creff:
                    metrics, results_per_class = self.trainer.test(self.val_global, self.device, self.args, self.creff_retrained_classifier)
                else:
                    metrics, results_per_class = self.trainer.test(self.val_global, self.device, self.args)
                
            test_tot_correct, test_num_sample, test_loss = metrics['test_correct'], metrics['test_total'], metrics[
                'test_loss']
            test_tot_corrects.append(copy.deepcopy(test_tot_correct))
            test_num_samples.append(copy.deepcopy(test_num_sample))
            test_losses.append(copy.deepcopy(test_loss))

            # test on test dataset
            test_acc = sum(test_tot_corrects) / sum(test_num_samples)
            test_loss = sum(test_losses) / sum(test_num_samples)
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            # statistics for tail classes (tail top-30%)
            test_tail_tot_correct, test_tail_num_samples = 0, 0
            acc_per_class = {}
            total_classes = len(results_per_class.keys())
            for k in results_per_class:
                acc_per_class[k] = results_per_class[k]['test_correct'] / results_per_class[k]['test_total']
                if k >= int(total_classes * 0.7):
                    test_tail_tot_correct += results_per_class[k]['test_correct']
                    test_tail_num_samples += results_per_class[k]['test_total']
            logging.info(stats)
            logging.info(acc_per_class)
            for k in results_per_class:
                wandb.log({"acc_" + str(k): acc_per_class[k]})
            wandb.log({"Test/Tail_Acc": test_tail_tot_correct / test_tail_num_samples, "round": round_idx})

