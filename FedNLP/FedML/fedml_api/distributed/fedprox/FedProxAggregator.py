import copy
import logging
import random
import time

import numpy as np
import torch
import wandb

from .utils import transform_list_to_tensor


class FedProxAggregator(object):

    def __init__(self, train_global, test_global, all_train_data_num,
                 train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_num, device,
                 args, model_trainer):
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
        self.update_step = 0
        self.centralized_params_list = []
        self.individual_grad_norm_sum = 0.0
        self.distance_norm_sum = 0.0
        self.centralized_avg_params = None
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
        ###

    def get_global_model_params(self):
        return self.trainer.get_model_params()

    def set_global_model_params(self, model_parameters):
        self.trainer.set_model_params(model_parameters)

    def add_local_trained_result(self, index, model_params, sample_num):
        logging.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
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
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
            training_num += self.sample_num_dict[idx]

        logging.info("len of self.model_dict[idx] = " + str(len(self.model_dict)))

        # logging.info("################aggregate: %d" % len(model_list))
        (num0, averaged_params) = model_list[0]
        original_param_0 = model_list[0][1].copy()

        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

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
                        distance_dict[k].append(
                            (torch.norm(averaged_params[k] - original_param_0[k]) ** 2).item())
                    else:
                        dist += torch.norm(averaged_params[k] - model_list[i][1][k]) ** 2
                        # if 'classifier' in k or 'fc' in k:
                        #     classifier_distance += torch.norm(averaged_params[k] - model_list[i][1][k]) ** 2
                        # else:
                        #     encoder_distance += torch.norm(averaged_params[k] - model_list[i][1][k]) ** 2
                        distance_dict[k].append(
                            (torch.norm(averaged_params[k] - model_list[i][1][k]) ** 2).item())
                else:
                    print(k)
            distance_list.append(dist.item())
            # classifier_distance_list.append(classifier_distance.item())
            # encoder_distance_list.append(encoder_distance.item())
        # print("grad 2-norm distance: ", distance_list)

        # calculate each gradient's 2-norm
        grad_norm_list = []
        # classifier_grad_norm_list, encoder_grad_norm_list = [], []
        for i in range(0, len(model_list)):
            grad_norm = 0
            # classifier_grad_norm, encoder_grad_norm = 0, 0
            for k in averaged_params.keys():
                if 'weight' in k or 'bias' in k:
                    if i == 0:
                        grad_norm += torch.norm(
                            original_param_0[k] - self.trainer.model.state_dict()[k].cpu()) ** 2
                        # if 'classifier' in k or 'fc' in k:
                        #     classifier_grad_norm += torch.norm(
                        #         original_param_0[k] - self.trainer.model.state_dict()[k].cpu()) ** 2
                        # else:
                        #     encoder_grad_norm += torch.norm(
                        #         original_param_0[k] - self.trainer.model.state_dict()[k].cpu()) ** 2
                        grad_norm_dict[k].append((torch.norm(
                            original_param_0[k] - self.trainer.model.state_dict()[k].cpu()) ** 2).item())
                    else:
                        grad_norm += torch.norm(
                            model_list[i][1][k] - self.trainer.model.state_dict()[k].cpu()) ** 2
                        # if 'classifier' in k or 'fc' in k:
                        #     classifier_grad_norm += torch.norm(
                        #         model_list[i][1][k] - self.trainer.model.state_dict()[k].cpu()) ** 2
                        # else:
                        #     encoder_grad_norm += torch.norm(
                        #         model_list[i][1][k] - self.trainer.model.state_dict()[k].cpu()) ** 2
                        grad_norm_dict[k].append((torch.norm(
                            model_list[i][1][k] - self.trainer.model.state_dict()[k].cpu()) ** 2).item())
            grad_norm_list.append(grad_norm.item())
            # classifier_grad_norm_list.append(classifier_grad_norm.item())
            # encoder_grad_norm_list.append(encoder_grad_norm.item())
        # print("grad norm list: ", grad_norm_list)
        # print("mean of square grads: ", np.mean(grad_norm_list))
        # wandb.log({"Mean Square Norm of Client's Grad.": np.mean(grad_norm_list)})
        # print("var of square grads: ", np.var(grad_norm_list))

        # calculate var term in the denominator
        var_dict = dict()
        for k in distance_dict.keys():
            var_dict[k] = pow(1 / (1 - np.sum(distance_dict[k]) / (np.sum(grad_norm_dict[k]) + 1e-12)), 1 / 2)
            # wandb.log({k: var_dict[k]})

        var_term = pow(1 / (1 - np.sum(distance_list) / np.sum(grad_norm_list)), 1 / 2)
        classifier_var_term = 1 #pow(
            # 1 / (1 - np.sum(classifier_distance_list) / np.sum(classifier_grad_norm_list)), 1 / 2)
        encoder_var_term = 1 # pow(1 / (1 - np.sum(encoder_distance_list) / np.sum(encoder_grad_norm_list)), 1 / 2)
        wandb.log({"Classifier Var Term": classifier_var_term})
        wandb.log({"Encoder Var Term": encoder_var_term})
        if self.update_step == 0:  # self.args.var_adjust_begin_round:
            self.exp_var_term = var_term
            self.classifier_exp_var_term = classifier_var_term
            self.encoder_exp_var_term = encoder_var_term
            for k in var_dict.keys():
                self.exp_var_term_dict[k] = var_dict[k]

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

        # whether warm-up lr
        if self.args.warmup_steps != 0 and self.update_step < self.args.warmup_steps:
            adjusted_lr = 1.0 + self.update_step * (adjusted_lr - 1.0) / self.args.warmup_steps
            adjusted_classifier_lr = 1.0 + self.update_step * (
                        adjusted_classifier_lr - 1.0) / self.args.warmup_steps
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
        # print("Exact Server LR: ", adjusted_lr)
        wandb.log({"Var Term": var_term})
        # wandb.log({"Exp Var Term": self.exp_var_term})
        # print("Var Term: ", var_term)

        # update the global model which is cached at the server side
        self.set_global_model_params(averaged_params)

        if self.update_step > 0:  # self.args.var_adjust_begin_round:
            self.exp_var_term = self.exp_var_term_beta * self.exp_var_term + (1 - self.exp_var_term_beta) * var_term
            self.classifier_exp_var_term = self.exp_var_term_beta * self.classifier_exp_var_term + (1 - self.exp_var_term_beta) * classifier_var_term
            self.encoder_exp_var_term = self.exp_var_term_beta * self.encoder_exp_var_term + (1 - self.exp_var_term_beta) * encoder_var_term
            for k in self.exp_var_term_dict.keys():
                self.exp_var_term_dict[k] = self.exp_var_term_beta * self.exp_var_term_dict[k] + (1 - self.exp_var_term_beta) * var_dict[k]
        self.update_step += 1
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
                pass
            #client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        if self.args.dataset.startswith("stackoverflow"):
            test_data_num  = len(self.test_global.dataset)
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
            train_num_samples = []
            train_tot_corrects = []
            train_losses = []
            for client_idx in range(self.args.client_num_in_total):
                # train data
                metrics = self.trainer.test(self.train_data_local_dict[client_idx], self.device, self.args)
                train_tot_correct, train_num_sample, train_loss = metrics['test_correct'], metrics['test_total'], metrics['test_loss']
                train_tot_corrects.append(copy.deepcopy(train_tot_correct))
                train_num_samples.append(copy.deepcopy(train_num_sample))
                train_losses.append(copy.deepcopy(train_loss))

                """
                Note: CI environment is CPU-based computing. 
                The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
                """
                if self.args.ci == 1:
                    break

            # test on training dataset
            train_acc = sum(train_tot_corrects) / sum(train_num_samples)
            train_loss = sum(train_losses) / sum(train_num_samples)
            wandb.log({"Train/Acc": train_acc, "round": round_idx})
            wandb.log({"Train/Loss": train_loss, "round": round_idx})
            stats = {'training_acc': train_acc, 'training_loss': train_loss}
            logging.info(stats)

            # test data
            test_num_samples = []
            test_tot_corrects = []
            test_losses = []

            if round_idx == self.args.comm_round - 1:
                metrics = self.trainer.test(self.test_global, self.device, self.args)
            else:
                metrics = self.trainer.test(self.val_global, self.device, self.args)
                
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
            logging.info(stats)