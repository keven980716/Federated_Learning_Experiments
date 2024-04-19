import logging
import copy
import torch
from torch import nn
from torch.nn import functional as F
import math
import numpy as np
import random

try:
    from fedml_core.trainer.model_trainer_imb import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer_imb import ModelTrainer


class focal_loss(nn.Module):
    def __init__(self, alpha=-1, gamma=2, num_classes=10, reduction='mean'):
        super(focal_loss, self).__init__()
        self.reduction = reduction

        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            self.alpha = torch.Tensor(alpha)
        elif alpha < 0:
            self.alpha = -1
        else:
            assert alpha < 1
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)

        # self.alpha = alpha

        self.gamma = gamma
        self.eps = 1e-6

    def forward(self, preds, labels):
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1, preds.size(-1))
        if self.alpha != -1:
            alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=1)
        preds_logsoft = torch.log(preds_softmax + self.eps)

        # focal_loss func, Loss = -α(1-yi)**γ *ce_loss(xi,yi)
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        if self.alpha != -1:
            alpha = alpha.gather(0, labels.view(-1))
        # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)
        # loss = - preds_logsoft
        if self.alpha != -1:
            loss = torch.mul(alpha, loss.t())
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class ratio_loss(nn.Module):
    def __init__(self, alpha=None, num_classes=10, reduction='mean'):
        super(ratio_loss, self).__init__()
        self.reduction = reduction

        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            self.alpha = torch.Tensor(alpha)
        else:
            self.alpha = torch.ones(num_classes)

        self.eps = 1e-6

    def forward(self, preds, labels):
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1, preds.size(-1))

        alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=1)
        preds_logsoft = torch.log(preds_softmax + self.eps)

        # preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = alpha.gather(0, labels.view(-1))
        #loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)
        # print("Ratio alpha: ", alpha)
        loss = torch.mul(alpha, - preds_logsoft)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


# supervised contrastive loss
class ct_loss(nn.Module):
    def __init__(self, num_classes=10, reduction='mean'):
        super(ct_loss, self).__init__()
        self.reduction = reduction
        self.eps = 1e-6
        self.tau = 1.0

    def forward(self, hidden_states, labels, prototype_matrix):
        # assert preds.dim()==2 and labels.dim()==1
        loss = 0.0
        for i in range(len(hidden_states)):
            hidden_state = hidden_states[i] / hidden_states[i].norm().item()
            exp_cos_sim = torch.exp(torch.squeeze(torch.matmul(torch.unsqueeze(hidden_state, 0), prototype_matrix.transpose(1, 0))) / self.tau)  # num_label
            # print(exp_cos_sim)
            label = labels[i].item()
            # print(label, exp_cos_sim[label], exp_cos_sim)
            per_loss = - torch.log(exp_cos_sim[label] / (torch.sum(exp_cos_sim)))
            if self.reduction == 'mean':
                loss = loss + per_loss / len(hidden_states)
            elif self.reduction == 'sum':
                loss = loss + per_loss
        return loss


class MyModelTrainer(ModelTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args, alpha=None, prototypes=None, balanced_gradients=None,
              ba_params=None, creff_retrained_params=None, class_num=10):
        # prototypes: {class_id: tensor}, balanced_gradients: {class_id: {param_name: grad}},
        # ba_params: {param_name: param_data}
        round_idx = args.round_idx + 1
        gradients_each_label = None
        if args.use_creff or args.decoupled_training:
            original_model = copy.deepcopy(self.model)
        if args.decoupled_training:
            gradients_each_label = self.get_gradients_per_class(original_model, train_data, device)

        
        # calculate missing balanced gradients for current client
        sample_per_class = {}
        for batch_idx, (x, labels) in enumerate(train_data):
            for i in range(len(labels)):
                if labels[i].item() in sample_per_class:
                    sample_per_class[labels[i].item()].append(copy.deepcopy(x[i].numpy()))
                else:
                    sample_per_class[labels[i].item()] = [copy.deepcopy(x[i].numpy())]
        # shuffle in each round
        for class_id in sample_per_class:
            random.shuffle(sample_per_class[class_id])
        bal_x = []
        bal_y = []
        bal_batch_classes = {}
        for class_id in sample_per_class:
            if len(sample_per_class[class_id]) >= args.ba_local_balanced_data_num_per_class:
                bal_batch_classes[class_id] = 1
                bal_x.extend(sample_per_class[class_id][: args.ba_local_balanced_data_num_per_class])
                bal_y.extend([class_id for _ in range(args.ba_local_balanced_data_num_per_class)])
        bal_x = torch.tensor(bal_x).to(device)
        bal_y = torch.tensor(bal_y).to(device)

        balanced_grad = None  # {param_name: grad}
        if args.decoupled_training and balanced_gradients is not None:
            class_n = 0
            for class_id in balanced_gradients:
                if class_id not in bal_batch_classes:
                    class_n += 1
                    if balanced_grad is None:
                        balanced_grad = {}
                        for param_name in balanced_gradients[class_id]:
                            balanced_grad[param_name] = balanced_gradients[class_id][param_name]
                    else:
                        for param_name in balanced_gradients[class_id]:
                            balanced_grad[param_name] += balanced_gradients[class_id][param_name]
            if balanced_grad is not None:
                for param_name in balanced_grad:
                    balanced_grad[param_name] /= class_n  # balanced_grad[param_name].norm().item()
                    balanced_grad[param_name] = balanced_grad[param_name].to(device)

        model = self.model

        for param_name, param in model.named_parameters():
            if 'weight' in param_name and (
                    'classifier' in param_name or 'fc' in param_name or 'linear_2' in param_name):
                class_num = param.data.shape[0]

        model.zero_grad()
        if args.decoupled_training and ba_params is not None:
            for k, v in model.named_parameters():
                if 'bias_attractor' in k:
                    for k_ba, v_ba in ba_params.items():
                        if k == k_ba:
                            v.data = v_ba
                            break

        model.to(device)
        model.train()

        # train and update
        if args.local_loss == 'CE':
            criterion = nn.CrossEntropyLoss().to(device)
        elif args.local_loss == 'Focal':
            criterion = focal_loss().to(device)
        elif args.local_loss == 'Ratio':
            criterion = ratio_loss(alpha=alpha, num_classes=class_num).to(device)
        else:
            print("Not implemented loss")
            assert 0 == 1

        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                        momentum=args.client_momentum)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=False)

        epoch_loss = []
        epoch_ce_loss, epoch_ct_loss = [], []
        if args.fl_algorithm == "FedProx":
            global_model = copy.deepcopy(self.model)
        for epoch in range(args.epochs):
            batch_loss = []
            batch_ce_loss, batch_ct_loss = [], []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                if args.decoupled_training:
                    log_probs, final_states = model(x, use_ba=True)
                    if len(bal_batch_classes) > 0:
                        log_probs_linear, _ = model(bal_x, only_linear=True)
                else:
                    log_probs, final_states = model(x)
                
                # CE loss
                # loss on the local data distribution
                classification_loss = criterion(log_probs, labels)
                balanced_grad_batch = None
                if args.decoupled_training and len(bal_batch_classes) > 0 and balanced_gradients is not None and round_idx >=1:
                    criterion_linear = nn.CrossEntropyLoss().to(device)
                    linear_loss = criterion_linear(log_probs_linear, bal_y)
                    linear_loss.backward()
                    if balanced_grad is not None:
                        for k, v in model.named_parameters():
                            if ('classifier' in k or 'fc' in k or 'linear_2' in k):  # and 'weight' in k:
                                for param_name, bal_grad in balanced_grad.items():
                                    if k == param_name:
                                        #v.grad.data = (v.grad.data * len(bal_batch_classes) + (bal_grad * v.grad.data.norm().item() / bal_grad.norm().item()) * (class_num - len(bal_batch_classes))) / class_num
                                        v.grad.data = (v.grad.data * len(bal_batch_classes) + bal_grad * (class_num - len(bal_batch_classes))) / class_num
                                        # v.grad.data *= args.ba_lambda
                                        break
                            # else:
                            #     v.grad.data *= args.ba_lambda
                    balanced_grad_batch = {}
                    for k, v in model.named_parameters():
                        if ('classifier' in k or 'fc' in k or 'linear_2' in k):
                            balanced_grad_batch[k] = copy.deepcopy(v.grad.data)
                    model.zero_grad()
                elif args.decoupled_training and len(bal_batch_classes) == 0 and balanced_gradients is not None and round_idx >=1:
                    balanced_grad_batch = {}
                    for param_name, bal_grad in balanced_grad.items():
                        balanced_grad_batch[param_name] = copy.deepcopy(bal_grad)
                        # v.grad.data += (args.ba_lambda * bal_grad * (v.grad.data.norm().item() / bal_grad.norm().item()))

                loss = classification_loss

                # loss = classification_loss
                if args.fl_algorithm == "FedProx":
                    fed_prox_reg = 0.0
                    mu = args.fedprox_mu
                    for (p, g_p) in zip(self.model.parameters(),
                                        global_model.parameters()):
                        fed_prox_reg += ((mu / 2) * torch.norm((p - g_p.data)) ** 2)
                    loss += fed_prox_reg

            
                loss.backward()

                if args.decoupled_training and balanced_grad_batch is not None and round_idx >=1:
                    for k, v in model.named_parameters():
                        if ('classifier' in k or 'fc' in k or 'linear_2' in k): # and 'weight' in k:
                            for param_name in balanced_grad_batch:
                                if k == param_name:
                                    balanced_grad_batch[param_name] = balanced_grad_batch[param_name] * (v.grad.data.norm().item() / balanced_grad_batch[param_name].norm().item())
                                    # v.grad.data += (args.ba_lambda * bal_grad * (v.grad.data.norm().item() / bal_grad.norm().item()))
                                    break

                # to avoid nan loss
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                # global balanced gradients correction
                if args.decoupled_training and balanced_grad_batch is not None:
                    for k, v in model.named_parameters():
                        if ('classifier' in k or 'fc' in k or 'linear_2' in k):  # and 'weight' in k:
                            for param_name, bal_grad in balanced_grad_batch.items():
                                if k == param_name:
                                    # v.grad.data += (args.ba_lambda * bal_grad)
                                    v.data = v.data - args.lr * args.ba_lambda * bal_grad
                                    break
                # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, (batch_idx + 1) * args.batch_size, len(train_data) * args.batch_size,
                #            100. * (batch_idx + 1) / len(train_data), loss.item()))
                batch_loss.append(loss.item())
                batch_ce_loss.append(classification_loss.item())
                batch_ct_loss.append(classification_loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_ce_loss.append(sum(batch_ce_loss) / len(batch_ce_loss))
            epoch_ct_loss.append(sum(batch_ct_loss) / len(batch_ct_loss))
            logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}\tCE_Loss: {:.6f}\tCT_Loss: {:.6f}'.format(
                self.id, epoch, sum(epoch_loss) / len(epoch_loss), sum(epoch_ce_loss) / len(epoch_ce_loss), sum(epoch_ct_loss) / len(epoch_ct_loss)))

        #if args.decoupled_training:
        #    gradients_each_label = self.get_gradients_per_class(original_model, train_data, device)
        if args.use_creff:
            gradients_each_label = self.creff_get_gradients_per_class(original_model, train_data, device, creff_retrained_params)


        return 0.0, gradients_each_label  # the output balanced gradients is supposed to be a dict
    

    def test(self, test_data, device, args, creff_retrained_params=None):
        # model = self.model
        model = copy.deepcopy(self.model)
        if args.use_creff and creff_retrained_params is not None:
            for param_name, param in model.named_parameters():
                if 'weight' in param_name and (
                        'classifier' in param_name or 'fc' in param_name or 'linear_2' in param_name):
                    param.data = creff_retrained_params.state_dict()['weight']

                    break
        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0
        }
        results_per_class = {}

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred, _ = model(x)
                loss = criterion(pred, target)

                _, predicted = torch.max(pred, -1)
                for j in range(len(predicted)):
                    if target[j].item() in results_per_class:
                        results_per_class[target[j].item()]['test_total'] += 1
                        results_per_class[target[j].item()]['test_correct'] += int(predicted[j].item() == target[j].item())
                    else:
                        results_per_class[target[j].item()] = {'test_total': 0, 'test_correct': 0}
                        results_per_class[target[j].item()]['test_total'] += 1
                        results_per_class[target[j].item()]['test_correct'] += int(predicted[j].item() == target[j].item())

                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
        return metrics, results_per_class

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False

    def get_class_prototype(self, train_data, device):
        model = self.model
        model.to(device)
        model.eval()
        prototypes = {}
        samples = {}

        with torch.no_grad():
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                _, final_states = model(x)
                for j in range(len(final_states)):
                    label = labels[j].item()
                    if label not in prototypes:
                        prototypes[label] = final_states[j].cpu().detach()
                        samples[label] = 1
                    else:
                        prototypes[label] += final_states[j].cpu().detach()
                        samples[label] += 1

            for label in prototypes.keys():
                prototypes[label] /= samples[label]
                prototypes[label] = prototypes[label].to(device)
        return prototypes

    def get_gradients_per_class(self, original_model, train_data, device):
        model = copy.deepcopy(self.model)
        
        model.to(device)
        model.eval()

        # criterion = nn.CrossEntropyLoss().to(device)
        gradients_each_label = {}  # {class_id: {param_name: grad}}
        samples_each_label = {}  # {class_id: num}

        for batch_idx, (x, labels) in enumerate(train_data):
            x, labels = x.to(device), labels.to(device)
            # model.zero_grad()
            probs, final_states = model(x)
            # loss = criterion(log_probs, labels)
            # loss.backward()
            for i in range(len(probs)):
                softmax_prob = torch.softmax(probs[i].data, dim=-1)

                class_id = labels[i].item()
                if class_id not in samples_each_label:
                    samples_each_label[class_id] = 1
                else:
                    samples_each_label[class_id] += 1
                softmax_prob[class_id] -= 1
                cur_weight_grad = torch.matmul(torch.unsqueeze(softmax_prob, dim=1), torch.unsqueeze(final_states[i].data, dim=0))
                cur_bias_grad = copy.deepcopy(softmax_prob)
                if class_id not in gradients_each_label:
                    gradients_each_label[class_id] = {}
                    for param_name in model.state_dict().keys():
                        if 'weight' in param_name and ('classifier' in param_name or 'fc' in param_name or 'linear_2' in param_name):
                            gradients_each_label[class_id][param_name] = copy.deepcopy(cur_weight_grad.cpu())
                        elif 'bias' in param_name and ('classifier' in param_name or 'fc' in param_name or 'linear_2' in param_name):  # bias
                            gradients_each_label[class_id][param_name] = copy.deepcopy(cur_bias_grad.cpu())
                else:
                    for param_name in model.state_dict().keys():
                        if 'weight' in param_name and ('classifier' in param_name or 'fc' in param_name or 'linear_2' in param_name):
                            gradients_each_label[class_id][param_name] += copy.deepcopy(cur_weight_grad.cpu())
                        elif 'bias' in param_name and ('classifier' in param_name or 'fc' in param_name or 'linear_2' in param_name):
                            gradients_each_label[class_id][param_name] += copy.deepcopy(cur_bias_grad.cpu())

        for class_id in gradients_each_label:
            for param_name in gradients_each_label[class_id]:
                gradients_each_label[class_id][param_name] /= samples_each_label[class_id]
        return gradients_each_label

    def creff_get_gradients_per_class(self, original_model, train_data, device, creff_retrained_params):
        model = copy.deepcopy(original_model)
        if creff_retrained_params is not None:
            for param_name, param in model.named_parameters():
                if 'weight' in param_name and (
                        'classifier' in param_name or 'fc' in param_name or 'linear_2' in param_name):
                    param.data = creff_retrained_params.state_dict()['weight']

                if 'bias' in param_name and (
                        'classifier' in param_name or 'fc' in param_name or 'linear_2' in param_name):
                    param.data = creff_retrained_params.state_dict()['bias']

        model.to(device)
        model.eval()

        # criterion = nn.CrossEntropyLoss().to(device)
        gradients_each_label = {}  # {class_id: {param_name: grad}}
        samples_each_label = {}  # {class_id: num}

        for batch_idx, (x, labels) in enumerate(train_data):
            x, labels = x.to(device), labels.to(device)
            # model.zero_grad()
            probs, final_states = model(x)
            # loss = criterion(log_probs, labels)
            # loss.backward()
            for i in range(len(probs)):
                softmax_prob = torch.softmax(probs[i].data, dim=-1)

                class_id = labels[i].item()
                if class_id not in samples_each_label:
                    samples_each_label[class_id] = 1
                else:
                    samples_each_label[class_id] += 1
                softmax_prob[class_id] -= 1
                cur_weight_grad = torch.matmul(torch.unsqueeze(softmax_prob, dim=1),
                                               torch.unsqueeze(final_states[i].data, dim=0))
                cur_bias_grad = copy.deepcopy(softmax_prob)
                if class_id not in gradients_each_label:
                    gradients_each_label[class_id] = {}
                    for param_name in model.state_dict().keys():
                        if 'weight' in param_name and (
                                'classifier' in param_name or 'fc' in param_name or 'linear_2' in param_name):
                            gradients_each_label[class_id][param_name] = copy.deepcopy(cur_weight_grad.cpu())
                        elif 'bias' in param_name and (
                                'classifier' in param_name or 'fc' in param_name or 'linear_2' in param_name):  # bias
                            gradients_each_label[class_id][param_name] = copy.deepcopy(cur_bias_grad.cpu())
                else:
                    for param_name in model.state_dict().keys():
                        if 'weight' in param_name and (
                                'classifier' in param_name or 'fc' in param_name or 'linear_2' in param_name):
                            gradients_each_label[class_id][param_name] += copy.deepcopy(cur_weight_grad.cpu())
                        elif 'bias' in param_name and (
                                'classifier' in param_name or 'fc' in param_name or 'linear_2' in param_name):
                            gradients_each_label[class_id][param_name] += copy.deepcopy(cur_bias_grad.cpu())

        for class_id in gradients_each_label:
            for param_name in gradients_each_label[class_id]:
                gradients_each_label[class_id][param_name] /= samples_each_label[class_id]
        return gradients_each_label
