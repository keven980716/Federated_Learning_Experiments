import logging
import copy
import torch
from torch import nn
from torch.nn import functional as F
import math
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


class MyModelTrainer(ModelTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args, alpha=None, class_num=10):
        local_training_loss = 0.0
        if args.use_climb:
            local_training_loss = self.calculate_local_loss(train_data, device)

        model = self.model

        for param_name, param in model.named_parameters():
            if 'weight' in param_name and (
                    'classifier' in param_name or 'fc' in param_name or 'linear_2' in param_name):
                class_num = param.data.shape[0]

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
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr, momentum=args.client_momentum)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=False)

        epoch_loss = []
        if args.fl_algorithm == "FedProx":
            global_model = copy.deepcopy(self.model)
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                if args.fl_algorithm == "FedProx":
                    fed_prox_reg = 0.0
                    mu = args.fedprox_mu
                    for (p, g_p) in zip(self.model.parameters(),
                                        global_model.parameters()):
                        fed_prox_reg += ((mu / 2) * torch.norm((p - g_p.data)) ** 2)
                    loss += fed_prox_reg
                loss.backward()
                # to avoid nan loss
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, (batch_idx + 1) * args.batch_size, len(train_data) * args.batch_size,
                #            100. * (batch_idx + 1) / len(train_data), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
                self.id, epoch, sum(epoch_loss) / len(epoch_loss)))

           
        return local_training_loss # to do

    def test(self, test_data, device, args):
        model = self.model
        
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
                pred = model(x)
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

    def calculate_local_loss(self, train_data, device):
        model = copy.deepcopy(self.model)
        model.to(device)
        model.eval()

        criterion = nn.CrossEntropyLoss().to(device)
        total_loss = 0.0
        total_sample = 0

        for batch_idx, (x, labels) in enumerate(train_data):
            x, labels = x.to(device), labels.to(device)
            # model.zero_grad()
            probs = model(x)
            loss = criterion(probs, labels)
            total_sample += len(labels)
            total_loss += loss.item() * len(labels)

        return total_loss / total_sample

    def get_gradients_per_class(self, train_data, device):
        model = copy.deepcopy(self.model)
        model.to(device)
        model.eval()

        # criterion = nn.CrossEntropyLoss().to(device)
        gradients_each_label = {}
        samples_each_label = {}

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
                cur_grad = torch.matmul(torch.unsqueeze(softmax_prob, dim=1), torch.unsqueeze(final_states[i].data, dim=0))
                if class_id not in gradients_each_label:
                    gradients_each_label[class_id] = copy.deepcopy(cur_grad.cpu())
                else:
                    gradients_each_label[class_id] += copy.deepcopy(cur_grad.cpu())

        for class_id in gradients_each_label:
            gradients_each_label[class_id] /= samples_each_label[class_id]
        return gradients_each_label




