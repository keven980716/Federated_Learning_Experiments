import logging

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import math
from .datasets import CIFAR10_truncated
import random
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# generate the non-IID distribution for all methods
def read_data_distribution(filename='./data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt'):
    distribution = {}
    with open(filename, 'r') as data:
        for x in data.readlines():
            if '{' != x[0] and '}' != x[0]:
                tmp = x.split(':')
                if '{' == tmp[1].strip():
                    first_level_key = int(tmp[0])
                    distribution[first_level_key] = {}
                else:
                    second_level_key = int(tmp[0])
                    distribution[first_level_key][second_level_key] = int(tmp[1].strip().replace(',', ''))
    return distribution


def read_net_dataidx_map(filename='./data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt'):
    net_dataidx_map = {}
    with open(filename, 'r') as data:
        for x in data.readlines():
            if '{' != x[0] and '}' != x[0] and ']' != x[0]:
                tmp = x.split(':')
                if '[' == tmp[-1].strip():
                    key = int(tmp[0])
                    net_dataidx_map[key] = []
                else:
                    tmp_array = x.split(',')
                    net_dataidx_map[key] = [int(i.strip()) for i in tmp_array]
    return net_dataidx_map


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10():
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, valid_transform


def load_cifar10_data(datadir):
    train_transform, test_transform = _data_transforms_cifar10()

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=train_transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=test_transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)


# the following code is borrow from the data pre-processing for quantity uniformly distributed setting
def dynamic_batch_fill(label_index_tracker, label_index_matrix,
                       remaining_length, current_label_id):
    """
    params
    ------------------------------------------------------------------------
    label_index_tracker : 1d numpy array track how many data each label has used
    label_index_matrix : 2d array list of indexs of each label
    remaining_length : int remaining empty space in current partition client list
    current_label_id : int current round label id
    ------------------------------------------------------------------------

    return
    ---------------------------------------------------------
    label_index_offset: dict  dictionary key is label id
    and value is the offset associated with this key
    ----------------------------------------------------------
    """
    remaining_unfiled = remaining_length
    label_index_offset = {}
    label_remain_length_dict = {}
    total_label_remain_length = 0
    # calculate total number of all the remaing labels and each label's remaining length
    for label_id, label_list in enumerate(label_index_matrix):
        if label_id == current_label_id:
            label_remain_length_dict[label_id] = 0
            continue
        label_remaining_count = len(label_list) - label_index_tracker[label_id]
        if label_remaining_count > 0:
            total_label_remain_length = (total_label_remain_length +
                                         label_remaining_count)
        else:
            label_remaining_count = 0
        label_remain_length_dict[label_id] = label_remaining_count
    length_pointer = remaining_unfiled

    if total_label_remain_length > 0:
        label_sorted_by_length = {
            k: v
            for k, v in sorted(label_remain_length_dict.items(),
                               key=lambda item: item[1])
        }
    else:
        label_index_offset = label_remain_length_dict
        return label_index_offset
    # for each label calculate the offset move forward by distribution of remaining labels
    for label_id in label_sorted_by_length.keys():
        fill_count = math.ceil(label_remain_length_dict[label_id] /
                               total_label_remain_length * remaining_length)
        fill_count = min(fill_count, label_remain_length_dict[label_id])
        offset_forward = fill_count
        # if left room not enough for all offset set it to 0
        if length_pointer - offset_forward <= 0 and length_pointer > 0:
            label_index_offset[label_id] = length_pointer
            length_pointer = 0
            break
        else:
            length_pointer -= offset_forward
            label_remain_length_dict[label_id] -= offset_forward
        label_index_offset[label_id] = offset_forward

    # still has some room unfilled
    if length_pointer > 0:
        for label_id in label_sorted_by_length.keys():
            # make sure no infinite loop happens
            fill_count = math.ceil(label_sorted_by_length[label_id] /
                                   total_label_remain_length * length_pointer)
            fill_count = min(fill_count, label_remain_length_dict[label_id])
            offset_forward = fill_count
            if length_pointer - offset_forward <= 0 and length_pointer > 0:
                label_index_offset[label_id] += length_pointer
                length_pointer = 0
                break
            else:
                length_pointer -= offset_forward
                label_remain_length_dict[label_id] -= offset_forward
            label_index_offset[label_id] += offset_forward

    return label_index_offset


def label_skew_process(dataset, datadir, partition, n_nets, alpha, imbalance_version,
                       imbalanced_ratio, minority_classes_list,
                       need_server_auxiliary_data=False, server_auxiliary_data_per_class=None,
                       retrain_server_classifier=False):
    """
    params
    -------------------------------------------------------------------
    label_vocab : dict label vocabulary of the dataset
    label_assignment : 1d list a list of label, the index of list is the index associated to label
    client_num : int number of clients
    alpha : float similarity of each client, the larger the alpha the similar data for each client
    -------------------------------------------------------------------
    return
    ------------------------------------------------------------------
    partition_result : 2d array list of partition index of each client
    ------------------------------------------------------------------
    """
    logging.info("*********partition data***************")
    X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    n_train = X_train.shape[0]
    # n_test = X_test.shape[0]
    # create imbalanced dataset
    new_X_train, new_y_train = [], []
    id_per_class = {}
    for i in range(n_train):
        if y_train[i] in id_per_class:
            id_per_class[y_train[i]].append(i)
        else:
            id_per_class[y_train[i]] = [i]
    auxiliary_data_server = None
    if need_server_auxiliary_data:
        if not retrain_server_classifier:
            auxiliary_data_server = {}
            for c in id_per_class:
                auxiliary_data_server_X, auxiliary_data_server_y = [], []
                for j in range(server_auxiliary_data_per_class):
                    auxiliary_data_server_X.append(X_train[id_per_class[c][j]])
                    auxiliary_data_server_y.append(y_train[id_per_class[c][j]])
                auxiliary_data_server[c] = {'X': auxiliary_data_server_X.copy(), 'y': auxiliary_data_server_y.copy()}
            # important!!
            auxiliary_data_server = {k: auxiliary_data_server[k] for k in sorted(auxiliary_data_server.keys())}
        else:
            auxiliary_data_server_X, auxiliary_data_server_y = [], []
            for j in range(server_auxiliary_data_per_class):
                for c in id_per_class:
                    auxiliary_data_server_X.append(X_train[id_per_class[c][j]])
                    auxiliary_data_server_y.append(y_train[id_per_class[c][j]])
            auxiliary_data_server = {'X': auxiliary_data_server_X.copy(), 'y': auxiliary_data_server_y.copy()}

    if imbalance_version == 'binary':
        for minority in minority_classes_list:
            random.shuffle(id_per_class[minority])
            class_len = len(id_per_class[minority])
            id_per_class[minority] = id_per_class[minority][: int(class_len / imbalanced_ratio)]
    elif imbalance_version == 'exp_long_tail':
        num_classes = len(id_per_class.keys())
        for class_idx in range(num_classes):
            random.shuffle(id_per_class[class_idx])
            class_len = int(len(id_per_class[class_idx]) * ((1 / imbalanced_ratio) ** (class_idx / (num_classes - 1))))
            id_per_class[class_idx] = id_per_class[class_idx][: class_len]
    else:
        print('not implemented imbelance version')
        pass
    available_ids = []
    for c in id_per_class:
        for id in id_per_class[c]:
            available_ids.append(id)
            new_X_train.append(X_train[id])
            new_y_train.append(y_train[id])
    X_train = np.array(new_X_train)
    y_train = np.array(new_y_train)
    print(X_train.shape)
    print(y_train.shape)
    n_train = X_train.shape[0]

    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        # the following code is adopted (also modified) from FedNLP
        label_vocab = {i: i for i in range(10)}
        label_assignment = y_train
        data_length = y_train.shape[0]
        label_index_matrix = [[] for _ in label_vocab]
        label_proportion = []
        partition_result = [[] for _ in range(n_nets)]
        client_length = 0
        print("client_num", n_nets)
        # shuffle indexs and calculate each label proportion of the dataset
        for index, value in enumerate(label_vocab):
            label_location = np.where(label_assignment == value)[0]
            label_proportion.append(len(label_location) / data_length)
            np.random.shuffle(label_location)
            label_index_matrix[index].extend(label_location[:])
        print(label_proportion)
        # calculate size for each partition client
        label_index_tracker = np.zeros(len(label_vocab), dtype=int)
        total_index = data_length
        each_client_index_length = int(total_index / n_nets)
        print("each index length", each_client_index_length)
        client_dir_dis = np.array([alpha * l for l in label_proportion])
        print("alpha", alpha)
        print("client dir dis", client_dir_dis)
        proportions = np.random.dirichlet(client_dir_dis)
        print("dir distribution", proportions)
        # add all the unused data to the client
        for client_id in range(len(partition_result)):
            each_client_partition_result = partition_result[client_id]
            proportions = np.random.dirichlet(client_dir_dis)
            client_length = min(each_client_index_length, total_index)
            if total_index < client_length * 2:
                client_length = total_index
            total_index -= client_length
            client_length_pointer = client_length
            # for each label calculate the offset length assigned to by Dir distribution and then extend assignment
            for label_id, _ in enumerate(label_vocab):
                offset = round(proportions[label_id] * client_length)
                if offset >= client_length_pointer:
                    offset = client_length_pointer
                    client_length_pointer = 0
                else:
                    if label_id == (len(label_vocab) - 1):
                        offset = client_length_pointer
                    client_length_pointer -= offset

                start = int(label_index_tracker[label_id])
                end = int(label_index_tracker[label_id] + offset)
                label_data_length = len(label_index_matrix[label_id])
                # if the the label is assigned to a offset length that is more than what its remaining length
                if end > label_data_length:
                    each_client_partition_result.extend(
                        label_index_matrix[label_id][start:])
                    label_index_tracker[label_id] = label_data_length
                    label_index_offset = dynamic_batch_fill(
                        label_index_tracker, label_index_matrix,
                        end - label_data_length, label_id)
                    for fill_label_id in label_index_offset.keys():
                        start = label_index_tracker[fill_label_id]
                        end = (label_index_tracker[fill_label_id] +
                               label_index_offset[fill_label_id])
                        each_client_partition_result.extend(
                            label_index_matrix[fill_label_id][start:end])
                        label_index_tracker[fill_label_id] = (
                                label_index_tracker[fill_label_id] +
                                label_index_offset[fill_label_id])
                else:
                    each_client_partition_result.extend(
                        label_index_matrix[label_id][start:end])
                    label_index_tracker[
                        label_id] = label_index_tracker[label_id] + offset

            # if last client still has empty rooms, fill empty rooms with the rest of the unused data
            if client_id == len(partition_result) - 1:
                print("last id length", len(each_client_partition_result))
                print("Last client fill the rest of the unfilled lables.")
                for not_fillall_label_id in range(len(label_vocab)):
                    if label_index_tracker[not_fillall_label_id] < len(
                            label_index_matrix[not_fillall_label_id]):
                        print("fill more id", not_fillall_label_id)
                        start = label_index_tracker[not_fillall_label_id]
                        each_client_partition_result.extend(
                            label_index_matrix[not_fillall_label_id][start:])
                        label_index_tracker[not_fillall_label_id] = len(
                            label_index_matrix[not_fillall_label_id])
            partition_result[client_id] = each_client_partition_result

        net_dataidx_map = {}

        for j in range(n_nets):
            np.random.shuffle(partition_result[j])
            net_dataidx_map[j] = partition_result[j]

    elif partition == "hetero-fix":
        dataidx_map_file_path = './data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt'
        net_dataidx_map = read_net_dataidx_map(dataidx_map_file_path)

    if partition == "hetero-fix":
        distribution_file_path = './data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt'
        traindata_cls_counts = read_data_distribution(distribution_file_path)
    else:
        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts, auxiliary_data_server


def partition_data(dataset, datadir, partition, n_nets, alpha):
    logging.info("*********partition data***************")
    X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    n_train = X_train.shape[0]
    # n_test = X_test.shape[0]

    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        min_size = 0
        K = 10
        N = y_train.shape[0]
        logging.info("N = " + str(N))
        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition == "hetero-fix":
        dataidx_map_file_path = './data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt'
        net_dataidx_map = read_net_dataidx_map(dataidx_map_file_path)

    if partition == "hetero-fix":
        distribution_file_path = './data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt'
        traindata_cls_counts = read_data_distribution(distribution_file_path)
    else:
        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts


# for centralized training
def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):
    return get_dataloader_CIFAR10(datadir, train_bs, test_bs, dataidxs)


# for local devices
def get_dataloader_test(dataset, datadir, train_bs, test_bs, dataidxs_train, dataidxs_test):
    return get_dataloader_test_CIFAR10(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test)


def get_dataloader_CIFAR10(datadir, train_bs, test_bs, dataidxs=None):
    dl_obj = CIFAR10_truncated

    transform_train, transform_test = _data_transforms_cifar10()

    train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)

    return train_dl, test_dl


def get_dataloader_CIFAR10_auxiliary(datadir, data_points_dict, bs):
    dl_obj = CIFAR10_truncated

    transform_train, transform_test = _data_transforms_cifar10()

    train_ds = dl_obj(datadir, dataidxs=None, train=True, transform=transform_train, download=True)
    # test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

    new_X = np.array(data_points_dict['X'].copy())
    new_y = np.array(data_points_dict['y'].copy())
    train_ds.data = new_X
    train_ds.target = new_y

    train_dl = data.DataLoader(dataset=train_ds, batch_size=bs, shuffle=True, drop_last=False)
    # test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)

    return train_dl


def get_dataloader_test_CIFAR10(datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None):
    dl_obj = CIFAR10_truncated

    transform_train, transform_test = _data_transforms_cifar10()

    train_ds = dl_obj(datadir, dataidxs=dataidxs_train, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, dataidxs=dataidxs_test, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)

    return train_dl, test_dl


def load_partition_data_distributed_cifar10(process_id, dataset, data_dir, partition_method, partition_alpha,
                                            client_number, batch_size):
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(dataset,
                                                                                             data_dir,
                                                                                             partition_method,
                                                                                             client_number,
                                                                                             partition_alpha)
    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    # get global test data
    if process_id == 0:
        train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
        logging.info("train_dl_global number = " + str(len(train_data_global)))
        logging.info("test_dl_global number = " + str(len(test_data_global)))
        train_data_local = None
        test_data_local = None
        local_data_num = 0
    else:
        # get local dataset
        dataidxs = net_dataidx_map[process_id - 1]
        local_data_num = len(dataidxs)
        logging.info("rank = %d, local_sample_number = %d" % (process_id, local_data_num))
        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
                                                 dataidxs)
        logging.info("process_id = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            process_id, len(train_data_local), len(test_data_local)))
        train_data_global = None
        test_data_global = None
    return train_data_num, train_data_global, test_data_global, local_data_num, train_data_local, test_data_local, class_num


def load_partition_data_cifar10(dataset, data_dir, partition_method, partition_alpha, imbalance_version, imbalanced_ratio,
                                minority_classes, client_number, batch_size, need_server_auxiliary_data=False,
                                server_auxiliary_data_per_class=None, retrain_server_classifier=False):
    # X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(dataset,
    #                                                                                          data_dir,
    #                                                                                          partition_method,
    #                                                                                          client_number,
    #                                                                                          partition_alpha)
    # use the following code to avoid quantity skew
    minority_classes = minority_classes.split('_')
    minority_classes_list = [int(k) for k in minority_classes]
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts, \
    auxiliary_data_server = label_skew_process(dataset,
                                               data_dir,
                                               partition_method,
                                               client_number,
                                               partition_alpha,
                                               imbalance_version,
                                               imbalanced_ratio,
                                               minority_classes_list,
                                               need_server_auxiliary_data,
                                               server_auxiliary_data_per_class,
                                               retrain_server_classifier
                                               )
    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))
    test_data_num = len(test_data_global)

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        dataidxs = net_dataidx_map[client_idx]
        local_data_num = len(dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
                                                 dataidxs)
        new_X, new_y = [], []
        for id in dataidxs:
            new_X.append(X_train[id])
            new_y.append(y_train[id])
        new_X = np.array(new_X.copy())
        new_y = np.array(new_y.copy())
        print(train_data_local.dataset.target)
        train_data_local.dataset.data = new_X
        train_data_local.dataset.target = new_y
        print(train_data_local.dataset.target)

        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
    auxiliary_data_server_dl = None
    if auxiliary_data_server is not None:
        if not retrain_server_classifier:
            auxiliary_data_server_dl = {}
            for id in auxiliary_data_server:
                aux_data_dl = get_dataloader_CIFAR10_auxiliary(data_dir, auxiliary_data_server[id], batch_size)
                auxiliary_data_server_dl[id] = aux_data_dl
        else:
            auxiliary_data_server_dl = get_dataloader_CIFAR10_auxiliary(data_dir, auxiliary_data_server, batch_size)
    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, auxiliary_data_server_dl
