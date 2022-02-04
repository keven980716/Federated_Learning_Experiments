import json
import os

import numpy as np
import torch
import logging

import torch.utils.data as data
import torchvision.transforms as transforms
import math
from torchvision.datasets.mnist import MNIST
from PIL import Image
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class MNIST_truncated(data.Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=True):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        mnist_dataobj = MNIST(self.root, self.train, self.transform, self.target_transform, self.download)

        data = mnist_dataobj.data
        target = mnist_dataobj.targets

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


def read_data_distribution(filename='./data_preprocessing/non-iid-distribution/MNIST/distribution.txt'):
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


def read_net_dataidx_map(filename='./data_preprocessing/non-iid-distribution/MNIST/net_dataidx_map.txt'):
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


def _data_transforms_mnist():
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    return train_transform, valid_transform


def load_mnist_data(datadir, download=True):
    train_transform, test_transform = _data_transforms_mnist()

    mnist_train_ds = MNIST_truncated(datadir, train=True, download=download, transform=train_transform)
    mnist_test_ds = MNIST_truncated(datadir, train=False, download=download, transform=test_transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    return (X_train, y_train, X_test, y_test)


# for centralized training
def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):
    return get_dataloader_mnist(datadir, train_bs, test_bs, dataidxs)


# for local devices
def get_dataloader_test(dataset, datadir, train_bs, test_bs, dataidxs_train, dataidxs_test):
    return get_dataloader_test_mnist(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test)


def get_dataloader_mnist(datadir, train_bs, test_bs, dataidxs=None):
    dl_obj = MNIST_truncated

    transform_train, transform_test = _data_transforms_mnist()

    train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl


def get_dataloader_test_mnist(datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None):
    dl_obj = MNIST_truncated

    transform_train, transform_test = _data_transforms_mnist()

    train_ds = dl_obj(datadir, dataidxs=dataidxs_train, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, dataidxs=dataidxs_test, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl


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


def label_skew_process(dataset, datadir, partition, n_nets, alpha):
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
    X_train, y_train, X_test, y_test = load_mnist_data(datadir)
    n_train = X_train.shape[0]
    # n_test = X_test.shape[0]

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
        dataidx_map_file_path = './data_preprocessing/non-iid-distribution/MNIST/net_dataidx_map.txt'
        net_dataidx_map = read_net_dataidx_map(dataidx_map_file_path)

    if partition == "hetero-fix":
        distribution_file_path = './data_preprocessing/non-iid-distribution/MNIST/distribution.txt'
        traindata_cls_counts = read_data_distribution(distribution_file_path)
    else:
        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return X_train, y_train, X_test, y_test , net_dataidx_map, traindata_cls_counts


def load_partition_data_mnist(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size):
    # X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(dataset,
    #                                                                                          data_dir,
    #                                                                                          partition_method,
    #                                                                                          client_number,
    #                                                                                          partition_alpha)
    # use the following code to avoid quantity skew
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = label_skew_process(dataset,
                                                                                             data_dir,
                                                                                             partition_method,
                                                                                             client_number,
                                                                                             partition_alpha)
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
        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size, dataidxs)
        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num


def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of non-unique client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = sorted(cdata['users'])

    return clients, groups, train_data, test_data


def batch_data(data, batch_size):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    batch_data = list()
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i + batch_size]
        batched_y = data_y[i:i + batch_size]
        batched_x = torch.from_numpy(np.asarray(batched_x)).float()
        batched_y = torch.from_numpy(np.asarray(batched_y)).long()
        batch_data.append((batched_x, batched_y))
    return batch_data


def load_partition_data_mnist_by_device_id(batch_size,
                                           device_id,
                                           train_path="MNIST_mobile",
                                           test_path="MNIST_mobile"):
    train_path += '/' + device_id + '/' + 'train'
    test_path += '/' + device_id + '/' + 'test'
    return load_partition_data_mnist(batch_size, train_path, test_path)


def load_partition_data_mnist_old(batch_size,
                              train_path="./../../../data/MNIST/train",
                              test_path="./../../../data/MNIST/test"):
    users, groups, train_data, test_data = read_data(train_path, test_path)

    if len(groups) == 0:
        groups = [None for _ in users]
    train_data_num = 0
    test_data_num = 0
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    train_data_global = list()
    test_data_global = list()
    client_idx = 0
    for u, g in zip(users, groups):
        user_train_data_num = len(train_data[u]['x'])
        user_test_data_num = len(test_data[u]['x'])
        train_data_num += user_train_data_num
        test_data_num += user_test_data_num
        train_data_local_num_dict[client_idx] = user_train_data_num

        # transform to batches
        train_batch = batch_data(train_data[u], batch_size)
        test_batch = batch_data(test_data[u], batch_size)

        # index using client index
        train_data_local_dict[client_idx] = train_batch
        test_data_local_dict[client_idx] = test_batch
        train_data_global += train_batch
        test_data_global += test_batch
        client_idx += 1
    client_num = client_idx
    class_num = 10

    return client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
           train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num