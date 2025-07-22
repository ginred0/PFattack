import argparse
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision.datasets import CIFAR10

import flwr as fl

from module_test.modules import RegressionModel
from module_test.client import FlowerClient
from module_test.FedCustom import FedCustom
from module_test.FL_customized import FedAvg_Clipping_Median
from flwr.common import Metrics
from typing import List, Tuple
from collections import OrderedDict
import numpy as np

import os


import sys

from module_test.load_datasets import StandardScaler


from dataset.acs.create_dataset import (
    load_datasets,
    Federated_Subset,
    Federated_ConcatDataset,
)


# from dataset.adult.create_dataset import (
#     load_datasets_adult,
#     Federated_ConcatDataset_Adult,
# )

from custom.strategy.FairFed import FairFed
from custom.preprocessing.FairBatchSampler_Multi import FairBatch
from torch.utils.data import Subset
from flwr.common import parameters_to_ndarrays, NDArrays, ndarrays_to_parameters
from functools import reduce
import random
import copy

DEVICE = torch.device("cuda")
print(f"Training on {DEVICE} using Pytorch {torch.__version__} Flower {fl.__version__}")


def setup_seed(seed=0):
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.deterministic = True  # 为了保证每次结果一样


""" parameters configuration"""
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", type=str, default="2")

parser.add_argument("--survey_year", type=int, default=2022)
parser.add_argument("--horizon", type=str, default="1-Year")
parser.add_argument("--survey", type=str, default="person")
parser.add_argument("--root_dir", type=str, default="dataset/acs")
parser.add_argument("--train_dir", type=str, default="dataset/adult/train")
parser.add_argument("--test_dir", type=str, default="dataset/adult/test")
parser.add_argument("--client_num", type=int, default=10)  # acs: 51 adult:11
parser.add_argument("--n_feats", type=int, default=18)  # acs: 18 adult:94
parser.add_argument("--training_rounds", type=int, default=1)
parser.add_argument("--communication_rounds", type=int, default=5)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--select_rate", type=float, default=0.5)
parser.add_argument("--gamma", type=float, default=1)
parser.add_argument("--attacker_rate", type=float, default=0.02)
parser.add_argument("--alpha", type=float, default=0.1)
parser.add_argument("--beta", type=float, default=1)
parser.add_argument("--infered_weight", type=float, default=0.1)
parser.add_argument("--single_choose", action="store_true")
parser.add_argument("--attack_type", type=str, default="s")  # n/s/m
parser.add_argument(
    "--dataset", type=str, default="ACSPublicCoverage"
)  # ACSPublicCoverage/ACSIncome/ACSEmployment/ACSMobility/ACSTravelTime/ACSPublicCoverage_M
# parser.add_argument("--attacker_rate", type=float, default=0.02)
# parser.add_argument("--num_rounds", type=int, default=5)
# parser.add_argument("--communication_rounds", type=int, default=5)
parser.add_argument(
    "--defense", type=str, default="krum"
)  # trimmed mean, median, Krum, Bulyan, norm clipping.
parser.add_argument("--metric", type=str, default="dp")
parser.add_argument(
    "--aggragation", type=str, default="fairfed"
)  # fairfed/fairAvg/qfed
parser.add_argument("--q", type=float, default=2)
args = parser.parse_args()

if args.dataset == "ACSPublicCoverage":
    args.n_feats = 18
    All_A_unique_values = [1.0, 2.0]
    All_Y_unique_values = [0.0, 1.0]
if args.dataset == "ACSPublicCoverage_M":
    args.n_feats = 17
    All_A_unique_values = [
        np.array([1.0, 1.0]),
        np.array([1.0, 2.0]),
        np.array([2.0, 1.0]),
        np.array([2.0, 2.0]),
    ]
    All_Y_unique_values = [0, 1]
# if args.dataset == "ACSIncome":
#     args.n_feats = 94
#     args.client_num = 11
#     All_A_unique_values = [1.0, 2.0]
#     All_Y_unique_values = [0, 1]
if args.dataset == "ACSIncome":
    args.n_feats = 9
    All_A_unique_values = [1.0, 2.0]
    All_Y_unique_values = [0.0, 1.0]
if args.dataset == "ACSEmployment":
    args.n_feats = 17
    All_A_unique_values = [1.0, 2.0]
    All_Y_unique_values = [0.0, 1.0]
if args.dataset == "ACSMobility":
    args.n_feats = 20
    All_A_unique_values = [1.0, 2.0]
    All_Y_unique_values = [0.0, 1.0]
if args.dataset == "ACSTravelTime":
    args.n_feats = 16
    All_A_unique_values = [1.0, 2.0]
    All_Y_unique_values = [0.0, 1.0]
if args.dataset == "ACSHealthInsurance":
    args.n_feats = 18
    All_A_unique_values = [1.0, 2.0]
    All_Y_unique_values = [0.0, 1.0]
"""load datasets"""
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
# acs


_, _, train_datasets, val_datasets = load_datasets(
    function=args.dataset,
    REDUCE=False,
    sample_rate=1,
)

print(train_datasets[0].X.shape)

all_val_dataset = Federated_ConcatDataset(val_datasets)
testloader = DataLoader(
    all_val_dataset, batch_size=len(all_val_dataset), shuffle=True, num_workers=4
)


net = RegressionModel(args.n_feats).to(DEVICE)
scaler = StandardScaler()


def norm_grad(grad_list):
    # input: nested gradients
    # output: square of the L-2 norm

    client_grads = grad_list[0]  # shape now: (784, 26)

    for i in range(1, len(grad_list)):
        client_grads = np.append(client_grads, grad_list[i])  # output a flattened array

    return np.sum(np.square(client_grads))


def euclid(v1, v2):
    diff = v1 - v2
    # return torch.matmul(diff, diff.T)
    return np.dot(diff, diff.T)


def acc_backdoor_attackmodel(net, trainloader, epochs: int):
    initial_params = {name: param.clone() for name, param in net.named_parameters()}
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-1)
    net.train()
    for epoch in range(epochs):
        for batch_idx, (X, Y, A) in enumerate(trainloader):
            X = scaler.fit_transform(X)

            X, Y, A = (
                X.float().to(DEVICE),
                (
                    Y.float().to(DEVICE).flatten()
                    if Y.dim() == 2
                    else Y.float().to(DEVICE)
                ),
                A.float().to(DEVICE),
            )

            optimizer.zero_grad()
            ys_pre = net(X).flatten()
            ys = torch.sigmoid(ys_pre)

            l2_reg = sum(
                (param - initial_params[name]).pow(2).sum()
                for name, param in net.named_parameters()
            )

            loss = criterion(
                ys,
                Y,
            )
            loss = 0.2 * loss + 0.8 * l2_reg

            loss.backward()
            optimizer.step()


def get_bool_index(arr, values):
    bool_index = np.isin(arr, values).all(axis=1)
    return bool_index


# the attacker training functions
def train_attackmodel(net, trainloader, epochs: int, lambdas: Dict):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    net.train()
    for _ in range(epochs):
        for _, (X, Y, A) in enumerate(trainloader):
            X = scaler.fit_transform(X)
            X, Y, A = (
                X.float().to(DEVICE),
                (
                    Y.float().to(DEVICE).flatten()
                    if Y.dim() == 2
                    else Y.float().to(DEVICE)
                ),
                A.float().to(DEVICE),
            )

            optimizer.zero_grad()
            ys_pre = net(X).flatten()
            ys = torch.sigmoid(ys_pre)

            # 1
            # group_losses = {str(i): None for i in All_A_unique_values}
            # A_unique_values = torch.unique(A, dim=0).cpu().numpy()

            # for A_unique_value in A_unique_values:
            #     A_mask = get_bool_index(A.view(-1, 1).cpu().numpy(), [A_unique_value])
            #     A_mask = torch.from_numpy(A_mask).to(DEVICE)
            #     group_loss = criterion(
            #         ys.masked_select(A_mask),
            #         Y.masked_select(A_mask),
            #     )

            #     group_losses[str(A_unique_value)] = group_loss

            # lambdas_sum = 0
            # for A_unique_value in A_unique_values:
            #     if group_losses[str(A_unique_value)] != None:
            #         lambdas_sum += lambdas[str(A_unique_value)]
            # for A_unique_value in A_unique_values:
            #     if group_losses[str(A_unique_value)] != None:
            #         lambdas[str(A_unique_value)] /= lambdas_sum

            # loss = sum(
            #     [
            #         lambdas[str(A_unique_value)] * group_losses[str(A_unique_value)]
            #         for A_unique_value in A_unique_values
            #         if group_losses[str(A_unique_value)] != None
            #     ]
            # )
            # 2
            group_losses = {
                str(i) + str(j): None
                for j in All_Y_unique_values
                for i in All_A_unique_values
            }
            A_unique_values = torch.unique(A, dim=0).cpu().numpy()
            Y_unique_values = torch.unique(Y, dim=0).cpu().numpy()
            for A_unique_value in A_unique_values:
                for Y_unique_value in Y_unique_values:
                    A_mask = get_bool_index(
                        A.view(-1, 1).cpu().numpy(), [A_unique_value]
                    )
                    Y_mask = get_bool_index(
                        Y.view(-1, 1).cpu().numpy(), [Y_unique_value]
                    )
                    A_mask = torch.from_numpy(A_mask).to(DEVICE)
                    Y_mask = torch.from_numpy(Y_mask).to(DEVICE)
                    group_loss = criterion(
                        ys.masked_select(A_mask & Y_mask),
                        Y.masked_select(A_mask & Y_mask),
                    )

                    group_losses[str(A_unique_value) + str(Y_unique_value)] = group_loss

            lambdas_sum = 0
            for A_unique_value in A_unique_values:
                for Y_unique_value in Y_unique_values:
                    if group_losses[str(A_unique_value) + str(Y_unique_value)] != None:
                        lambdas_sum += lambdas[
                            str(A_unique_value) + str(Y_unique_value)
                        ]
            for A_unique_value in A_unique_values:
                for Y_unique_value in Y_unique_values:
                    if group_losses[str(A_unique_value) + str(Y_unique_value)] != None:
                        lambdas[
                            str(A_unique_value) + str(Y_unique_value)
                        ] /= lambdas_sum

            loss = sum(
                [
                    lambdas[str(A_unique_value) + str(Y_unique_value)]
                    * group_losses[str(A_unique_value) + str(Y_unique_value)]
                    for A_unique_value in A_unique_values
                    for Y_unique_value in Y_unique_values
                    if group_losses[str(A_unique_value) + str(Y_unique_value)] != None
                ]
            )
            loss.backward()
            optimizer.step()


# the usual training and test functions
def train(net, trainloader, epochs: int, verbose=False):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0

        for batch_idx, (X, Y, A) in enumerate(trainloader):
            X = scaler.fit_transform(X)
            X, Y, A = (
                X.float().to(DEVICE),
                (
                    Y.float().to(DEVICE).flatten()
                    if Y.dim() == 2
                    else Y.float().to(DEVICE)
                ),
                A.float().to(DEVICE),
            )

            optimizer.zero_grad()
            ys_pre = net(X).flatten()
            ys = torch.sigmoid(ys_pre)
            hat_ys = (ys >= 0.5).float()

            loss = criterion(ys, Y)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += Y.size(0)
            correct += ((hat_ys == Y)).sum().item()

        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        # test_loss, test_accuracy, test_bias = test(net, valloader)
        if verbose:
            print(f"Epoch {epoch+1}:  loss {epoch_loss}, accuracy {epoch_acc}")


def test(net, testloader, sign=False):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    # correct, total, loss = 0, 0, 0.0
    correct, Nf1, Nf, total, loss = 0, {}, {}, 0, 0.0
    net.eval()
    with torch.no_grad():
        for X, Y, A in testloader:
            X = scaler.fit_transform(X)

            X, Y, A = (
                X.float().to(DEVICE),
                Y.float().to(DEVICE),
                A.float().to(DEVICE),
            )

            #
            # A_unique_values = torch.unique(A, dim=1)
            # print( A_unique_values.shape)
            A_unique_values = torch.unique(A, dim=0)
            # print( A_unique_values.shape)
            ys_pre = net(X).flatten()
            ys = torch.sigmoid(ys_pre)
            hat_ys = (ys >= 0.5).float()
            loss += criterion(ys, Y).item()
            total += Y.size(0)
            correct += (hat_ys == Y).sum().item()

            A = A.view(-1, 1)
            if args.metric == "dp":
                for A_unique_value in A_unique_values:
                    if str(A_unique_value.cpu().numpy()) not in Nf:
                        Nf[str(A_unique_value.cpu().numpy())] = (
                            torch.all(A == A_unique_value, dim=1).sum().item()
                        )
                        Nf1[str(A_unique_value.cpu().numpy())] = torch.sum(
                            torch.all(A == A_unique_value, dim=1) & (hat_ys == 1)
                        ).item()
                    else:
                        Nf[str(A_unique_value.cpu().numpy())] += (
                            torch.all(A == A_unique_value, dim=1).sum().item()
                        )
                        Nf1[str(A_unique_value.cpu().numpy())] += torch.sum(
                            torch.all(A == A_unique_value, dim=1) & (hat_ys == 1)
                        ).item()
            # EO
            if args.metric == "eqodds":
                for A_unique_value in A_unique_values:
                    A_unique_value = A_unique_value.cpu().numpy()
                    if str(A_unique_value.cpu().numpy()) not in Nf:
                        Nf[str(A_unique_value.cpu().numpy())] = torch.sum(
                            (A == A_unique_value) & (Y == 1)
                        ).item()
                        Nf1[str(A_unique_value.cpu().numpy())] = torch.sum(
                            (A == A_unique_value) & (hat_ys == 1) & (Y == 1)
                        ).item()
                    else:
                        Nf[str(A_unique_value.cpu().numpy())] += torch.sum(
                            (A == A_unique_value) & (Y == 1)
                        ).item()
                        Nf1[str(A_unique_value.cpu().numpy())] += torch.sum(
                            (A == A_unique_value) & (hat_ys == 1) & (Y == 1)
                        ).item()

    loss /= len(testloader.dataset)
    accuracy = correct / total
    # 2-value sensitive
    all_bias = []
    for A_unique_value in A_unique_values:
        bias = Nf1[str(A_unique_value.cpu().numpy())] / Nf[
            str(A_unique_value.cpu().numpy())
        ] - sum(
            [
                Nf1[str(value.cpu().numpy())]
                for value in A_unique_values
                if (not torch.equal(value, A_unique_value))
                and str(value.cpu().numpy()) in Nf1
            ]
        ) / sum(
            [
                Nf[str(value.cpu().numpy())]
                for value in A_unique_values
                if (not torch.equal(value, A_unique_value))
                and str(value.cpu().numpy()) in Nf
            ]
        )
        all_bias.append(bias)
    max_abs_index = max(range(len(all_bias)), key=lambda index: abs(all_bias[index]))
    if not sign:
        return loss, accuracy, abs(all_bias[max_abs_index]), Nf, Nf1
    else:
        return loss, accuracy, all_bias[max_abs_index], Nf, Nf1


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


class Client:
    def __init__(self, args, client_id) -> None:
        self.id = client_id
        self.attacker = False
        self.local_model = RegressionModel(args.n_feats).to(DEVICE)

        self.ori_trainloader = DataLoader(
            train_datasets[client_id],
            batch_size=len(train_datasets[client_id]),
            shuffle=True,
            num_workers=4,
        )

        sampler = FairBatch(
            self.local_model,
            train_datasets[client_id].X,
            train_datasets[client_id].Y,  # True False
            train_datasets[client_id].A,  # 1 2
            batch_size=256,
            alpha=args.alpha,
            target_fairness=args.metric,
            replacement=False,
            seed=0,
        )

        self.trainloader = DataLoader(
            train_datasets[client_id], sampler=sampler, num_workers=4
        )

        # self.trainloader = DataLoader(
        #     train_datasets[client_id],
        #     batch_size=256,
        #     shuffle=True,
        #     num_workers=4,
        # )

        self.valloader = DataLoader(
            val_datasets[client_id],
            batch_size=len(val_datasets[client_id]),
            shuffle=True,
            num_workers=4,
        )

        self.epochs = args.epochs
        self.infered_weight = args.infered_weight
        self.infered_weight_his = []
        self.beta = 0
        self.target_attack_model = None
        self.lambdas = {
            str(i) + str(j): 1 / (len(All_A_unique_values) * len(All_Y_unique_values))
            for i in All_A_unique_values
            for j in All_Y_unique_values
        }

        self.gamma = args.gamma
        self.last_local_model = copy.deepcopy(self.local_model)
        self.first_attack = True

    def receive_param(self, param):
        set_parameters(self.local_model, param)

    def send_param(self):
        return get_parameters(self.local_model)

    def train(self, current_round, communication_rounds):
        if not self.attacker:
            train(self.local_model, self.trainloader, self.epochs, False)
        else:
            if current_round < communication_rounds - 1:
                train(self.local_model, self.trainloader, self.epochs, False)
            else:
                if current_round == communication_rounds:
                    self.first_attack = False

                if (
                    current_round
                    == communication_rounds
                    # or current_round == communication_rounds - 1
                ):
                    G_last_parameter = get_parameters(self.last_local_model)
                    G_parameter = get_parameters(self.local_model)
                    X_parameter = get_parameters(self.target_attack_model)

                    # print(
                    #     f"G_parameter {G_parameter} X_parameter {X_parameter} G_last_parameter {G_last_parameter}"
                    # )
                    ratio = [
                        np.sum(np.abs(g - g_last)) / np.sum(np.abs(x - g_last))
                        for g, x, g_last in zip(
                            G_parameter, X_parameter, G_last_parameter
                        )
                    ]
                    ratio = sum(ratio) / len(ratio)

                    self.infered_weight *= ratio
                self.infered_weight_his.append(self.infered_weight)
                self.attack_strategy()

                # difference between send and received global model

    def test(self, current_round, communication_rounds):
        loss, accuracy, bias, _, _ = test(self.local_model, self.valloader)
        if self.attacker and current_round >= communication_rounds - 2:
            # if self.attacker:
            return loss, accuracy, 0
        else:
            return loss, accuracy, bias

    def get_datasize(self):
        return len(train_datasets[self.id])

    def weight_inference(self):
        self.infered_weight = 0.2

    def attack_strategy(self):
        self.last_local_model = copy.deepcopy(self.local_model)
        print("Attack!")
        # DP
        G_parameter = get_parameters(self.local_model)

        if self.first_attack == True:
            local_ref_model = RegressionModel(args.n_feats).to(DEVICE)
            train(local_ref_model, self.trainloader, self.epochs, False)
            _, _, bias_l, Nf_l, Nf1_l = test(local_ref_model, self.valloader, sign=True)
            _, _, bias_g, Nf_g, Nf1_g = test(
                self.local_model, self.valloader, sign=True
            )

            print(f"local_bias: {bias_l} global_bias: {bias_g}")

            for A_unique_value in All_A_unique_values:
                if str(A_unique_value) in Nf1_l:
                    if (
                        self.lambdas[str(A_unique_value) + str(1.0)] > 0.01
                        and self.lambdas[str(A_unique_value) + str(1.0)] <= 0.99
                    ):
                        self.lambdas[str(A_unique_value) + str(1.0)] -= self.gamma * (
                            (
                                Nf1_g[str(A_unique_value)] / Nf_g[str(A_unique_value)]
                                - Nf1_l[str(A_unique_value)] / Nf_l[str(A_unique_value)]
                            )
                            / abs(
                                Nf1_l[str(A_unique_value)] / Nf_l[str(A_unique_value)]
                            )
                        )
                    if (
                        self.lambdas[str(A_unique_value) + str(0.0)] > 0.01
                        and self.lambdas[str(A_unique_value) + str(0.0)] <= 0.99
                    ):
                        self.lambdas[str(A_unique_value) + str(0.0)] += self.gamma * (
                            (
                                Nf1_g[str(A_unique_value)] / Nf_g[str(A_unique_value)]
                                - Nf1_l[str(A_unique_value)] / Nf_l[str(A_unique_value)]
                            )
                            / abs(
                                Nf1_l[str(A_unique_value)] / Nf_l[str(A_unique_value)]
                            )
                        )
            for A_unique_value in All_A_unique_values:
                for Y_unique_value in All_Y_unique_values:
                    if self.lambdas[str(A_unique_value) + str(Y_unique_value)] <= 0.01:
                        self.lambdas[str(A_unique_value) + str(Y_unique_value)] = 0.01
                    if self.lambdas[str(A_unique_value) + str(Y_unique_value)] >= 0.99:
                        self.lambdas[str(A_unique_value) + str(Y_unique_value)] = 0.99
            # print(f"lambdas: {self.lambdas}")

            train_attackmodel(
                self.local_model,
                self.ori_trainloader,
                20,
                self.lambdas,
            )

            self.target_attack_model = copy.deepcopy(self.local_model)
            _, acc_test, bias_test, _, _ = test(local_ref_model, testloader)
            print(f"acc_test {acc_test} bias_test {bias_test}")
        X_parameter = get_parameters(self.target_attack_model)
        L_parameter = [
            (X_parameter[i] - G_parameter[i]) / self.infered_weight + G_parameter[i]
            for i in range(len(X_parameter))
        ]
        print("X_parameter:", X_parameter)
        print("G_parameter:", G_parameter)
        print("infered weight: ", self.infered_weight)
        print("lambdas:", self.lambdas)
        # weights_prime: NDArrays = [
        #     reduce(np.add, layer_updates) for layer_updates in zip(*L_parameter)
        # ]
        print("L_parameter:", L_parameter)
        set_parameters(self.local_model, L_parameter)


class Server:
    def __init__(self, args) -> None:
        self.training_rounds = args.training_rounds
        self.communication_rounds = args.communication_rounds
        self.global_model = RegressionModel(args.n_feats).to(DEVICE)
        self.beta = args.beta  # fair aggregation param
        self.testloader = testloader
        self.args = args

    def multi_vectorization(self, local_params):
        vectors = copy.deepcopy(local_params)

        for i, v in enumerate(vectors):
            for j in range(len(v)):
                v[j] = v[j].reshape([-1])
            vectors[i] = np.concatenate(v)

        return vectors

    def pairwise_distance(self, local_params):

        vectors = self.multi_vectorization(local_params)
        distance = np.zeros([len(vectors), len(vectors)])

        for i, v_i in enumerate(vectors):
            for j, v_j in enumerate(vectors[i:]):
                distance[i][j + i] = distance[j + i][i] = euclid(v_i, v_j)

        return distance

    def trimmed_median(self, local_params, aggragate_w):
        n = len(local_params)
        distance = self.pairwise_distance(local_params)
        distance = np.sum(distance, axis=1)
        med = np.median(distance)
        chosen = np.argsort(abs(distance - med))
        chosen = chosen[:n]
        # FedAvg
        weighted_weights = [
            [layer * aggragate_wk for layer in weights]
            for aggragate_wk, weights in zip(
                [aggragate_w[chosen[i]] for i in range(n - 2)],
                [local_params[chosen[i]] for i in range(n - 2)],
            )
        ]

        w_a: NDArrays = [
            reduce(np.add, layer_updates) for layer_updates in zip(*weighted_weights)
        ]

        return w_a

    def trimmed_mean(self, local_params, aggragate_w):
        n = len(local_params)
        distance = self.pairwise_distance(local_params)
        distance = np.sum(distance, axis=1)
        med = np.mean(distance)
        chosen = np.argsort(np.abs(distance - med))
        chosen = chosen[:n]

        weighted_weights = [
            [layer * aggragate_wk for layer in weights]
            for aggragate_wk, weights in zip(
                [aggragate_w[chosen[i]] for i in range(n - 2)],
                [local_params[chosen[i]] for i in range(n - 2)],
            )
        ]

        w_a: NDArrays = [
            reduce(np.add, layer_updates) for layer_updates in zip(*weighted_weights)
        ]

        return w_a

    def krum(self, local_params, aggragate_w):
        n = len(local_params)

        distance = self.pairwise_distance(local_params)
        sorted_idx = np.argsort(np.sum(distance, axis=0))[:n]

        # chosen_idx = int(sorted_idx[0])
        chosen = sorted_idx

        weighted_weights = [
            [layer * aggragate_wk for layer in weights]
            for aggragate_wk, weights in zip(
                [aggragate_w[chosen[i]] for i in range(n - 2)],
                [local_params[chosen[i]] for i in range(n - 2)],
            )
        ]

        w_a: NDArrays = [
            reduce(np.add, layer_updates) for layer_updates in zip(*weighted_weights)
        ]

        return w_a

        # # FedAvg
        # weighted_weights = [
        #     [layer * aggragate_wk for layer in weights]
        #     for aggragate_wk, weights in zip(
        #         [copy.deepcopy(aggragate_w[chosen_idx])],
        #         [copy.deepcopy(local_params[chosen_idx])],
        #     )
        # ]

        # w_a: NDArrays = [
        #     reduce(np.add, layer_updates) for layer_updates in zip(*weighted_weights)
        # ]

        # return w_a

    def fang(self, local_params, aggragate_w):

        loss_impact = {}
        net_a = RegressionModel(self.args.n_feats).to(DEVICE)
        net_b = copy.deepcopy(net_a)

        for i in range(len(local_params)):
            tmp_w_locals = copy.deepcopy(local_params)
            w_a = self.trimmed_mean(tmp_w_locals, aggragate_w)

            tmp_w_locals.pop(i)
            w_b = self.trimmed_mean(tmp_w_locals, aggragate_w)

            set_parameters(net_a, w_a)
            set_parameters(net_b, w_b)

            loss_a, _, _, _, _ = test(net_a.to(DEVICE), self.testloader)
            loss_b, _, _, _, _ = test(net_b.to(DEVICE), self.testloader)

            loss_impact.update({i: loss_a - loss_b})

        sorted_loss_impact = sorted(loss_impact.items(), key=lambda item: item[1])
        filterd_clients = [
            sorted_loss_impact[i][0] for i in range(len(local_params) - 2)
        ]

        # FedAvg
        weighted_weights = [
            [layer * aggragate_wk for layer in weights]
            for aggragate_wk, weights in zip(
                [copy.deepcopy(aggragate_w[i]) for i in filterd_clients],
                [copy.deepcopy(local_params[i]) for i in filterd_clients],
            )
        ]

        w_c: NDArrays = [
            reduce(np.add, layer_updates) for layer_updates in zip(*weighted_weights)
        ]

        return w_c

    def test_global(self):
        return test(self.global_model, self.testloader)

    def q_aggregate(self, weights_before, Deltas, hs):
        demominator = np.sum(np.asarray(hs))
        num_clients = len(Deltas)
        scaled_deltas = []
        for client_delta in Deltas:
            scaled_deltas.append([layer * 1.0 / demominator for layer in client_delta])

        updates = []
        for i in range(len(Deltas[0])):
            tmp = scaled_deltas[0][i]
            for j in range(1, len(Deltas)):
                tmp += scaled_deltas[j][i]
            updates.append(tmp)

        new_solutions = [(u - v) * 1.0 for u, v in zip(weights_before, updates)]

        return new_solutions

    def fl_run(self):
        attack_list = []
        if args.attack_type == "s":
            attack_list = [0]
        if args.attack_type == "m":
            attack_list = list(range(int(args.client_num * args.attacker_rate)))
        for i in attack_list:
            clients[i].attacker = True
        for t in range(1, self.training_rounds + 1):
            print(f"---------------------{t}-th training round --------------------- ")
            # 一个攻击者，且只开始被选中一次
            if args.single_choose == True:
                print("Single Attack ")
                if t == 1:
                    selected_clients = copy.deepcopy(attack_list)

                    client_ids_new = [
                        client_id
                        for client_id in client_ids
                        if client_id not in attack_list
                    ]

                    selected_clients_append = random.sample(
                        client_ids_new,
                        int(len(client_ids) * args.select_rate) - len(selected_clients),
                    )

                    selected_clients += selected_clients_append

                if not t == 1:
                    client_ids_new = [
                        client_id
                        for client_id in client_ids
                        if client_id not in attack_list
                    ]

                    selected_clients = random.sample(
                        client_ids_new, int(len(client_ids) * args.select_rate)
                    )
            else:
                # if t == 1:
                print("Random Attack ")
                selected_clients = copy.deepcopy(attack_list)

                client_ids_new = [
                    client_id
                    for client_id in client_ids
                    if client_id not in attack_list
                ]

                selected_clients_append = random.sample(
                    client_ids_new,
                    int(len(client_ids) * args.select_rate) - len(selected_clients),
                )

                selected_clients += selected_clients_append
                # if not t == 1:
                #     selected_clients = random.sample(
                #         client_ids, int(len(client_ids) * args.select_rate)
                #     )

            """initialize aggregate weight"""
            aggragate_w = [clients[i].get_datasize() for i in selected_clients]
            aggragate_w = [w / sum(aggragate_w) for w in aggragate_w]

            for c in range(1, self.communication_rounds + 1):
                """update local model"""
                for i in selected_clients:
                    clients[i].receive_param(get_parameters(self.global_model))
                """test global model performance"""
                _, global_accuracy, global_bias, _, _ = self.test_global()
                print(
                    f"{c-1}-th communication --- global accuracy:{global_accuracy}, global bias:{global_bias}, aggregation weight: {aggragate_w} "
                )
                """local train"""
                local_bias = []
                delta_list = []
                local_loss = []
                for i in selected_clients:
                    clients[i].train(c, self.communication_rounds)
                """local eveluate"""
                for i in range(len(selected_clients)):
                    loss, _, bias = clients[selected_clients[i]].test(
                        c, self.communication_rounds
                    )
                    local_loss.append(loss)
                    local_bias.append(bias)
                    delta_list.append(abs(global_bias - local_bias[i]))  # abs?
                """update aggregate weight"""
                if args.aggragation == "fairfed" or args.aggragation == "fairavg":
                    aggragate_w = [
                        max(
                            aggragate_w[i]
                            - self.beta
                            * (delta_list[i] - sum(delta_list) / len(delta_list)),
                            0,
                        )
                        for i in range(len(aggragate_w))
                    ]

                if args.aggragation == "qfed":
                    # q-Fed
                    aggragate_w = [
                        max(
                            aggragate_w[i]
                            * pow(1 - abs(local_bias[i]), args.q)
                            / (args.q + 1),
                            0,
                        )
                        for i in range(len(aggragate_w))
                    ]
                    # all_client_grads = []
                    # all_client_Deltas = []
                    # all_hs = []
                    # local_params = [clients[i].send_param() for i in selected_clients]
                    # for i in range(len(local_params)):
                    #     weights_before = copy.deepcopy(
                    #         get_parameters(self.global_model)
                    #     )
                    #     new_weights = copy.deepcopy(local_params[i])
                    #     grads = [
                    #         (u - v) * 1.0 / 1e-2
                    #         for u, v in zip(weights_before, new_weights)
                    #     ]
                    #     Deltas = [
                    #         np.float_power(local_loss[i] + 1e-10, args.q) * grad
                    #         for grad in grads
                    #     ]
                    #     hs = args.q * np.float_power(
                    #         local_loss[i] + 1e-10, (args.q - 1)
                    #     ) * norm_grad(grads) + (1.0 / 1e-2) * np.float_power(
                    #         local_loss[i] + 1e-10, args.q
                    #     )
                    #     all_client_grads.append(grads)
                    #     all_client_Deltas.append(Deltas)
                    #     all_hs.append(hs)

                aggragate_w = [w / sum(aggragate_w) for w in aggragate_w]
                """update global model"""
                local_params = [clients[i].send_param() for i in selected_clients]

                weighted_weights = [
                    [layer * aggragate_wk for layer in weights]
                    for aggragate_wk, weights in zip(aggragate_w, local_params)
                ]

                weights_prime: NDArrays = [
                    reduce(np.add, layer_updates)
                    for layer_updates in zip(*weighted_weights)
                ]

                # add defense -trimmed mean, median, Krum, Bulyan, norm clipping.
                if args.defense == "trimmed_mean":
                    weights_prime = self.trimmed_mean(local_params, aggragate_w)
                if args.defense == "trimmed_median":
                    weights_prime = self.trimmed_median(local_params, aggragate_w)
                if args.defense == "krum":
                    weights_prime = self.krum(local_params, aggragate_w)
                if args.defense == "fang":
                    weights_prime = self.fang(local_params, aggragate_w)

                print("weights_prime:", weights_prime)
                set_parameters(self.global_model, weights_prime)
            _, global_accuracy, global_bias, _, _ = self.test_global()
            print(
                f"{c}-th communication --- global accuracy:{global_accuracy}, global bias:{global_bias}, aggregation weight: {aggragate_w} "
            )


""" global visible """
client_ids = list(range(args.client_num))
attack_ids = random.sample(client_ids, int(len(client_ids) * args.attacker_rate))
benign_ids = [id for id in client_ids if id not in attack_ids]
clients = []
if __name__ == "__main__":
    setup_seed(args.seed)
    server = Server(args)

    """ initialize clients """
    for i in client_ids:
        client = Client(args, i)
        clients.append(client)

    server.fl_run()
    print("end")
