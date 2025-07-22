from folktables import (
    ACSDataSource,
    ACSPublicCoverage,
    ACSEmployment,
    ACSIncome,
    ACSMobility,
    ACSTravelTime,
    ACSHealthInsurance,
)
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
import torch
from torch.utils.data import Subset
import bisect

state_list = [
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
    "PR",
]


class Federated_ConcatDataset(Dataset):
    def __init__(self, datasets):
        self.X = torch.cat([d.X for d in datasets], dim=0)
        self.Y = torch.cat([d.Y for d in datasets], dim=0)
        self.A = torch.cat([d.A for d in datasets], dim=0)

    def __getitem__(self, index):
        X = self.X[index]
        Y = self.Y[index]
        A = self.A[index]
        return X, Y, A

    def __len__(self):
        return self.X.shape[0]


class Federated_Subset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.X = dataset.X[indices]
        self.Y = dataset.Y[indices]
        self.A = dataset.A[indices]


class Federated_Dataset(Dataset):
    def __init__(self, X, Y, A):
        self.X = torch.tensor(X)
        self.Y = torch.tensor(Y)
        self.A = torch.tensor(A)

    def __getitem__(self, index):
        X = self.X[index]
        Y = self.Y[index]
        A = self.A[index]
        return X, Y, A

    def __len__(self):
        return self.X.shape[0]


# def load_datasets(
#     BATCH_SIZE, REDUCE, survey_year, horizon, survey, root_dir, sample_rate
# ):
def load_datasets(function, REDUCE, sample_rate):
    train_datasets = []
    val_datasets = []

    data_source = ACSDataSource(
        survey_year=2022, horizon="1-Year", survey="person", root_dir="dataset/acs"
    )
    scaler = StandardScaler()
    sum = 0
    for state in state_list:
        acs_data14 = data_source.get_data(states=[state], download=False)

        if len(acs_data14) > 20000:
            acs_data14 = acs_data14.sample(frac=sample_rate)

        if function == "ACSPublicCoverage":
            features14, labels14, group = ACSPublicCoverage.df_to_numpy(acs_data14)
        if function == "ACSIncome":
            features14, labels14, group = ACSIncome.df_to_numpy(acs_data14)
        if function == "ACSEmployment":
            features14, labels14, group = ACSEmployment.df_to_numpy(acs_data14)
        if function == "ACSMobility":
            features14, labels14, group = ACSMobility.df_to_numpy(acs_data14)
        if function == "ACSTravelTime":
            features14, labels14, group = ACSTravelTime.df_to_numpy(acs_data14)
        if function == "ACSHealthInsurance":
            features14, labels14, group = ACSHealthInsurance.df_to_numpy(acs_data14)
        # remove group !=0 and 1
        mask = np.logical_or(group == 1, group == 2)
        features14 = features14[mask]
        # sum += features14.shape[0]
        features14 = scaler.fit_transform(features14)

        labels14 = labels14[mask]
        group = group[mask]

        # split to train/test

        (
            train_features14,
            test_features14,
            train_labels14,
            test_labels14,
            group_train,
            group_test,
        ) = train_test_split(features14, labels14, group, test_size=0.2, random_state=0)

        # train_features14 = transforms.ToTensor()(train_features14)
        # test_features14 = transforms.ToTensor()(test_features14)
        # train_labels14 = transforms.ToTensor()(train_labels14)
        # test_labels14 = transforms.ToTensor()(test_labels14)
        # group_train = transforms.ToTensor()(group_train)
        # group_test = transforms.ToTensor()(group_test)
        if REDUCE:
            train_dataset = Federated_Dataset(
                train_features14[0 : train_features14.shape[0] // 10],
                train_labels14[0 : train_labels14.shape[0] // 10],
                group_train[0 : group_train.shape[0] // 10],
            )
        else:
            train_dataset = Federated_Dataset(
                train_features14, train_labels14, group_train
            )
        # trainloaders.append(
        #     DataLoader(
        #         train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
        #     )
        # )
        train_datasets.append(train_dataset)

        if REDUCE:
            val_dataset = Federated_Dataset(
                test_features14[0 : test_features14.shape[0] // 10],
                test_labels14[0 : test_labels14.shape[0] // 10],
                group_test[0 : group_test.shape[0] // 10],
            )
        else:
            val_dataset = Federated_Dataset(test_features14, test_labels14, group_test)
        # valloaders.append(
        #     DataLoader(
        #         val_dataset,
        #         batch_size=test_features14.shape[0],
        #         shuffle=True,
        #         num_workers=4,
        #     )
        # )
        val_datasets.append(val_dataset)

    return None, None, train_datasets, val_datasets
