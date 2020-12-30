# -*- coding: utf-8 -*-
"""
@Time   : 2020/9/27

@Author : Shen Fang
"""
import os
import os.path as osp

import torch
import numpy as np

from torch.utils.data import Dataset


def read_one_day(file_name):
    one_day_data = np.load(file_name)
    return one_day_data


def split_train_valid_test(folder, train_ratio, valid_ratio):
    name_list = sorted(os.listdir(folder))
    name_list = list(filter(lambda x: x.split(".")[1] == "npy", name_list))
    name_list = list(map(lambda x: osp.join(folder, x), name_list))

    train_idx = int(len(name_list) * train_ratio)
    valid_idx = int(len(name_list) * valid_ratio)

    return name_list[: train_idx], name_list[train_idx: train_idx + valid_idx], name_list[train_idx + valid_idx: ]


class TrafficData(Dataset):
    def __init__(self, folder, train_ratio, valid_ratio, data_type, h_step, f_step):
        train_files, valid_files, test_files = split_train_valid_test(folder, train_ratio, valid_ratio)
        if data_type == "train":
            self.files = train_files
        elif data_type == "valid":
            self.files = valid_files
        elif data_type == "test":
            self.files = test_files
        else:
            raise KeyError("data type is not correct")
        self.h_step = h_step
        self.f_step = f_step

        sample_file = self.files[0]
        sample_data = np.load(sample_file)

        self.num_nodes = sample_data.shape[0]
        self.day_length = sample_data.shape[1] - self.h_step - self.f_step

    def __getitem__(self, index):
        file_idx = index // self.day_length
        file = self.files[file_idx]
        data_idx = index % self.day_length
        data = np.load(file)
        x = data[:, data_idx: data_idx + self.h_step]
        y = data[:, data_idx + self.h_step: data_idx + self.h_step + self.f_step]

        return {"x": TrafficData.to_tensor(x),
                "y": TrafficData.to_tensor(y)}

    def __len__(self):
        return self.day_length * len(self.files)

    @staticmethod
    def to_tensor(data):
        return torch.LongTensor(data)


if __name__ == '__main__':
    files = sorted(os.listdir("data"))
    files = list(filter(lambda x: x.split(".")[1] == "npy", files))

    files = list(map(lambda x: osp.join("data", x), files))

    for file in files:
        data = np.load(file)
        print(file, data.shape)
    # train_set = TrafficData(folder="data", train_ratio=0.6, valid_ratio=0.2, data_type="train", h_step=12, f_step=1)
    # for data in train_set:
    #     print(data["x"].size(), data["y"].size())
