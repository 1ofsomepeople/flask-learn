# -*- coding: utf-8 -*-
"""
@Time   : 2020/10/9

@Author : Shen Fang
"""
import os
import time
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR


def train_epoch(model: nn.Module, train_loader: DataLoader, criterion: nn.CrossEntropyLoss, optimizer: torch.optim.Adam,
                device: torch.device, ration):
    epoch_loss = 0.0

    model.train()

    for data in train_loader:
        optimizer.zero_grad()

        prediction, target = model(data, device=device, ration=ration)

        loss = criterion(prediction, target)
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(train_loader.dataset)


def eval_epoch(model, valid_loader, criterion, device):
    epoch_loss = 0.0

    model.eval()

    with torch.no_grad():
        for data in valid_loader:

            prediction, target = model(data, device=device, ration=0)

            loss = criterion(prediction, target)

            epoch_loss += loss.item()

    return epoch_loss / len(valid_loader.dataset)


def train(model: nn.Module, train_loader: DataLoader, valid_loader: DataLoader, criterion: nn.MSELoss(),
          optimizer: torch.optim.Adam, scheduler: MultiStepLR, option, device: torch.device):
    train_file = None
    valid_file = None

    if option.log:
        train_file = option.log + "_train.csv"
        valid_file = option.log + "_valid.csv"

        print("[ INFO ] Training information will be written.")

    best_valid_loss = float("inf")
    ration = 1

    for each_epoch in range(option.epoch):
        if (each_epoch + 1) % 5 == 0:
            ration *= 0.9

        print("[ Epoch {:d}]".format(each_epoch))

        # Train One Epoch
        start_time = time.time()

        train_loss = train_epoch(model=model, train_loader=train_loader, criterion=criterion, optimizer=optimizer,
                                 device=device, ration=ration)

        train_minute, train_second = epoch_time(start_time, time.time())
        print("  - (Training)   loss: {:2.4f}, time: {:2d} min {:2d} sec ".format(train_loss, train_minute, train_second))

        scheduler.step()

        # Validate One Epoch
        start_time = time.time()

        valid_loss = eval_epoch(model=model, valid_loader=valid_loader, criterion=criterion, device=device)

        valid_minute, valid_second = epoch_time(start_time, time.time())

        print("  - (Validation) loss: {:2.4f}, time: {:2d} min {:2d} sec ".format(valid_loss, valid_minute, valid_second))

        # Save Model
        model_state_dict = model.state_dict()

        checkpoint = {"model": model_state_dict,
                      "setting": option,
                      "epoch": each_epoch}

        if option.save_model:
            if option.save_mode == "best":
                model_name = option.log + ".pkl"
                if valid_loss <= best_valid_loss:
                    best_valid_loss = valid_loss

                    torch.save(checkpoint, model_name)
                    print("  - [ INFO ] The checkpoint file is updated.")

            elif option.save_mode == "all":
                model_name = option.log + "_epoch_{:2d}.pkl".format(each_epoch)
                torch.save(checkpoint, model_name)

        if train_file and valid_file:
            with open(train_file, "a") as train_obj, open(valid_file, "a") as valid_obj:
                train_obj.write("{:2d}, {:2.4f}, {:2d} min {:2d} sec \n".format(each_epoch, train_loss, train_minute, train_second))
                valid_obj.write("{:2d}, {:2.4f}, {:2d} min {:2d} sec \n".format(each_epoch, valid_loss, valid_minute, valid_second))


def test(model: nn.Module, test_loader: DataLoader, criterion: nn.MSELoss(), option, device: torch.device):
    test_loss = 0.0

    data_to_save = {"predict": np.zeros(shape=[1, option.num_nodes, option.f_step, 3]),  # [B, N, TRG_len, C]
                    "target": np.zeros(shape=[1, option.num_nodes, option.f_step, 3])}   # [B, N, TRG_len, C]

    model.eval()

    with torch.no_grad():
        for data in test_loader:
            prediction, target = model(data, device=device, ration=0)
            # prediction : [B * N, in_dim]
            # target:  [B * N]

            loss = criterion(prediction, target)

            test_loss += loss.item()

            recovered = recover_data(prediction, target, option)

            data_to_save["predict"] = np.concatenate([data_to_save["predict"], recovered["predict"]], axis=0)
            data_to_save["target"] = np.concatenate([data_to_save["target"], recovered["target"]], axis=0)

    # data_to_save["predict"] = np.delete(data_to_save["predict"], 0, axis=0)
    # data_to_save["target"] = np.delete(data_to_save["target"], 0, axis=0)

    acc = compute_performance(data_to_save["predict"], data_to_save["target"])

    if option.log:
        test_file = option.log + "_test.csv"
        with open(test_file, "a") as test_obj:
            test_obj.write("Acc:  ")
            for item in acc:
                test_obj.write("  {:2.4f}".format(item))
            test_obj.write("\n")

        result_file = option.log + "_result.npz"

        for i in range(option.f_step):
            np.savez(result_file, {"predict_time_{:d}".format(i): data_to_save["predict"][:, :, i].transpose([1, 0, 2]),
                                   "target_time_{:d}".format(i): data_to_save["target"][:, :, i].transpose([1, 0, 2])})

    return np.mean(acc), test_loss / (len(test_loader.dataset))


def recover_data(prediction, target, option):
    num_nodes = option.num_nodes
    batch_size = target.size(0) // num_nodes

    predict = prediction.view(batch_size, num_nodes, option.f_step, -1)

    target = target.view(batch_size, num_nodes, option.f_step)
    target = F.one_hot(target, 3)

    return {"predict": predict.to(torch.device("cpu")).numpy(),
            "target": target.to(torch.device("cpu")).numpy()}


def compute_performance(predict, target):
    # [B, N, T, C]
    predict = np.argmax(predict, axis=-1)  # [B, N, T]
    target = np.argmax(target, axis=-1)  # [B, N, T]

    predict = np.reshape(predict, -1)  # []
    target = np.reshape(target, -1)  # []

    correct = float((predict == target).sum())
    total = float(target.size)

    result = [correct / total]

    classes = list(np.unique(target))  # [0, 1, 2]


    for cls in classes:
        indices = np.argwhere(target == cls)
        correct_i = float((target[indices] == predict[indices]).sum())
        total_i = float(target[indices].size)

        result.append(correct_i / total_i)

    return result


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.1)
        else:
            nn.init.constant_(param.data, 0.01)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def load_graph():
    graph_data = np.load("graph.npz")
    src_id = graph_data["src_id"]
    dst_id = graph_data["dst_id"]

    return list(src_id), list(dst_id)


def create_graph(threshold):
    data = np.load("image_coord.npy")
    N = data.shape[0]
    src_id, dst_id = [], []
    for i in range(N):
        print(i)
        for j in range(N):
            norm = np.linalg.norm(data[i] - data[j])
            if norm <= threshold:
                src_id.append(i)
                dst_id.append(j)

    src_id = np.array(src_id)
    dst_id = np.array(dst_id)

    np.savez("graph.npz", src_id=src_id, dst_id=dst_id)

    return src_id, dst_id


class OptionClass:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Traffic congestion prediction.")

        # dataset args
        parser.add_argument("--data_folder", type=str)
        parser.add_argument("--h_step", type=int)
        parser.add_argument("--f_step", type=int)

        # model args
        parser.add_argument("--save_model", type=bool, default=True)
        parser.add_argument("--save_mode", type=str, default="best")
        parser.add_argument("--epoch", type=int)
        parser.add_argument("--log", type=str)

        # result args
        parser.add_argument("--result_folder", type=str)

        args = parser.parse_args()

        self.args = args

    def add_result_folder(self, folder):
        self.args.result_folder = folder

    def add_log_name(self, log):
        self.args.log = log

    def initialize(self):
        if not os.path.exists(self.args.result_folder):
            os.mkdir(self.args.result_folder)

        self.args.log = os.path.join(self.args.result_folder, self.args.log)

        return self.args


if __name__ == '__main__':
    # graph = create_graph(12)
    src_id, dst_id = load_graph()
