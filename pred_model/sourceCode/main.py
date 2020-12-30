# -*- coding: utf-8 -*-
"""
@Time   : 2020/9/30

@Author : Shen Fang
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from transform_dataset import TrafficData
from torch.optim.lr_scheduler import MultiStepLR
from train import train, test, count_parameters, init_weights, load_graph

import argparse

from model import PredictModel, LinearRegression, NormalizeLR


def train_main(model, option, test_flag):

    print("[  INFO  ] Loading Training Data")
    train_set = TrafficData(folder="data", train_ratio=0.6, valid_ratio=0.2, data_type="train",
                            h_step=option.h_step, f_step=option.f_step)

    train_loader = DataLoader(train_set, batch_size=option.batch_size, shuffle=True, num_workers=32)

    valid_set = TrafficData(folder="data", train_ratio=0.6, valid_ratio=0.2, data_type="valid",
                            h_step=option.h_step, f_step=option.f_step)

    valid_loader = DataLoader(valid_set, batch_size=option.batch_size, shuffle=False, num_workers=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[ INFO ] Load Model")

    model = model.to(device)

    print("  - Number of parameters: ", count_parameters(model))

    model.apply(init_weights)

    model_structure = model

    print("  - {} GPU(s) will be used.".format(torch.cuda.device_count()))

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss(reduction="sum")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = MultiStepLR(optimizer, [100], gamma=1)

    train(model, train_loader, valid_loader, criterion, optimizer, scheduler, option, device)

    if test_flag:
        test_main(option.log, model_structure)


def test_main(model_file_name, model_structure):
    checkpoint = torch.load(model_file_name + ".pkl")

    model_state_dict = checkpoint["model"]
    option = checkpoint["setting"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cudnn.benchmark = True

    print("[ INFO ] Load Testing Data")

    test_set = TrafficData(folder="data", train_ratio=0.6, valid_ratio=0.2, data_type="test",
                           h_step=option.h_step, f_step=option.f_step)

    test_loader = DataLoader(test_set, batch_size=option.batch_size, shuffle=False, num_workers=32)

    print("  - {} GPU(s) will be used.".format(torch.cuda.device_count()))

    if torch.cuda.device_count() > 1:
        model_structure = nn.DataParallel(model_structure)

    model_structure.load_state_dict(model_state_dict)
    model_structure = model_structure.to(device)

    criterion = nn.CrossEntropyLoss(reduction="sum")

    acc, loss = test(model_structure, test_loader, criterion, option, device)

    print("\nTest Data:")
    print("[ Results ]\n  - loss: {:2.4f}  ,  acc: {:2.2f}".format(loss, acc[0]))

    for i in range(1, len(acc)):
        print("  class [{:02d}]: {:2.2f}".format(i, acc[i]))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Traffic congestion prediction.")

    # dataset args
    parser.add_argument("--data_folder", type=str, default="data")
    parser.add_argument("--h_step", type=int)
    parser.add_argument("--f_step", type=int)

    # model args
    parser.add_argument("--model", type=str, default="SAGE")
    parser.add_argument("--save_model", type=bool, default=True)
    parser.add_argument("--save_mode", type=str, default="best")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--log", type=str)

    parser.add_argument("--hid_c", type=int, default=12)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--num_nodes", type=int, default=17531)

    # result args
    parser.add_argument("--result_folder", type=str)
    parser.add_argument("--gpu", type=str, default="0")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if not os.path.exists(args.result_folder):
        os.mkdir(args.result_folder)

    args.log = os.path.join(args.result_folder, args.log)

    src_id, dst_id = load_graph()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "lr":
        model = LinearRegression(3, args.hid_c, args.h_step)
    elif args.model == "norm_lr":
        model = NormalizeLR(3, args.hid_c, args.h_step)
    else:
        model = PredictModel(args.model, src_id, dst_id, 3, args.hid_c, args.h_step, args.n_layer, device)

    train_main(model, args, True)
    # test_main(args.log, model)
