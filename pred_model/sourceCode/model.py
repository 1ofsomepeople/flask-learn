# -*- coding: utf-8 -*-
"""
@Time   : 2020/9/30

@Author : Shen Fang
"""

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from graph import GraphModel, SGModel, SAGEModel


class OneHotProcess(nn.Module):
    def __init__(self, in_dim, hid_c):
        super(OneHotProcess, self).__init__()
        self.embedding = nn.Embedding(in_dim, hid_c)

    def forward(self, source, target):
        source = source // 20 - 1
        target = target // 20 - 1
        source = self.embedding(source)

        return source, target


class PredictModel(nn.Module):
    def __init__(self, keyword, src_id, dst_id, in_dim, hid_c, src_len, n_layers, device):
        super(PredictModel, self).__init__()

        self.oneHotEmbed = OneHotProcess(in_dim, hid_c)

        if keyword == "Graph":
            self.model = GraphModel(src_id, dst_id, src_len * hid_c, src_len *  hid_c, device)
        elif keyword == "SG":
            self.model = SGModel(src_id, dst_id, src_len * hid_c, src_len * hid_c, n_layers, device)
        elif keyword == "SAGE":
            self.model = SAGEModel(src_id, dst_id, src_len * hid_c, src_len * hid_c, n_layers, device)
        else:
            raise KeyError("Keyword is not defined! ")

        self.linear = nn.Linear(src_len * hid_c, in_dim)

    def forward(self, input_data, device, **kwargs):
        source = input_data["x"].to(device)  # [B, N, src_len]
        target = input_data["y"].to(device)[:, :, 0]  # [B, N]
        # target = input_data["y"].to(device)  # [B, N]

        input_feature, target = self.oneHotEmbed(source, target)

        B, N, src_len, hid_c = input_feature.size()

        input_feature = input_feature.view(B, N, -1).permute(1, 0, 2)  # [N, B, src_len * hid_c]

        output_feature = self.model(input_feature)  # [N, B, hid_c]

        output_feature = self.linear(output_feature)  # [N, B, in_dim]

        predict = F.softmax(output_feature, dim=-1).permute(1, 0, 2)  # [B, N, in_dim]

        predict = predict.reshape(B * N, -1)

        target = target.reshape(-1)

        return predict, target


class LinearRegression(nn.Module):
    def __init__(self, in_dim, hid_c, src_len):
        super(LinearRegression, self).__init__()
        self.oneHotEmbed = OneHotProcess(in_dim, hid_c)
        self.linear = nn.Linear(src_len * hid_c, in_dim)

    def forward(self, input_data, device, **kwargs):
        source = input_data["x"].to(device)  # [B, N, src_len]
        target = input_data["y"].to(device)[:, :, 0]  # [B, N]

        input_feature, target = self.oneHotEmbed(source, target)  # [B, N, src_len, hid_dim]

        B, N, src_len, hid_c = input_feature.size()

        input_feature = input_feature.view(B, N, -1)  # [B, N, src_len * hid_dim]

        out_feature = F.relu(self.linear(input_feature)) # [B, N, in_dim]

        predict = F.softmax(out_feature, dim=-1)

        predict = predict.reshape(B * N, -1)

        target = target.reshape(-1)

        return predict, target


class NormalizeLR(nn.Module):
    def __init__(self, in_dim, hid_c, src_len):
        super(NormalizeLR, self).__init__()
        self.oneHotEmbed = OneHotProcess(in_dim, hid_c)
        self.linear = nn.Linear(src_len * hid_c, in_dim)
        self.norm = nn.BatchNorm1d(in_dim)

    def forward(self, input_data, device, **kwargs):
        source = input_data["x"].to(device)  # [B, N, src_len]
        target = input_data["y"].to(device)[:, :, 0]  # [B, N]

        input_feature, target = self.oneHotEmbed(source, target)  # [B, N, src_len, hid_dim]

        B, N, src_len, hid_c = input_feature.size()

        input_feature = input_feature.view(B, N, -1)  # [B, N, src_len * hid_dim]

        out_feature = self.norm(self.linear(input_feature).permute(0, 2, 1))  # [B, N, in_dim]

        out_feature = F.relu(out_feature.permute(0, 1, 2))

        predict = F.softmax(out_feature, dim=-1)

        predict = predict.reshape(B * N, -1)

        target = target.reshape(-1)

        return predict, target


if __name__ == '__main__':
    input_data = {"x": torch.LongTensor([[20, 40, 60, 20, 20, 40],
                                         [20, 40, 60, 20, 20, 40],
                                         [20, 40, 60, 20, 20, 40]]).unsqueeze(0),  # [1, 3, 6]  B, N, T
                  "y": torch.LongTensor([[20, 40, 60, 20, 20, 40],
                                         [20, 40, 60, 20, 20, 40],
                                         [20, 40, 60, 20, 20, 40]]).unsqueeze(0)}  # [1, 3, 6]  B, N, T
    # print(input_data)
    model = PredictModel(keyword="SAGE",
                         src_id=[0, 1, 2], dst_id=[1, 2, 0],
                         in_dim=3, hid_c=12, src_len=6, n_layers=2, device=torch.device("cpu"))

    result = model(input_data, torch.device("cpu"))

    print(result[0].size())
    print(result[1].size())
