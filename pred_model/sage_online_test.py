# -*- coding: utf-8 -*-
"""
@Time   : 2020/10/19

@Author : Shen Fang
"""
import dgl
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import dgl.nn as gnn
# from transform_dataset import TrafficData
import random
import pandas as pd
import os

def load_graph():

    file_name = "graph.npz"
    path=os.path.abspath('.')   #表示执行环境的绝对路径
    if(os.path.split(path)[-1] == 'pred_model'):
        file_name = os.path.join(path,'graph.npz')
    elif(os.path.split(path)[-1] == 'flask-learn'):
        file_name = os.path.join(path,'pred_model','graph.npz')

    graph_data = np.load(file_name)
    src_id = graph_data["src_id"]
    dst_id = graph_data["dst_id"]

    return list(src_id), list(dst_id)


class OneHotProcess(nn.Module):
    def __init__(self, in_dim, hid_c):
        super(OneHotProcess, self).__init__()
        self.embedding = nn.Embedding(in_dim, hid_c)

    def forward(self, source):
        source = source // 20 - 1
        source = self.embedding(source)

        return source


class SAGEModel(nn.Module):
    def __init__(self, src_id, dst_id, in_c, hid_c, n_layers, device):
        super(SAGEModel, self).__init__()
        self.graph = dgl.graph((src_id, dst_id), device=device)

        self.gcn = nn.ModuleList([gnn.SAGEConv(in_c if i == 0 else hid_c, hid_c, "pool") for i in range(n_layers)])

        self.residual = nn.ModuleList([nn.Identity() if i != 0 else nn.Linear(in_c, hid_c) for i in range(n_layers)])

    def forward(self, features):
        input_features = features
        for i, conv in enumerate(self.gcn):
            output_features = F.relu(conv(self.graph, input_features)) + self.residual[i](input_features)
            input_features = output_features

        return input_features


class PredictModel(nn.Module):
    def __init__(self, keyword, src_id, dst_id, in_dim, hid_c, src_len, n_layers, device):
        super(PredictModel, self).__init__()

        self.oneHotEmbed = OneHotProcess(in_dim, hid_c)

        if keyword == "SAGE":
            self.model = SAGEModel(src_id, dst_id, src_len * hid_c, src_len * hid_c, n_layers, device)
        else:
            raise KeyError("Keyword is not defined! ")

        self.linear = nn.Linear(src_len * hid_c, in_dim)

    def forward(self, input_data, device):
        source = input_data.to(device)  # [B, N, src_len]

        input_feature = self.oneHotEmbed(source)

        B, N, src_len, hid_c = input_feature.size()

        input_feature = input_feature.view(B, N, -1).permute(1, 0, 2)  # [N, B, src_len * hid_c]

        output_feature = self.model(input_feature)  # [N, B, hid_c]

        output_feature = self.linear(output_feature)  # [N, B, in_dim]

        predict = F.softmax(output_feature, dim=-1).permute(1, 0, 2)  # [B, N, in_dim]

        return predict


def test(test_data):
    """
    test_data: [B, N, src_len]  20, 40, 60
    prediction: [B, N, in_dim]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    file_name = "sage.pkl"
    path=os.path.abspath('.')   #表示执行环境的绝对路径
    if(os.path.split(path)[-1] == 'pred_model'):
        file_name = os.path.join(path,'sage.pkl')
    elif(os.path.split(path)[-1] == 'flask-learn'):
        file_name = os.path.join(path,'pred_model','sage.pkl')

    checkpoint = torch.load(file_name, map_location=device)
    model_para = checkpoint["model"]
    option = checkpoint["setting"]

    cudnn.benchmark = True

    src_id, dst_id = load_graph()

    model = PredictModel(option.model, src_id, dst_id, 3, option.hid_c, option.h_step, option.n_layer, device)

    model.load_state_dict(model_para)

    model = model.to(device)

    prediction = model(test_data, device)

    return prediction

def mockData():
    # 生成17531*12的二维数组
    timeData = [[random.randrange(20,61,20) for col in range(12)] for row in range(17531)]
    tensorData = torch.Tensor(timeData).unsqueeze(0).long()
    # print(tensorData.size())
    # print(tensorData.type())
    return tensorData

# if __name__ == '__main__':
#     test_set = TrafficData(folder="data", train_ratio=0.6, valid_ratio=0.2, data_type="test", h_step=12, f_step=1)
#     test_data = torch.cat((test_set[0]["x"].unsqueeze(0), test_set[1]["x"].unsqueeze(0)), dim=0)
#     prediction = test(test_data)
#     print(prediction.size())

# if __name__ == '__main__':
#     tensorData = mockData()
#     prediction = test(tensorData)
#     print(prediction.size())
#     resultIndexList = torch.max(prediction[0],1)[1].numpy().tolist()
#     for i in range(len(resultIndexList)):
#         resultIndexList[i] = 20 + resultIndexList[i]*20
#     print(resultIndexList)