# -*- coding: utf-8 -*-
"""
@Time   : 2020/10/19

@Author : Shen Fang
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
# from transform_dataset import TrafficData


class OneHotProcess(nn.Module):
    def __init__(self, in_dim, hid_c):
        super(OneHotProcess, self).__init__()
        self.embedding = nn.Embedding(in_dim, hid_c)

    def forward(self, source):
        source = source // 20 - 1
        source = self.embedding(source)

        return source

class LinearRegression(nn.Module):
    def __init__(self, in_dim, hid_c, src_len):
        super(LinearRegression, self).__init__()
        self.oneHotEmbed = OneHotProcess(in_dim, hid_c)
        self.linear = nn.Linear(src_len * hid_c, in_dim)

    def forward(self, input_data, device):
        source = input_data.to(device)  # [B, N, src_len]

        input_feature = self.oneHotEmbed(source)  # [B, N, src_len, hid_dim]

        B, N, src_len, hid_c = input_feature.size()

        input_feature = input_feature.view(B, N, -1)  # [B, N, src_len * hid_dim]

        out_feature = F.relu(self.linear(input_feature)) # [B, N, in_dim]

        predict = F.softmax(out_feature, dim=-1)

        return predict # [B, N, in_dim]


def test(test_data):
    """
    test_data: [B, N, src_len]  20, 40, 60
    prediction: [B, N, in_dim]
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_name = "lr.pkl"

    checkpoint = torch.load(file_name, device)
    model_para = checkpoint["model"]
    option = checkpoint["setting"]

    cudnn.benchmark = True

    model = LinearRegression(3, option.hid_c, option.h_step)

    model.load_state_dict(model_para)

    model = model.to(device)

    prediction = model(test_data, device)

    return prediction


# if __name__ == '__main__':
#     test_set = TrafficData(folder="data", train_ratio=0.6, valid_ratio=0.2, data_type="test", h_step=12, f_step=1)
#     test_data = torch.cat((test_set[0]["x"].unsqueeze(0), test_set[1]["x"].unsqueeze(0)), dim=0)
#     prediction = test(test_data)
#     print(prediction.size())