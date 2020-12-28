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
import random
import numpy as np
import pandas as pd
import os

class OneHotProcess(nn.Module):
    def __init__(self, in_dim, hid_c):
        super(OneHotProcess, self).__init__()
        self.embedding = nn.Embedding(in_dim, hid_c) # 生成一个固定维度的矩阵

    def forward(self, source):
        source = source // 20 - 1 # onehot标签 （0，0，0）  （0，1，0） 普通标签：0 1 2
        source = self.embedding(source) 

        return source

class LinearRegression(nn.Module):
    def __init__(self, in_dim, hid_c, src_len):
        super(LinearRegression, self).__init__()
        self.oneHotEmbed = OneHotProcess(in_dim, hid_c) #in_dim 输入维度  hid ：hidden
        self.linear = nn.Linear(src_len * hid_c, in_dim)

    def forward(self, input_data, device):
        source = input_data.to(device)  # [B, N, src_len]

        input_feature = self.oneHotEmbed(source)  # [B, N, src_len, hid_dim]

        B, N, src_len, hid_c = input_feature.size()

        input_feature = input_feature.view(B, N, -1)  # [B, N, src_len * hid_dim]  # resize

        out_feature = F.relu(self.linear(input_feature)) # [B, N, in_dim] #线性回归+激活函数

        predict = F.softmax(out_feature, dim=-1)

        return predict # [B, N, in_dim]


def test(test_data):
    """
    test_data: [B, N, src_len]  20, 40, 60
    prediction: [B, N, in_dim]
    """
    # 用CPU或者GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_name = "lr.pkl"
    path=os.path.abspath('.')   #表示执行环境的绝对路径
    if(os.path.split(path)[-1] == 'pred_model'):
        file_name = os.path.join(path,'lr.pkl')
    elif(os.path.split(path)[-1] == 'flask-learn'):
        file_name = os.path.join(path,'pred_model','lr.pkl')

    checkpoint = torch.load(file_name, device) # .pth   load权重，类型：dic
    print(checkpoint)
    model_para = checkpoint["model"] # load权重的参赛
    option = checkpoint["setting"] # 某些参数设置

    cudnn.benchmark = True  # 无用设置

    model = LinearRegression(3, option.hid_c, option.h_step) # 模型实例

    model.load_state_dict(model_para) # load权重

    model = model.to(device) # 模型放到CPU或者GPU

    prediction = model(test_data, device) # 得到预测

    return prediction


# if __name__ == '__main__':
#     test_set = TrafficData(folder="data", train_ratio=0.6, valid_ratio=0.2, data_type="test", h_step=12, f_step=1)
#     test_data = torch.cat((test_set[0]["x"].unsqueeze(0), test_set[1]["x"].unsqueeze(0)), dim=0)
#     prediction = test(test_data)
#     print(prediction.size())

def mockData():
    # 生成17531*12的二维数组
    timeData = [[random.randrange(20,61,20) for col in range(12)] for row in range(17531)]
    tensorData = torch.Tensor(timeData).unsqueeze(0).long()
    print(tensorData.size())
    print(tensorData.type())
    return tensorData

# 加载用于模型预测的数据
def loadDataForPred():
    # 读csv，生成dataFrame
    readCsv = pd.read_csv('../data.csv')
    # 经纬度的pointindex数组
    pointsIndex = np.array(readCsv.values)[:,0].tolist()

    dataForPred = np.array(readCsv.values)[:,-12:].tolist()
    tensorData = torch.Tensor(dataForPred).unsqueeze(0).long()
    print(tensorData.size())
    print(tensorData.type())
    return tensorData

if __name__ == '__main__':
    # tensorData = loadDataForPred()
    tensorData = mockData()
    prediction = test(tensorData)
    # print(prediction.size())
    # resultIndexList = torch.max(prediction[0],1)[1].numpy().tolist()
    # for i in range(len(resultIndexList)):
    #     resultIndexList[i] = 20 + resultIndexList[i]*20
    # print(resultIndexList)