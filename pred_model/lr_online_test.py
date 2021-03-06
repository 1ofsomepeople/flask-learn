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
        # print(in_dim, hid_c) # 3 12
        self.embedding = nn.Embedding(in_dim, hid_c)

    def forward(self, source):
        source = source // 20 - 1
        # print("source",source)
        source = self.embedding(source)
        # print("source after embeding",source)
        return source

class LinearRegression(nn.Module):
    def __init__(self, in_dim, hid_c, src_len):
        # in_dim, hid_c, src_len 3,12,12
        super(LinearRegression, self).__init__()
        self.oneHotEmbed = OneHotProcess(in_dim, hid_c)
        self.linear = nn.Linear(src_len * hid_c, in_dim)

    def forward(self, input_data, device):
        source = input_data.to(device)  # [B, N, src_len]

        input_feature = self.oneHotEmbed(source)  # [B, N, src_len, hid_dim]
        # print("input_feature",input_feature)

        B, N, src_len, hid_c = input_feature.size() # 1 17531 12 12

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
    path=os.path.abspath('.')   #表示执行环境的绝对路径
    if(os.path.split(path)[-1] == 'pred_model'):
        file_name = os.path.join(path,'lr.pkl')
    elif(os.path.split(path)[-1] == 'flask-learn'):
        file_name = os.path.join(path,'pred_model','lr.pkl')

    checkpoint = torch.load(file_name, device)
    model_para = checkpoint["model"]
    option = checkpoint["setting"]

    cudnn.benchmark = True 
    # print("option.hid_c:", option.hid_c) # 12
    # print("option.h_step:", option.h_step) # 12
    model = LinearRegression(3, option.hid_c, option.h_step) # 3,12,12

    model.load_state_dict(model_para)

    model = model.to(device)

    # 计算模型参数数量
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2f" % (total)) # Number of parameter: 471.00

    prediction = model(test_data, device)

    return prediction


# if __name__ == '__main__':
#     test_set = TrafficData(folder="data", train_ratio=0.6, valid_ratio=0.2, data_type="test", h_step=12, f_step=1)
#     test_data = torch.cat((test_set[0]["x"].unsqueeze(0), test_set[1]["x"].unsqueeze(0)), dim=0)
#     prediction = test(test_data)
#     print(prediction.size())

def mockData():
    # 生成17531*12的二维数组
    timeData = [[random.randrange(20,61,20) for col in range(12)] for row in range(17531)]
    # 对数据维度进行扩充。给指定位置加上维数为一的维度，比如原本有个三行的数据（3），unsqueeze(0)后就会在0的位置加了一维就变成一行三列（1,3）
    tensorData = torch.Tensor(timeData).unsqueeze(0).long() 
    # print(tensorData.size()) # torch.Size([1, 17531, 12])
    # print(tensorData.type()) # torch.LongTensor
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
    # print("prediction.size():",prediction.size())
    # resultIndexList = torch.max(prediction[0],1)[1].numpy().tolist()
    # for i in range(len(resultIndexList)):
    #     resultIndexList[i] = 20 + resultIndexList[i]*20
    # print("resultIndexList",resultIndexList)