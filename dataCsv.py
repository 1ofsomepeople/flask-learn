import json
import csv
import numpy as np
import pandas as pd
import torch
import pred_model.lr_online_test as lrmodel
import pred_model.sage_online_test as sagemodel
import time
import datetime

# 经纬度相对应的index数组
# pointsIndex = []

# 读取tempdata.json中的测试json数据
# 返回值：(jsonName,data)
# jsonName：表示日期的string
# data：17531个点的数组，[[lon,lat,value],...]
def readtestdata(path = './tempdata.json'):
    jsonName = ''
    data = []
    with open(path, 'r') as f:
        # jsonName = json.load(f).get('jsonName')
        datajson = json.load(f)
    jsonName = datajson['jsonName']
    data = datajson['data']
    return (jsonName,data)

# 创建并初始化csv
def createCsv(testjsonName,testdata):
    pointIndex = []
    valueList = []
    for point in testdata:
        lon = str(point[0])
        lat = str(point[1])
        lonlat = lon+'-'+lat
        pointIndex.append(lonlat)
        value = 20
        if(point[2] == 3):
            value = 20
        elif(point[2] == 7):
            value = 40
        elif(point[2] == 10):
            value = 60
        else:
            value = 20
        valueList.append(value)

    initCsv = pd.DataFrame({'lonlat':pointIndex,testjsonName:valueList})
    initCsv.to_csv('data.csv',index=0,encoding='gbk')

# 将json的data插入csv中
# dataname：列名，json数据日期string
# data：一列数据，值为1,3,7,10,在此函数中映射到20 40 60
# max：表的最大数据列数，当超过max组数据被插入的时候删除第一列数据,当max=-1时不删除第一列
def appendData(dataname,data,max=16):
    # 读csv，生成dataFrame
    readCsv = pd.read_csv('data.csv')
    # 经纬度的pointindex数组
    pointsIndex = np.array(readCsv.values)[:,0].tolist()
    # 要插入的value数组
    valueList = [20]*len(pointsIndex)
    for point in data:
        lon = str(point[0])
        lat = str(point[1])
        lonlat = lon+'-'+lat
        value = 20
        if(point[2] == 3):
            value = 20
            continue
        elif(point[2] == 7):
            value = 40
        elif(point[2] == 10):
            value = 60
        else:
            value = 20
            continue
        index = pointsIndex.index(lonlat)
        valueList[index] = value
    # csv列的长度
    csv_columns_length = readCsv.shape[1]
    readCsv.insert(csv_columns_length,dataname,valueList,allow_duplicates=True)
    if(max != -1 and csv_columns_length >= max):
        readCsv.drop(readCsv.columns[1], axis=1, inplace=True)
    readCsv.to_csv('data.csv',index=0,encoding='gbk')

# 加载用于模型预测的数据
def loadDataForPred():
    # 读csv，生成dataFrame
    readCsv = pd.read_csv('data.csv')
    # 经纬度的pointindex数组
    pointsIndex = np.array(readCsv.values)[:,0].tolist()

    # 最后一列的列名，时间
    lastColumnName = readCsv.columns.values.tolist()[-1]
    # 转换为时间数组 strptime()
    timeArray = time.strptime(lastColumnName, "%Y-%m-%d_%H-%M-%S")
    # 转换为时间戳 mktime()
    timeStamp = int(time.mktime(timeArray)) + 60*5
    # 时间戳转时间字符串
    predictTime = datetime.datetime.fromtimestamp(timeStamp)
    predictTime = predictTime.strftime("%Y-%m-%d_%H-%M-%S")

    dataForPred = np.array(readCsv.values)[:,-12:].tolist()
    tensorData = torch.Tensor(dataForPred).unsqueeze(0).long()
    # print(tensorData.size())
    # print(tensorData.type())
    return (predictTime,pointsIndex,tensorData)

def getPred(type='lr'):
    (predictTime,pointsIndex,tensorData) = loadDataForPred()
    prediction = None
    # 使用lr方法预测
    if(type == 'lr'):
        prediction = lrmodel.test(tensorData)
    # 使用sage方法预测
    elif(type == 'sage'):
        prediction = sagemodel.test(tensorData)
    print(prediction.size())
    resultIndexList = torch.max(prediction[0],1)[1].numpy().tolist()
    for i in range(len(resultIndexList)):
        # resultIndexList[i] = 20 + resultIndexList[i]*20
        [lon,lat] = pointsIndex[i].split('-')
        lon = float(lon)
        lat = float(lat)
        value = 3
        if(resultIndexList[i] == 1):
            value = 7
        elif(resultIndexList[i] == 2):
            value = 10
        resultIndexList[i] = [lon,lat,value]
    predObj = {
        "jsonName":predictTime,
        "data":resultIndexList
    }
    return predObj