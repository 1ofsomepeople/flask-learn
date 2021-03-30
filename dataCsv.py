import csv
import datetime
import json
import time

import numpy as np
import pandas as pd
import torch

import pred_model.lr_online_test as lrmodel
import pred_model.sage_online_test as sagemodel


def readtestdata(path='./tempdata.json') -> (str, list):
    '''读取tempdata.json中的测试json数据.

    Args:
        path(str):文件路径

    Returns:
        jsonName(str):表示日期的string
        data(list):17531个点的数组，[[lon,lat,value],...]

    '''

    with open(path, 'r') as f:
        datajson = json.load(f)
    jsonName = datajson['jsonName']
    data = datajson['data']
    return (jsonName, data)


def createCsv(testjsonName:str, testdata:list, csvName='data.csv') -> None:
    '''读取json文件并创建csv.

    Args:
        testjsonName(str):要读取的json文件文件名
        testdata(list):json文件里的数据
        csvName(str):要保存的csv文件文件名

    '''
    pointIndex = []
    valueList = []
    for point in testdata:
        lonlat = f'{point[0]}-{point[1]}'
        pointIndex.append(lonlat)
        value = 20
        if (point[2] == 7):
            value = 40
        if (point[2] == 10):
            value = 60
        valueList.append(value)

    initCsv = pd.DataFrame({'lonlat': pointIndex, testjsonName: valueList})  # 使用字典初始化DataFrame类 
    initCsv.to_csv(csvName, index=0, encoding='gbk')  # 保存DataFrame到文件


def appendData(dataname, data, max_num=16, csvName='data.csv') -> None:
    '''将json的data插入csv中.

    Args:
        dataname(str):列名，json数据获取的日期
        data(list):要插入的数据，形如[[116.626503, 39.864, 3], ...]
        max(int):表的最大数据列数，当超过max组数据被插入的时候删除第一列数据,当max=-1时不删除第一列
        csvName(str):要修改的csv文件，默认为./data.csv
    
    '''

    # 读csv，生成dataFrame
    readCsv = pd.read_csv(csvName)

    # 经纬度的pointindex数组
    pointsIndex = np.array(readCsv.values)[:, 0].tolist()

    # 要插入的value数组
    valueList = [20] * len(pointsIndex)

    for point in data:
        lonlat = f'{point[0]}-{point[1]}'
        if (point[2] == 7):
            value = 40
        elif (point[2] == 10):
            value = 60
        else:
            continue
        index = pointsIndex.index(lonlat)
        valueList[index] = value

    # csv列的长度
    csv_columns_length = readCsv.shape[1]
    readCsv.insert(csv_columns_length,
                   dataname,
                   valueList,
                   allow_duplicates=True)
    
    # 如果超出最大列数，先丢弃第2列，再添加数据到最后一列
    if (max_num != -1 and csv_columns_length >= max_num):
        readCsv.drop(readCsv.columns[1], axis=1, inplace=True)
    readCsv.to_csv(csvName, index=0, encoding='gbk')


def loadDataForPred() -> tuple:
    '''加载data.csv文件的数据用于模型预测.

    Returns:
        predictTime(str):预测时间
        pointsIndex(list):存放经纬度坐标的数组
        tensorData(Tensor):用来预测的数据

    '''

    # 读csv，生成dataFrame
    readCsv = pd.read_csv('data.csv')

    # 经纬度的pointindex数组
    pointsIndex = np.array(readCsv.values)[:, 0].tolist()

    # 最后一列的列名，时间
    lastColumnName = readCsv.columns.values.tolist()[-1]

    # 转换为时间数组 strptime()
    timeArray = time.strptime(lastColumnName, "%Y-%m-%d_%H-%M-%S")

    # 转换为时间戳 mktime()
    timeStamp = int(time.mktime(timeArray)) + 60 * 5

    # 时间戳转时间字符串
    predictTime = datetime.datetime.fromtimestamp(timeStamp)
    predictTime = predictTime.strftime("%Y-%m-%d_%H-%M-%S")

    dataForPred = np.array(readCsv.values)[:, -12:].tolist()
    # 将数据变为Tensor类型
    tensorData = torch.Tensor(dataForPred).unsqueeze(0).long()
    return (predictTime, pointsIndex, tensorData)


def getPred(type='lr') -> dict:
    '''获取每个坐标的预测数据.

    Args:
        type(str):预测类型

    Returns：
        predObj(dict):预测数据

    '''

    (predictTime, pointsIndex, tensorData) = loadDataForPred()
    prediction = None
    # 使用lr方法预测
    if (type == 'lr'):
        prediction = lrmodel.test(tensorData)
    # 使用sage方法预测
    if (type == 'sage'):
        prediction = sagemodel.test(tensorData)
    print(prediction.size())

    resultIndexList = torch.max(prediction[0], 1)[1].numpy().tolist()
    
    # 为每一个数据点添加拥堵等级信息
    for i in range(len(resultIndexList)):
        [lon, lat] = pointsIndex[i].split('-')  # pointsIndex[i] looks like this:'116.626503-39.864'
        lon = float(lon)
        lat = float(lat)
        value = 3
        if (resultIndexList[i] == 1):
            value = 7
        if (resultIndexList[i] == 2):
            value = 10
        resultIndexList[i] = [lon, lat, value]
    
    predObj = {"jsonName": predictTime, "data": resultIndexList}  # resultIndexList looks like [[116.626503, 39.864, 3], ...]
    return predObj
