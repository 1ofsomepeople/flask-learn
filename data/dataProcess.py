import csv
import datetime
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

# 上级目录加到了sys.path里,导入上级目录下模块
sys.path.append("..")
import dataCsv as datacsv
import lnglat_mercator_tiles_convertor as convertor
import png2json as png2json
# 导入隔壁文件夹下的模块
import pred_model.lr_online_test as lrmodel
import pred_model.sage_online_test as sagemodel


def getOrinDataDirPath(dataDirName="originalData") -> str:
    '''获取originalData文件夹的绝对路径.

    Args:
        dataDirName(str):originalData文件夹的名称

    Returns:
        original_dataDir_path(str):originalData文件夹的绝对路径

    '''

    original_dataDir_path = dataDirName
    path = os.path.abspath('.')  #表示执行环境的绝对路径
    if (os.path.split(path)[-1] == 'data'):
        original_dataDir_path = os.path.join(path, dataDirName)
    elif (os.path.split(path)[-1] == 'flask-learn'):
        original_dataDir_path = os.path.join(path, 'data', dataDirName)
    return original_dataDir_path


def readJsonData(dirPath, fileName) -> (str, list):
    '''读取json文件，返回日期string和数据list.

    Args:
        dirPath(str):文件夹路径
        fileName(str):文件名

    Returns:
        jsonName(str):json文件的文件名
        data(list):数据点存放的列表，形如[[116.389463, 39.905922, 3], ...]
    '''

    path = os.path.join(dirPath, fileName)
    with open(path, 'r') as f:
        datajson = json.load(f)

    jsonName = fileName.split('.')[0]
    data = datajson['data']

    return (jsonName, data)


def getDataCsv(csvName, dataDirName) -> None:
    '''使用多个json数据文件生成data.csv数据文件.

    Args:
        csvName(str):要保存的文件名
        dataDirName(str):json文件存放的目录名./data/originalData

    '''

    # 文件的目录路径
    dataDirPath = getOrinDataDirPath(dataDirName)
    # 目录下所有文件名的list
    fileList = os.listdir(dataDirPath)
    # 首先添加一个json文件到csv文件
    jsonName, data = readJsonData(dataDirPath, fileList[0])
    datacsv.createCsv(jsonName, data, csvName)
    # 继续添加后面的json文件
    for index in range(1, len(fileList)):
        print('index:' + str(index) + ';fileName:' + fileList[index])
        jsonName, data = readJsonData(dataDirPath, fileList[index])
        datacsv.appendData(jsonName, data, -1, csvName)


def loadDataForPred(index=0, dataType='gt', dataName='data.csv') -> tuple:
    '''加载用于模型预测的数据.

    Args:
        index(int):
        dataType(str):加载类型
        dataName(str):文件名

    Returns:
        predictTime():
        pointsIndex():
        (tensorData)():
        dataGroudTruth():
    
    '''

    # index超出阈值的处理
    index = index % 10

    file_name = dataName
    path = os.path.abspath('.')  #表示执行环境的绝对路径
    if (os.path.split(path)[-1] == 'data'):
        file_name = os.path.join(path, dataName)
    elif (os.path.split(path)[-1] == 'flask-learn'):
        file_name = os.path.join(path, 'data', dataName)
    # 读csv，生成dataFrame
    readCsv = pd.read_csv(file_name)
    # 经纬度的pointindex数组
    pointsIndex = np.array(readCsv.values)[:, 0].tolist()
    # 每隔5列 选取数据(可用数据范围7:30~8:39，70列)
    selectData = np.array(readCsv.values)[:, index + 1::5]
    # 前12列数据 用于预测
    dataForPred = selectData[:, 0:12:1].tolist()
    # 第13列数据 预测结果的groundTruth
    dataGroudTruth = selectData[:, 12].tolist()
    # 预测的时间
    predictTime = readCsv.columns.values.tolist()[index - 10]

    if (dataType == 'gt'):
        return (predictTime, pointsIndex, dataGroudTruth)
    else:
        tensorData = torch.Tensor(dataForPred).unsqueeze(0).long()
        return (predictTime, pointsIndex, tensorData, dataGroudTruth)


def getGtData(inputDataIndex=0, dataName='data.csv') -> dict:
    '''获取真实数据.

    Args:
        inputDataIndex(int):
        dataName(str):文件名

    Returns:
        resObj(dict):真实数据

    '''

    (predictTime, pointsIndex, dataGroudTruth) = loadDataForPred(inputDataIndex, 'gt', dataName)

    for i in range(len(dataGroudTruth)):
        [lon, lat] = pointsIndex[i].split('-')
        lon = float(lon)
        lat = float(lat)
        value = 3
        if (dataGroudTruth[i] == 40):
            value = 7
        elif (dataGroudTruth[i] == 60):
            value = 10
        dataGroudTruth[i] = [lon, lat, value]

    resObj = {"jsonName": predictTime, "data": dataGroudTruth}
    return resObj


def getPredData(inputDataIndex=0, method='lr', dataName='data.csv') -> dict:
    '''获取预测数据.

    Args:
        inputDataIndex(int):
        method(str):预测方法
        dataName(str):文件名

    Returns:
        predObj(dict):预测数据
    '''

    (predictTime, pointsIndex, tensorData, dataGroudTruth) = loadDataForPred(inputDataIndex, 'pred', dataName)

    prediction = None
    if (method == 'lr'):
        prediction = lrmodel.test(tensorData)
    elif (method == 'sage'):
        prediction = sagemodel.test(tensorData)
    print(prediction.size())

    resultIndexList = torch.max(prediction[0], 1)[1].numpy().tolist()

    # numpy实现MAE MAPE
    y_pred = np.array(resultIndexList)
    y_pred = (y_pred + 1) * 20  # 预测值nparray
    y_gt = np.array(dataGroudTruth)  # 真实值nparray
    scorePrecision = round(np.mean(np.equal(y_gt, y_pred)) * 100, 4)  # 准确率
    scoreMAE = round(np.mean(np.abs(y_pred - y_gt)), 4)  # 平均绝对误差
    scoreMAPE = round(np.mean(np.abs((y_pred - y_gt) / y_gt)) * 100, 4)  # 平均绝对百分比误差
    y_pred_list = y_pred.tolist()  # 预测值list
    y_gt_list = dataGroudTruth  # 真实值list
    TPclear = 0  # 畅通的TP个数
    TPslow = 0  # 缓行的TP个数
    TPjam = 0  # 拥堵的TP个数

    for index in range(len(y_gt_list)):
        if (y_gt_list[index] == 20 and y_pred_list[index] == 20):
            TPclear += 1
        if (y_gt_list[index] == 40 and y_pred_list[index] == 40):
            TPslow += 1
        if (y_gt_list[index] == 60 and y_pred_list[index] == 60):
            TPjam += 1
    
    clearNum = len(y_gt[np.where(y_gt == 20)])           # 畅通的真值节点个数
    slowNum = len(y_gt[np.where(y_gt == 40)])            # 缓行的真值节点个数
    jamNum = len(y_gt[np.where(y_gt == 60)])             # 拥堵的真值节点个数
    precisionClear = round(TPclear / clearNum * 100, 4)  # 通畅准确率
    precisionSlow = round(TPslow / slowNum * 100, 4)     # 缓行准确率
    precisionJam = round(TPjam / jamNum * 100, 4)        # 拥堵准确率
    precisionSlowJam = round((TPslow + TPjam) / (slowNum + jamNum) * 100, 4)                # 缓行和拥堵的准确率
    precision = round((TPclear + TPslow + TPjam) / (clearNum + slowNum + jamNum) * 100, 4)  # 总体的准确率，等同于scorePrecision

    for i in range(len(resultIndexList)):
        [lon, lat] = pointsIndex[i].split('-')
        lon = float(lon)
        lat = float(lat)
        value = 3
        if (resultIndexList[i] == 1):
            value = 7
        elif (resultIndexList[i] == 2):
            value = 10
        resultIndexList[i] = [lon, lat, value]
    
    predObj = {
        "jsonName": predictTime,                # data数据名，时间string
        "data": resultIndexList,                # data数据
        "scorePrecision": scorePrecision,       # 准确率
        "scoreMAE": scoreMAE,                   # 平均绝对误差
        "scoreMAPE": scoreMAPE,                 # 平均绝对百分比误差
        "precisionClear": precisionClear,       # 通畅准确率
        "precisionSlow": precisionSlow,         # 缓行准确率
        "precisionJam": precisionJam,           # 拥堵准确率
        "precisionSlowJam": precisionSlowJam,   # 缓行和拥堵的准确率
        "precision": precision,                 # 总体的准确率，等同于scorePrecision
    }
    return predObj


def processPointData() -> None:
    '''处理点数据示例.csv wgs84坐标转换到gcj02.

    '''

    # 读csv，生成dataFrame
    readCsv = pd.read_csv("点数据示例.csv")
    # 数据数组
    points = np.array(readCsv.values).tolist()
    # 表头
    header = readCsv.columns.values.tolist()

    for index in range(len(points)):
        curData = points[index]
        lon = float(curData[0].split(',')[0])
        lat = float(curData[0].split(',')[1])
        lonlat = list(convertor.wgs84_to_gcj02(lon, lat))
        lon = lonlat[0]
        lat = lonlat[1]
        lonlat = str(lon) + ',' + str(lat)
        points[index][0] = lonlat

    # numpy转pandas
    readCsv = pd.DataFrame(points)
    # 修改DataFrame的列名
    readCsv.columns = header
    # 保存成csv文件
    fileName = '点数据示例gcj02.csv'
    readCsv.to_csv(fileName, index=0, encoding='utf_8_sig')


def reverseCSV(path) -> None:
    '''csv行列互换.

    Args:
        path(str):文件所在路径

    '''

    df = pd.read_csv(path)
    data = df.values  # data是数组，直接从文件读出来的数据格式是数组
    index1 = list(df.keys())  # 获取原有csv文件的标题，并形成列表
    data = list(map(list, zip(*data)))  # map()可以单独列出列表，将数组转换成列表
    data = pd.DataFrame(data, index=index1)  # 将data的行列转换
    data.to_csv(r'reverseCsv.csv', header=0, encoding='utf_8_sig')


# if __name__ == '__main__':
#     processPointData()
#     reverseCSV("2019-04-02.csv")
