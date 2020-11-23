import json
import csv
import numpy as np
import pandas as pd
import torch
import time
import datetime
import os
import sys 
# 上级目录加到了sys.path里,导入上级目录下模块
sys.path.append("..") 
import lnglat_mercator_tiles_convertor as convertor
import dataCsv as datacsv
import png2json as png2json
# 导入隔壁文件夹下的模块
import pred_model.lr_online_test as lrmodel
import pred_model.sage_online_test as sagemodel

def getOrinDataDirPath():
    # original_dataDir_path是原始数据存放的目录的绝对路径
    original_dataDir_path = "originalData"
    path=os.path.abspath('.')   #表示执行环境的绝对路径
    if(os.path.split(path)[-1] == 'data'):
        original_dataDir_path = os.path.join(path,'originalData')
    elif(os.path.split(path)[-1] == 'flask-learn'):
        original_dataDir_path = os.path.join(path,'data','originalData')
    return original_dataDir_path

#  读取json文件，返回日期string和数据list
#  dirPath：文件夹路径 fileName：文件名
def readJsonData(dirPath,fileName):
    path = os.path.join(dirPath,fileName)
    jsonName = ''
    data = []
    with open(path, 'r') as f:
        datajson = json.load(f)
    jsonName = fileName.split('.')[0]
    data = datajson['data']
    return (jsonName,data)

# 根据一系列json数据文件生成数据表data.csv
# 生成data.csv文件
def getDataCsv():
    # 文件的目录路径
    dataDirPath = getOrinDataDirPath()
    # 目录下所有文件名的list
    fileList = os.listdir(dataDirPath)

    jsonName,data = readJsonData(dataDirPath,fileList[0])

    datacsv.createCsv(jsonName,data)

    for index in range(1,len(fileList)):
        jsonName,data = readJsonData(dataDirPath,fileList[index])
        datacsv.appendData(jsonName,data,-1)

# 加载用于模型预测的数据
def loadDataForPred(index = 0, dataType = 'gt'):
    # index超出阈值的处理
    index = index % 10

    file_name = 'data.csv'
    path=os.path.abspath('.')   #表示执行环境的绝对路径
    if(os.path.split(path)[-1] == 'data'):
        file_name = os.path.join(path,'data.csv')
    elif(os.path.split(path)[-1] == 'flask-learn'):
        file_name = os.path.join(path,'data','data.csv')
    # 读csv，生成dataFrame
    readCsv = pd.read_csv(file_name)
    # 经纬度的pointindex数组
    pointsIndex = np.array(readCsv.values)[:,0].tolist()

    # 每隔5列 选取数据
    selectData = np.array(readCsv.values)[:, index+1::5]
    # 前12列数据 用于预测
    dataForPred = selectData[:,0:12:1].tolist()
    # 第13列数据 预测结果的groundTruth
    dataGroudTruth = selectData[:,12].tolist()
    # 预测的时间
    predictTime = readCsv.columns.values.tolist()[index-10]

    if(dataType == 'gt'):
        return (predictTime,pointsIndex,dataGroudTruth)
    else:
        tensorData = torch.Tensor(dataForPred).unsqueeze(0).long()
        # # print(tensorData.size())
        # # print(tensorData.type())
        return (predictTime,pointsIndex,tensorData,dataGroudTruth)

# 获取gt数据object
def getGtData(inputDataIndex = 0):
    (predictTime,pointsIndex,dataGroudTruth) = loadDataForPred(inputDataIndex,'gt')

    for i in range(len(dataGroudTruth)):
        [lon,lat] = pointsIndex[i].split('-')
        lon = float(lon)
        lat = float(lat)
        value = 3
        if(dataGroudTruth[i] == 40):
            value = 7
        elif(dataGroudTruth[i] == 60):
            value = 10
        dataGroudTruth[i] = [lon,lat,value]
    resObj = {
        "jsonName":predictTime,
        "data":dataGroudTruth
    }
    return resObj

# 获取pred数据object
def getPredData(inputDataIndex = 0, method = 'lr'):
    (predictTime,pointsIndex,tensorData,dataGroudTruth) = loadDataForPred(inputDataIndex,'pred')

    prediction = None
    if(method == 'lr'):
        prediction = lrmodel.test(tensorData)
    elif(method == 'sage'):
        prediction = sagemodel.test(tensorData)
    print(prediction.size())

    resultIndexList = torch.max(prediction[0],1)[1].numpy().tolist()

    # numpy实现MAE MAPE
    y_pred = np.array(resultIndexList)
    y_pred = (y_pred+1)*20
    y_gt = np.array(dataGroudTruth)
    scorePrecision = round(np.mean(np.equal(y_gt,y_pred))*100,4)
    scoreMAE = round(np.mean(np.abs(y_pred - y_gt)),4)
    scoreMAPE = round(np.mean(np.abs((y_pred - y_gt)/y_gt))*100,4)

    for i in range(len(resultIndexList)):
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
        "data":resultIndexList,
        "scorePrecision":scorePrecision,
        "scoreMAE":scoreMAE,
        "scoreMAPE":scoreMAPE,
    }
    return predObj