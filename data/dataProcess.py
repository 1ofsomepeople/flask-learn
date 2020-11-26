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

def getOrinDataDirPath(dataDirName = "originalData"):
    # original_dataDir_path是原始数据存放的目录的绝对路径
    original_dataDir_path = dataDirName
    path=os.path.abspath('.')   #表示执行环境的绝对路径
    if(os.path.split(path)[-1] == 'data'):
        original_dataDir_path = os.path.join(path,dataDirName)
    elif(os.path.split(path)[-1] == 'flask-learn'):
        original_dataDir_path = os.path.join(path,'data',dataDirName)
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
def getDataCsv(csvName,dataDirName):
    # 文件的目录路径
    dataDirPath = getOrinDataDirPath(dataDirName)
    # 目录下所有文件名的list
    fileList = os.listdir(dataDirPath)

    jsonName,data = readJsonData(dataDirPath,fileList[0])

    datacsv.createCsv(jsonName,data,csvName)

    for index in range(1,len(fileList)):
        print('index:'+str(index)+';fileName:'+fileList[index])
        jsonName,data = readJsonData(dataDirPath,fileList[index])
        datacsv.appendData(jsonName,data,-1,csvName)

# 加载用于模型预测的数据
def loadDataForPred(index = 0, dataType = 'gt', dataName='data.csv'):
    # index超出阈值的处理
    index = index % 10

    file_name = dataName
    path=os.path.abspath('.')   #表示执行环境的绝对路径
    if(os.path.split(path)[-1] == 'data'):
        file_name = os.path.join(path,dataName)
    elif(os.path.split(path)[-1] == 'flask-learn'):
        file_name = os.path.join(path,'data',dataName)
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
def getGtData(inputDataIndex = 0, dataName='data.csv'):
    (predictTime,pointsIndex,dataGroudTruth) = loadDataForPred(inputDataIndex,'gt',dataName)

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
def getPredData(inputDataIndex = 0, method = 'lr', dataName='data.csv'):
    (predictTime,pointsIndex,tensorData,dataGroudTruth) = loadDataForPred(inputDataIndex,'pred',dataName)

    prediction = None
    if(method == 'lr'):
        prediction = lrmodel.test(tensorData)
    elif(method == 'sage'):
        prediction = sagemodel.test(tensorData)
    print(prediction.size())

    resultIndexList = torch.max(prediction[0],1)[1].numpy().tolist()

    # numpy实现MAE MAPE
    y_pred = np.array(resultIndexList)
    y_pred = (y_pred+1)*20 # 预测值nparray
    y_gt = np.array(dataGroudTruth) # 真实值nparray
    scorePrecision = round(np.mean(np.equal(y_gt,y_pred))*100,4) # 准确率
    scoreMAE = round(np.mean(np.abs(y_pred - y_gt)),4) # 平均绝对误差
    scoreMAPE = round(np.mean(np.abs((y_pred - y_gt)/y_gt))*100,4) # 平均绝对百分比误差

    y_pred_list = y_pred.tolist() # 预测值list
    y_gt_list = dataGroudTruth # 真实值list 
    TPclear = 0 # 畅通的TP个数
    TPslow = 0 # 缓行的TP个数
    TPjam = 0 # 拥堵的TP个数
    for index in range(len(y_gt_list)):
        if(y_gt_list[index] == 20 and y_pred_list[index] == 20):
            TPclear += 1
        if(y_gt_list[index] == 40 and y_pred_list[index] == 40):
            TPslow += 1
        if(y_gt_list[index] == 60 and y_pred_list[index] == 60):
            TPjam += 1
    clearNum = len(y_gt[np.where(y_gt==20)]) # 畅通的真值节点个数
    slowNum = len(y_gt[np.where(y_gt==40)]) # 缓行的真值节点个数
    jamNum = len(y_gt[np.where(y_gt==60)]) # 拥堵的真值节点个数
    precisionClear = round(TPclear/clearNum*100,4) # 通畅准确率
    precisionSlow = round(TPslow/slowNum*100,4) # 缓行准确率
    precisionJam = round(TPjam/jamNum*100,4) # 拥堵准确率
    precisionSlowJam = round((TPslow+TPjam)/(slowNum+jamNum)*100,4) # 缓行和拥堵的准确率
    precision = round((TPclear+TPslow+TPjam)/(clearNum+slowNum+jamNum)*100,4) # 总体的准确率，等同于scorePrecision

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
        "jsonName":predictTime, # data数据名，时间string
        "data":resultIndexList, # data数据
        "scorePrecision":scorePrecision, # 准确率
        "scoreMAE":scoreMAE, # 平均绝对误差
        "scoreMAPE":scoreMAPE, # 平均绝对百分比误差
        "precisionClear":precisionClear, # 通畅准确率
        "precisionSlow":precisionSlow, # 缓行准确率
        "precisionJam":precisionJam, # 拥堵准确率
        "precisionSlowJam":precisionSlowJam, # 缓行和拥堵的准确率
        "precision":precision, # 总体的准确率，等同于scorePrecision
    }
    return predObj