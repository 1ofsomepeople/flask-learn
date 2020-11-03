import json
import csv
import numpy as np
import pandas as pd

# 要输入预测模型的dataArray
# dataArray = []
# 经纬度相对应的index数组
# pointsIndex = []

# 读取tempdata.json中的测试json数据
# 返回值：(jsonName,data)
# jsonName：表示日期的string
# data：17531个点的数组，[[lon,lat,value],...]
def readtestdata():
    jsonName = ''
    data = []
    with open('./tempdata.json', 'r') as f:
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
        value = point[2]
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
def appendData(dataname,data):
    # 读csv，生成dataFrame
    readCsv = pd.read_csv('data.csv')
    # 经纬度的pointindex数组
    pointsIndex = np.array(readCsv.values)[:,0].tolist()
    # 要插入的value数组
    valueList = [0]*len(pointsIndex)
    for point in data:
        lon = str(point[0])
        lat = str(point[1])
        lonlat = lon+'-'+lat
        value = point[2]
        if(point[2] == 3):
            value = 20
        elif(point[2] == 7):
            value = 40
        elif(point[2] == 10):
            value = 60
        else:
            value = 20
        index = pointsIndex.index(lonlat)
        valueList[index] = value
    # csv列的长度
    csv_columns_length = readCsv.shape[1]
    readCsv.insert(csv_columns_length,dataname,valueList,allow_duplicates=True)
    if(csv_columns_length>=16):
        readCsv.drop(readCsv.columns[1], axis=1, inplace=True)
    readCsv.to_csv('data.csv',index=0,encoding='gbk')