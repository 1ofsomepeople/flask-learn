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
