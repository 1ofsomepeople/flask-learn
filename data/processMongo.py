import sys
import os
import datetime
import json
import re
import time
from tqdm import tqdm,trange
import threading
from multiprocessing import Process, Pool, freeze_support, RLock, cpu_count
import multiprocessing as mp
from pymongo import MongoClient

# 上级目录加到了sys.path里,导入上级目录下模块
sys.path.append("..")
import lnglat_mercator_tiles_convertor as convertor  # 引入坐标转换模块

from dateutil import parser # 处理MongoDB的日期时间格式

# 使用MongoClient建立连接
# client = MongoClient('localhost', 27017)
client = MongoClient('mongodb://localhost:27017/')

# pymongo简单测试
# 获取数据库，根据数据库名访问
# db = client.test
# db = client['test']
# 获取集合
# collection = db.testCollection
# collection = db['testCollection']
# CURD部分
# 增 add
# insert_one()插入单个文档 
# insert_many()插入多个文档
# collection.insert_many(newDocuments)
# 删 delete
# delete_one()和delete_many()方法来删除数据，括号中是筛选条件
# drop()用于删除集合collection
# collection.delete_one({'name':'456'})
# 改 update
# update_one() update_many()
# 查 find 
# find_one() find()
# print(collection.estimated_document_count())
# 以迭代的方式输出全部数据
# data = collection.find()
# print(list(data))

# 可以使用parser.parse() datetime.datetime()转换并插入时间
# 也可以使用这两个函数生成时间用于MongoDB的查找筛选
# time = parser.parse("2019-11-26 16:00")
# d = datetime.datetime(2019, 11, 26, 16, 15)
# 插入时间属性
# client.test.testCollection.insert_one({'date':d})
# client.test.testCollection.insert_one({'date':parser.parse("2019-11-26 16:00")})
# 查找筛选属性在时间范围内的文档
# print(list(client.test.testCollection.find({'date':{"$gte":time,"$lte":d}})))

# 原始数据的根目录
dataRootPath = 'E:\\traffic_data\\2.yang_traffic_data' 
# MongoDB实例client 数据库trafficData 集合roadData
collection = client.trafficData.roadData
# 测试集合
testCollection = client.test.testCollection
# 测试json数据文件
testPath = os.path.join(dataRootPath,'2020-01','2020-01-01','2020-01-01 00_00.json')

position = 0

# 读数据的单个json文件，整理和转换相关数据
def readJsonData(path):
    filesize = os.path.getsize(path)
    # 过滤脏文件
    if(filesize < 1024*280):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        datajson = json.load(f)
        if('roads' in datajson.keys()):
            roads = datajson['roads']
            for road in roads:
                if('time' in road.keys()):
                    road['time'] = parser.parse(road['time'])
                if('txtInfo' in road.keys()):
                    txtInfo = road['txtInfo']
                    for direction in txtInfo:
                        averageSpeed = 0
                        if('section' in direction.keys()):
                            section = direction['section']
                            secnum = len(section)
                            for secRoad in section:
                                if('info' in secRoad.keys()):
                                    speed = secRoad['info']
                                    speed = int(re.findall(r"\d+\.?\d*",speed)[0])
                                    secRoad['speed'] = speed
                                    averageSpeed = averageSpeed + speed
                                else:
                                    secnum -= 1
                            if(secnum > 0):
                                direction['avespeed'] = averageSpeed//len(section)
                            else:
                                direction['avespeed'] = 0
            if(len(roads) > 1):
                return roads

# 将转换后的数据data插入MongoDB的指定集合collectionName
def mongoInsertData(data,collectionInstance):
    dataId = collectionInstance.insert_many(data)
    # print("插入数据id是：", dataId)

# res = readJsonData(testPath)
# mongoInsertData(res,testCollection)

# 获取所有要插入文件的路径list
def getAllFilePathList(rootPath):
    filePathList = []
    monthList = os.listdir(rootPath)
    for month in monthList:
        monthPath = os.path.join(rootPath,month)
        if(os.path.isdir(monthPath)):
            print("enter month dir:",month)
            dayList = os.listdir(monthPath)
            for day in dayList:
                dayPath = os.path.join(monthPath,day)
                if(os.path.isdir(dayPath)):
                    # print("enter day dir:",day)
                    fileList = os.listdir(dayPath)
                    for file in fileList:
                        if(file.split('.')[-1] == 'json'):
                            filePath = os.path.join(dayPath,file)
                            filePathList.append(filePath)
    return filePathList

# 将原始数据批量插入到MongoDB中
def dataToMongoDB(fileList,threadIndex):
    try:
        with trange(len(fileList)) as t:
            for index in t:
                filename = os.path.basename(fileList[index])
                #设置进度条左边显示的信息
                t.set_description("ThreadID: %s,Procession: %s"%(threadIndex,filename))
                res = readJsonData(fileList[index])
                if(type(res).__name__=='list' and len(res)>1):
                    mongoInsertData(res,collection)
    except KeyboardInterrupt:
        t.close()
        raise
    t.close()

    # bar = tqdm(iterable=range(len(fileList)),ncols=50,position=threadIndex)
    # for index in bar:
    #     filename = os.path.basename(fileList[index])
    #     bar.set_description("Thread id: %s,Procession: %s"%(os.getpid(),filename))
    #     res = readJsonData(fileList[index])
    #     if(type(res).__name__=='list'):
    #         mongoInsertData(res,collection)

# 切分数组
def listSplit(originlist, n):
    splitList = []
    splitLen = len(originlist) // n
    start = 0
    while(start < len(originlist)):
        end = start + splitLen
        if(end > len(originlist)):
            end = -1
        templist = originlist[start:end]
        splitList.append(templist)
        start = start + splitLen
    return splitList
    

# 多线程执行任务
def threadTasks(fileList, threadNum = 10):
    splitList = listSplit(fileList, threadNum)
    threadList = []
    for threadIndex in range(threadNum): 
        taskDataList = splitList[threadIndex]
        threadtask = threading.Thread(target=dataToMongoDB,args=(taskDataList,threadIndex,))
        threadList.append(threadtask)
    return threadList

# 多线程执行主函数
def threadMain():
    # freeze_support()
    # threadNum = 5
    # L = list(range(threadNum))
    # fileList = getAllFilePathList(dataRootPath) 
    # splitList = listSplit(fileList, threadNum)
    # p = Pool(len(splitList),initializer=tqdm.set_lock, initargs=(RLock(),))
    # p.map(dataToMongoDB,splitList,[1,2,3,4,5])
    threadNum = 10
    fileList = getAllFilePathList(dataRootPath) 
    threadList = threadTasks(fileList, threadNum)
    for index in range(threadNum):
        threadList[index].start()
    for index in range(threadNum):
        threadList[index].join()
# threadMain()


# 多进程类
class MyMultiprocess(object):
    # 多进程初始化
    def __init__(self, process_num):
        self.pool = Pool(processes=process_num)

    def work(self, func, args):
        dataSplitList = args[0]
        processIndexList = args[1]
        print(len(dataSplitList[0]))
        print(len(processIndexList))
        for arg in processIndexList:
            self.pool.apply_async(func, (dataSplitList,arg,))
        self.pool.close()
        self.pool.join()


def processMongo(splitList,processIndex):
    data = splitList[processIndex]
    # for index in tqdm(range(len(data)), ncols=80, desc='执行任务' + str(processIndex) + ' pid:' + str(os.getpid())):
    #     filename = os.path.basename(data[index])
    #     res = readJsonData(fileList[index])
    #     if(type(res).__name__=='list' and len(res) > 1):
    #         mongoInsertData(res,collection)
    with tqdm(range(len(data)), 
        ncols=80, 
        # position=processIndex+1
        ) as t:
        for index in t:
            filename = os.path.basename(data[index])
            #设置进度条左边显示的信息
            t.set_description("执行任务%i pid:%i process:%s"%(processIndex,os.getpid(),filename))
            res = readJsonData(data[index])
            if(type(res).__name__=='list' and len(res) > 1):
                mongoInsertData(res,collection)

# 多进程执行主函数
def multiProcessMain():
    # 定义多进程数
    threadNum = 1000
    print('父进程 %s.' % os.getpid())
    # 获取所有文件的路径list
    fileList = getAllFilePathList(dataRootPath)
    # 获取路径list分片数组
    splitList = listSplit(fileList, threadNum)
    
    # 生成多进程任务
    mymultiprocess = MyMultiprocess(cpu_count())
    # mymultiprocess = MyMultiprocess(2)

    start = time.time()
    # 开始执行
    mymultiprocess.work(func=processMongo, args=(splitList,range(threadNum)))
    end = time.time()
    print("\n应用多进程耗时: %0.2f seconds" % (end - start))

if __name__ == '__main__':
    multiProcessMain()