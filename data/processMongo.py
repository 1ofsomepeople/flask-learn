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
def getAllFilePathList(rootPath, monthSelect = 0):
    filePathList = []
    monthList = os.listdir(rootPath)
    if(monthSelect != 0):
        monthList = [monthList[monthSelect-1]]
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

# 单个进程执行的任务
def processMongo(splitList,processIndex=1):
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

# mongo查询 根据时间和路名进行查询 返回{road time txtInfo.direction txtInfo.avespeed}组成的数组
# roadName 查询的路名字符串 '荣华南路'
# startStr1 开始时间字符串 "2020-11-09 00:00"
# endStr1 结束时间字符串 "2020-11-09 23:59"
def mongoSearch(roadName,startStr,endStr,direction,id):
    # print(collection.estimated_document_count())
    # startTime = datetime.datetime(2020, 11, 9, 0, 0)
    # startTime = parser.parse("2020-11-09 00:00")
    # endTime = datetime.datetime(2020, 11, 9, 23, 59)
    # endTime = parser.parse("2020-11-09 23:59")
    startTime = parser.parse(startStr)
    endTime = parser.parse(endStr)

    # 查找筛选属性在时间范围内的文档
    targetData = list(collection.find(
            # 查询条件
            {
                'time':{
                    "$gte":startTime,
                    "$lte":endTime
                    },
                'road': roadName,
                "txtInfo":{ "$elemMatch" :{"direction":direction,'section.id': id}},
                # 'txtInfo.direction': direction,
                # 'txtInfo.section.id': id,
            },
            # 检索的字段列表
            {
                "_id":0,
                "road":1,
                "time":1,
                "txtInfo.direction":1,
                "txtInfo.section":1
                # "txtInfo.section.speed":{ "$elemMatch" :{'txtInfo.section.id': id}},
                # "txtInfo.direction":1,
                # "txtInfo.section.speed":1
                # "txtInfo.section.speed":1,
                # "txtInfo.section":{"$slice": [int(id)-1,1] },
                
            }
        )
    )

    for item in targetData:
        # 转换时间格式
        item['time'] = str(item['time'])
        # 筛选出符合要求的路段的 方向 id 速度
        txtInfo = item['txtInfo']
        txtInfoRes = {}
        txtInfoRes['direction'] = direction
        for directionInfo in txtInfo:
            if(directionInfo['direction'] == direction):
                section = directionInfo['section']
                for sec in section:
                    if(sec['id'] == id):
                        txtInfoRes['sectionId'] = sec['id']
                        txtInfoRes['speed'] = sec['speed']
                        break
                break     
        item['txtInfo'] = txtInfoRes
    return targetData

# 根据查询参数进行查询，生成一个对象组成的数组
def paramSearch(roadname,direction,id):
    timeParam = [
        ["2020-11-09 07:00","2020-11-09 09:30"],["2020-11-09 17:00","2020-11-09 19:30"],
        ["2020-11-10 07:00","2020-11-10 09:30"],["2020-11-10 17:00","2020-11-10 19:30"],
        ["2020-11-11 07:00","2020-11-11 09:30"],["2020-11-11 17:00","2020-11-11 19:30"],
        ["2020-11-12 07:00","2020-11-12 09:30"],["2020-11-12 17:00","2020-11-12 19:30"],
        ["2020-11-13 07:00","2020-11-13 09:30"],["2020-11-13 17:00","2020-11-13 19:30"],
        ["2020-11-14 07:00","2020-11-14 09:30"],["2020-11-14 17:00","2020-11-14 19:30"],
        ["2020-11-15 07:00","2020-11-15 09:30"],["2020-11-15 17:00","2020-11-15 19:30"],
    ]
    resList = []
    for param in timeParam:
        res = mongoSearch(roadname,param[0],param[1],direction,id)
        resList.extend(res)
    return resList

# 对象数组导出转成json文件
# objArray对象数组
# jsonName json文件名
def objArray2Json(objArray, jsonName = 'test.json'):
    # 由于json中有中文，所以需要注明编码格式encoding及ensure_ascii
    with open(jsonName, 'w', encoding='utf-8') as json_file:  
        json.dump(objArray, json_file, ensure_ascii=False, indent=4)

# 根据路名生成相对应的json文件
def roadName2Json():
    roadList =[
        {
            "roadName": "中关村大街",  # NS id 3 SN id 5 
            "roadSectionParam": [
                ["中关村大街","NS","3"],
                ["中关村大街","SN","5"]
            ],
            "fileName": "1_中关村南大街海淀黄庄南公交站（靠十字路口的天桥）.json"
        },
        {
            "roadName": "中关村大街", # NS id 7 SN id 8  
            "roadSectionParam": [
                ["中关村大街","NS","7"],
                ["中关村大街","SN","8"]
            ],
            "fileName": "2_中关村南大街家乐福东边的天桥.json"
        },
        {
            "roadName": "中关村南大街", #NSid1 SNid3
            "roadSectionParam": [
                ["中关村南大街","NS","1"],
                ["中关村南大街","SN","3"]
            ],
            "fileName": "3_中关村南大街海淀科技大厦东天桥.json"
        },
        {
            "roadName": "北三环中路辅路（外环）",  # 北三环（外环）EWid4  #北三环（内环）WEid2
            "roadSectionParam": [
                ["北三环（外环）","EW","4"],
                ["北三环（内环）","WE","2"]
            ],
            "fileName": "4_蓟门桥东天桥（中国教育科学研究院北）.json"
        },
        {
            "roadName": "北三环西路辅路（外环）",  # 北三环（外环）EWid5  #北三环（内环）WEid1
            "roadSectionParam": [
                ["北三环（外环）","EW","5"],
                ["北三环（内环）","WE","1"]
            ],
            "fileName": "6_北三环西路（北京科学会堂）.json"
        },
        {
            "roadName": "北三环（内环）",  # 北三环（外环）EWid5  #北三环（内环）WEid1
            "roadSectionParam": [
                ["北三环（外环）","EW","5"],
                ["北三环（内环）","WE","1"]
            ],
            "fileName": "7_北三环西路（双安商场东-即北京UME影视城天桥）.json"
        },
        {
            "roadName": "远大路", # EWid1  WEid4
            "roadSectionParam": [
                ["远大路","EW","1"],
                ["远大路","WE","4"]
            ],
            "fileName": "8_世纪金源商场南边.json"
        },
        {
            "roadName": "北三环（外环）", # 北三环（外环）EWid5  #北三环（内环）WEid1
            "roadSectionParam": [
                ["北三环（外环）","EW","5"],
                ["北三环（内环）","WE","1"]
            ],
            "fileName": "9_北京华医皮肤医院北天桥（蓟门桥西）.json"
        },
        {
            "roadName": "西三环北路辅路（外环）", # 西三环北路辅路（外环） NSid1  西三环（外环）NSid1
            "roadSectionParam": [
                ["西三环北路辅路（外环）","NS","1"],
            ],
            "fileName": "10_西三环北路满庭芬芳西天桥.json"
        },
    ]

    for roadfile in roadList:
        roadParams = roadfile["roadSectionParam"]
        roadData = []
        for roadParam in roadParams:
            res = paramSearch(roadParam[0],roadParam[1],roadParam[2])
            roadData.extend(res)
        print(roadfile["fileName"])
        print(len(roadData))
        objArray2Json(roadData,roadfile["fileName"])



# 根据坐标点获取周围道路车流
def poiSearchMain():
    # [ 39.90711752 116.23571599] 八宝山地铁站 石景山路WE 上庄大街 NS 3 上庄大街 SN 1 
    # [ 39.85327291 116.37153198] 马家堡地铁站 马家堡西路 NS 1 马家堡西路 SN 5
    # [ 39.97587197 116.31781068] 海淀黄庄地铁站 知春路 EW 9 知春路 WE 1 中关村大街 NS 5,6 中关村大街 SN 5,6
    # [ 39.99881785 116.46822132] 望京地铁站  广顺北大街 NS 7 广顺北大街 SN 2
    # 四个节点周围道路的速度数据统计
    # 选一个工作日和一个周末就行
    # 生成numpy的矩阵
    timeParam = [
                    ["2020-11-09 00:00","2020-11-09 23:59"], # 周一
                    ["2020-11-14 00:00","2020-11-14 23:59"]  # 周六
                ]

    roadList = [
        {
            "pointName": "八宝山地铁站", # EWid1  WEid4
            "roadSectionParam": [
                ["上庄大街","NS","3"],
                ["上庄大街","SN","1"]
            ],
            "fileName": "1_八宝山地铁站.json"
        },
    ]        



if __name__ == '__main__':
    # res = mongoSearch("中关村大街","2020-11-09 07:00","2020-11-09 09:30","SN","5")
    # res = paramSearch("中关村大街","SN","5")
    # for i in res:
    #     print(i)
    # print(len(res))
    # roadName2Json()
    # fileList = getAllFilePathList(dataRootPath,11)
    # client['test']['testCollection'].
    