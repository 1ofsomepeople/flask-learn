# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 16:17:28 2017

@author: zhangqi
"""
"""
rewrite this file, change it to python3 Sat 
    Oct 27 20:08:19 2018

add functions of convert in BD09 longitude and latitude degree, 
plane coordinates of Mercator projection, tile coordinates（tileX,tileY）
and pixel coordinates（pixelX, pixelY）in tiles
    Fri Nov 09 15:44:37 2018

@author: dingchaofan
"""
"""
add MultiThread function
add terminal argv parameter
The downloaded function is encapsulated as a function that passes arguments on the command line
use try and except to use python2 or python3
    Nov 17 11:19:25 2019

@author: dingchaofan
"""

try:  #python2
    import cStringIO
    import urllib2
except ImportError:  #python3
    import urllib.request
    import io

import json
import math
import os
import signal
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

import copy
import random
import threading
from threading import Thread
from urllib.parse import parse_qs, parse_qsl, urlparse

import requests

import lnglat_mercator_tiles_convertor as convertor
import png2json as png2json

tilePoint = png2json.readpoints() # points.json文件所含数据分类后的字典
resJsonData = [] # downloadMain()返回结果的临时容器


# Todo注释未完成
def showBmap(my_url:str) -> None:
    '''获取拥堵等级信息.

    1. 访问指定url并获取文件
    2. 若指定瓦片存在，顺序访问瓦片中的坐标池，为每个坐标附加拥堵属性值，最后保存到全局变量resJsonData
    '''

    # 引用全局变量
    global resJsonData
    global tilePoint

    # 获取文件并打开 #! 是在内存中操作，未保存到本地吧？
    try:  # python2
        file = cStringIO.StringIO(urllib2.urlopen(my_url).read())
    except NameError:  # python3
        file = io.BytesIO(urllib.request.urlopen(my_url, timeout=3).read())
    try:
        img = Image.open(file)
        img = img.convert("RGBA")
    except IOError:
        return None
    
    urlparam = parse_qs(urlparse(my_url).query)
    '''
    urlparse模块主要是用于解析url中的参数，对url按照一定格式进行拆分或拼接
    将url分为6个部分，返回一个包含6个字符串项目的元组：scheme、netloc、path、params、query、fragment

    parse_qs()将请求参数转换为字典
    '''
    
    level = int(urlparam['level'][0])
    tile_x = int(urlparam['x'][0])
    tile_y = int(urlparam['y'][0])
    tileName = str(tile_x) + str(tile_y)

    if (tileName in tilePoint.keys()):
        tempArr = tilePoint[tileName]
        for temppoint in tempArr:
            R, G, B, A = img.getpixel((temppoint[0], 255 - temppoint[1]))
            if (R + G + B > 0):
                # RGB归一化
                r = R / (R + G + B)
                g = G / (R + G + B)
                b = 1.0 - r - g
                value = png2json.RGB2Value(r, g, b) # value is in [0, 1, 3, 7, 10]
                resJsonData.append([temppoint[2], temppoint[3], value])
            else:
                # 对噪声赋值
                resJsonData.append([temppoint[2], temppoint[3], 0])
    return None




# Todo注释未完成
def down_a_map(time_now:int, start_x:int, start_y:int, x_range:int, y_range:int, level:int) -> None:
    '''函数说明.

    1. 根据参数生成要访问的URL
    2. 调用showBmap()更改全局变量resJsonData
    '''

    # start_x = 25265
    # start_y = 9386
    # level = 14
    # x_range = 8
    # y_range = 8
    # tile_x,tile_y is (8,)
    tile_x = np.arange(start_x - x_range, start_x, 1)
    tile_y = np.arange(start_y - y_range, start_y, 1)
    address = 'http://its.map.baidu.com:8002/traffic/TrafficTileService?'

    # tiles start from (3158,1173) to (3167,1182), left_bottom to right_top #! 这里的数值可能是错误的，也包括上面的注释
    # i,j is in 1~10
    for i in range(len(tile_x)):
        for j in range(len(tile_y)):
            url_ij = address + f'level={level}&x={tile_x[i]}&y={tile_y[j]}&time={time_now}'  # 14级瓦片坐标系的瓦片坐标
            showBmap(url_ij)

    return None




class MyThread(Thread):
    def __init__(self, time_now, start_x, start_y, x_range, y_range, level):
        Thread.__init__(self)
        self.time_now = time_now
        self.start_x = start_x
        self.start_y = start_y
        self.x_range = x_range
        self.y_range = y_range
        self.level = level

    def run(self):
        try_times = 3
        while (try_times > 0):
            try:
                down_a_map(self.time_now, self.start_x,
                            self.start_y, self.x_range,
                            self.y_range, self.level)
                break

            except Exception as e:
                time.sleep(2)
                try_times -= 1
                if (try_times > 0):
                    pass
                else:
                    print("retry fail")
                    print(e)
                    os.kill(os.getpid(), signal.SIGKILL)





def creat_file_name():
    '''由当前时间生成访问url的时间戳和要保存的文件名.

    '''
    
    time_now = int(time.time())
    time_local = time.localtime(time_now)
    file_name = time.strftime("%Y-%m-%d_%H-%M-%S", time_local)
    timestamp = time_now * 1000

    return timestamp, file_name




# Todo注释未完成
def Multi_Thread_DownLoad(timestamp, file_name, TileRight, TileTop,
                          TileRange, threadNum, mapLevel) -> None:             #! 修改了全局变量resJsonData
    '''多线程访问服务器.

    '''
    # 保存各个线程的数组
    threadSize = TileRange // threadNum  # 每个线程中的tiles行/列个数 注意要是整数int，否则会报错
    ssList = []

    # 多线程下载任务启动
    time_download_start = int(time.time())
    print("download start:" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_download_start)))
    for i in range(threadNum):
        for j in range(threadNum):
            ssTemp = MyThread(timestamp, TileRight - j * threadSize,
                              TileTop - i * threadSize, threadSize,
                              threadSize, mapLevel)
            ssList.append(ssTemp)

    for item in ssList:
        item.start()    # start方法开启一个新线程。把需要并行处理的代码放在run()方法中，start()方法启动线程将自动调用run()方法。

    for item in ssList:
        item.join()     # 等待子进程执行结束，主进程再往下执行

    return None




# Todo注释未完成
def downloadMain(levelParam=14) -> dict:
    '''访问当前时刻的拥堵数据，并生成文件.

    '''

    # 引用全局变量
    global resJsonData

    time_process_start = int(time.time())
    print("process start:" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_process_start)))

    # 由当前时间生成访问url的时间戳和要保存的文件名
    timestamp, file_name = creat_file_name()

    # 线程和任务的各个参数                                    #! 这些参数是什么意思
    mapParameter_default14 = [3171, 1186, 16, 2, 14]
    level       = levelParam                                # 14
    baselevel   = 14                                        # 14
    levelup     = level - baselevel                         # 0
    TileRight   = mapParameter_default14[0] * (2**levelup)  # 3171
    TileTop     = mapParameter_default14[1] * (2**levelup)  # 1186
    TileRange   = mapParameter_default14[2] * (2**levelup)  # 16
    threadNum   = mapParameter_default14[3]                 # 2 线程数
    mapLevel    = mapParameter_default14[4] + levelup       # 14 地图等级

    
    # 更新线程参数，不能整太多、太频繁，容易被封IP 线程过多会导致[Errno 104] Connection reset by peer
    if (level == 14):
        Multi_Thread_DownLoad(timestamp, file_name, TileRight, TileTop,
                                TileRange, threadNum, mapLevel)
    if (level == 16) or (level == 17):
        threadNum = 4
        Multi_Thread_DownLoad(timestamp, file_name, TileRight, TileTop,
                                TileRange, threadNum, mapLevel)


    resData = copy.copy(resJsonData)
    resJsonData.clear()
    jsonData = {'jsonName': file_name, 'data': resData}

    return jsonData
