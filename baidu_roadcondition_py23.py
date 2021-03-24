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

# tilePoint是从points.json中读取的点数据
tilePoint = png2json.readpoints()
# resJsonData结果的jsondata
resJsonData = []


# show tile pictures 256*256
def showBmap(my_url):

    # 引用全局变量
    global resJsonData
    global tilePoint

    size = 256

    # 随便选一个代理ip
    # proxy = urllib.request.ProxyHandler({'http': resip[0]})
    # opener = urllib.request.build_opener(proxy, urllib.request.HTTPHandler)
    # urllib.request.install_opener(opener)

    try:  # python2
        file = cStringIO.StringIO(urllib2.urlopen(my_url).read())
    except NameError:  # python3
        file = io.BytesIO(urllib.request.urlopen(my_url, timeout=3).read())
    try:
        img = Image.open(file)
        img = img.convert("RGB")
    except IOError:
        # print('fail to convert')
        return np.zeros((size, size))
    
    urlparam = parse_qs(urlparse(my_url).query)                                                         #! 这一行代码做了什么？
    level = int(urlparam['level'][0])
    tile_x = int(urlparam['x'][0])
    tile_y = int(urlparam['y'][0])
    tileName = str(tile_x) + str(tile_y)

    if (tileName in tilePoint.keys()):
        tempArr = tilePoint[tileName]
        for temppoint in tempArr:
            R, G, B = img.getpixel((temppoint[0], 255 - temppoint[1]))
            if (R + G + B > 0):
                # RGB归一化 消除光照影响
                r = R / (R + G + B)
                g = G / (R + G + B)
                b = 1 - r - g
                value = png2json.RGB2Value(r, g, b)
                resJsonData.append([temppoint[2], temppoint[3], value])
            else:
                # print('RGB和不大于0',tileName,temppoint,R,G,B)
                resJsonData.append([temppoint[2], temppoint[3], 0])
    return img


def down_a_map(time_now, start_x, start_y, x_range, y_range, level):

    # start_x = 25265
    # start_y = 9386
    # level = 17
    # x_range = 48
    # y_range = 48
    size = 256

    # tile_x,tile_y is (10,)
    tile_x = np.arange(start_x - x_range, start_x, 1)
    tile_y = np.arange(start_y - y_range, start_y, 1)

    # creat a 3 axis array. The size of myMap is (2560,2560,3), type is uint8
    myMap = np.zeros((len(tile_y) * size, len(tile_x) * size, 3), dtype='uint8')
    address = 'http://its.map.baidu.com:8002/traffic/TrafficTileService?'

    # tiles start from (3158,1173) to (3167,1182), left_bottom to right_top
    # i,j is in 1~10
    for i in range(len(tile_x)):
        # print(i)
        for j in range(len(tile_y)):
            # print(i,j)
            # url_ij = address+('x=%d&y=%d&z=%d')%(tile_x[i],tile_y[j],level)
            url_ij = address + ('level=%d&x=%d&y=%d&time=%d') % (
                level, tile_x[i], tile_y[j], time_now)
            # print(url_ij)
            temp = showBmap(url_ij)

    return myMap                                                                                            #! myMap有修改过吗？


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
                self.result = down_a_map(self.time_now, self.start_x,
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

    def get_result(self):
        return self.result


# 三种merge stage2的保存方式：


# 1.用concatenate方法在np.array上进行合并并保存
def concatenateSave(ssListForVstack, file_name):

    # 用concatenate方法在np.array上进行合并并保存
    time_concatenate1 = int(time.time())
    # print(len(ssListForVstack))
    ss = np.concatenate((ssListForVstack[::-1]), axis=0)
    plt.imsave(file_name + '.png', ss)

    time_concatenate2 = int(time.time())
    print("concatenate merge time:" +
          str(time_concatenate2 - time_concatenate1))


# 2.用fromarray方法把np.array转换到PIL并保存，与concatenateSave耗时相当
def fromarraySave(ssListForVstack, file_name):
    time_fromarray1 = int(time.time())

    # print(dt)
    ss = np.concatenate((ssListForVstack[::-1]), axis=0)
    img = Image.fromarray(ss).convert('RGBA')
    img.save(file_name + 'fromarray.png')

    time_fromarray2 = int(time.time())
    print("fromarray merge time:" + str(time_fromarray2 - time_fromarray1))


# 3.用PIL方法先存再读，再合并，经测试，耗时太久。
def PIL_Save_Merge(ssListForVstack, file_name):
    # 用PIL方法先存再读，再合并
    time_PIL1 = int(time.time())
    img_arr = []
    toImage = Image.new('RGBA', (TileRange * 256, TileRange * 256))

    for i in range(len(ssListForVstack)):
        img_name = file_name + str(i) + '.png'
        plt.imsave(img_name, ssListForVstack[i])
        fromImge = Image.open(img_name)

        loc = (0, (len(ssListForVstack) - 1 - i) * threadSize * 256)
        toImage.paste(fromImge, loc)
        # 删除临时保存的文件
        os.remove(img_name)

    toImage.save(file_name + 'PILmerged.png')
    time_PIL2 = int(time.time())
    print("time_PIL merge time:" + str(time_PIL2 - time_PIL1))



def creat_file_name():
    '''由当前时间生成访问url的时间戳和要保存的文件名'''
    
    time_now = int(time.time())
    time_local = time.localtime(time_now)

    file_name = time.strftime("%Y-%m-%d_%H-%M-%S", time_local)
    timestamp = time_now * 1000.0

    return timestamp, file_name


# 多线程下载图片,传入要下载的地图的各个参数值
def Multi_Thread_DownLoad(timestamp, file_name, TitleRight, TitleTop,
                          TileRange, threadNum, mapLevel):                                                  #! 这个函数是在做什么，他没有返回值
    # 多线程下载图片
    # 保存各个线程的数组
    threadSize = TileRange // threadNum  # 每个线程中的tiles行/列个数 注意要是整数int，否则会报错
    ssList = []

    # 多线程下载任务启动
    time_download_start = int(time.time())
    print("download start:" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_download_start)))
    for i in range(threadNum):
        for j in range(threadNum):
            ssTemp = MyThread(timestamp, TitleRight - j * threadSize,
                              TitleTop - i * threadSize, threadSize,
                              threadSize, mapLevel)
            ssList.append(ssTemp)

    for item in ssList:
        item.start()    # start方法开启一个新线程。把需要并行处理的代码放在run()方法中，start()方法启动线程将自动调用run()方法。

    for item in ssList:
        item.join()     # 等待子进程执行结束，主进程再往下执行


    time_merge_start = int(time.time())
    print("download time:" + str(time_merge_start - time_download_start))
    print("merge start:" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_merge_start)))


    time_merge_stage1 = int(time.time())
    print("merge stage 1 finished:" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_merge_stage1)))
    print("merge stage 1 time:" + str(time_merge_stage1 - time_merge_start))


    time_finish = int(time.time())
    print("finish start:" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_finish)))
    print("merge time:" + str(time_finish - time_merge_start))



def downloadMain(levelParam=14):

    # 引用全局变量
    global resJsonData

    time_process_start = int(time.time())
    print("process start:" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_process_start)))

    # 由当前时间生成访问url的时间戳和要保存的文件名
    timestamp, file_name = creat_file_name()

    # 线程和任务的各个参数                                                                                   #! 这些参数是什么意思
    mapParameter_default14 = [3171, 1186, 16, 2, 14]
    # level = int(sys.argv[1])
    level = levelParam
    baselevel = 14
    levelup = level - baselevel
    TitleRight  = mapParameter_default14[0] * (2**levelup)  # 12684 25368
    TitleTop    = mapParameter_default14[1] * (2**levelup)  # 4744 9488
    TileRange   = mapParameter_default14[2] * (2**levelup)  # 64 128 # 每行/列tiles个数
    threadNum   = mapParameter_default14[3]                 # 8*8 64个线程并行
    mapLevel    = mapParameter_default14[4] + levelup       # 16 17 # 地图等级

    if (level <= 17):
        # 更新线程参数，不能整太多、太频繁，容易被封IP 线程过多会导致[Errno 104] Connection reset by peer
        if (level == 14):
            Multi_Thread_DownLoad(timestamp, file_name, TitleRight, TitleTop,
                                  TileRange, threadNum, mapLevel)
        if (level == 16):
            threadNum = 4
            Multi_Thread_DownLoad(timestamp, file_name, TitleRight, TitleTop,
                                  TileRange, threadNum, mapLevel)
        if (level == 17):
            threadNum = 4
            Multi_Thread_DownLoad(timestamp, file_name, TitleRight, TitleTop,
                                  TileRange, threadNum, mapLevel)


    resData = copy.copy(resJsonData)
    resJsonData.clear()
    jsonData = {'jsonName': file_name, 'data': resData}

    return jsonData
