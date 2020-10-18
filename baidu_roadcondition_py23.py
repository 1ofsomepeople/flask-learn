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

try:#python2
    import urllib2
    import cStringIO
except ImportError:#python3
    import urllib.request
    import io

import os
import signal
import matplotlib.pyplot as plt
import time
import numpy as np
import math
import sys
import json
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import threading
from threading import Thread
import lnglat_mercator_tiles_convertor as convertor

import requests
import random
import copy

from urllib.parse import urlparse, parse_qs, parse_qsl

import png2json as png2json


tilePoint = png2json.readpoints()
# print(tilePoint['31621178'])
resJsonData = []
# show tile pictures 256*256
def showBmap(my_url):
    size = 256

    # 随便选一个代理ip
    # proxy = urllib.request.ProxyHandler({'http': resip[0]})
    # opener = urllib.request.build_opener(proxy, urllib.request.HTTPHandler)
    # urllib.request.install_opener(opener)

    try:#python2
        file = cStringIO.StringIO(urllib2.urlopen(my_url).read())
    except NameError:#python3
        file = io.BytesIO(urllib.request.urlopen(my_url,timeout=3).read())
    try:
        img = Image.open(file)
        img = img.convert("RGB")
    except IOError:
#        print('fail to convert')
        return np.zeros((size,size))
    urlparam = parse_qs(urlparse(my_url).query)
    level = int(urlparam['level'][0])
    tile_x = int(urlparam['x'][0])
    tile_y = int(urlparam['y'][0])

    tileName = str(tile_x)+str(tile_y)

    if(tileName in tilePoint.keys()):
    #     # print(tileName)
        tempArr = tilePoint[tileName]
        for temppoint in tempArr:    
            if(temppoint[0] < 256 and temppoint[1] < 256):
                R,G,B = img.getpixel((temppoint[0],255-temppoint[1]))
                if(R+G+B>0):
                    # RGB归一化 消除光照影响
                    r = R / (R+G+B)
                    g = G / (R+G+B) 
                    b = 1 - r - g
                    value = png2json.RGB2Value(r,g,b)
                    if(value == 0):
                        value = 3
                    resJsonData.append([temppoint[2],temppoint[3],value])
    return img

# degree of longitude 经度 latitude 纬度    
# OpenStreetMap经纬度转行列号 不准确的 OpenStreetMap用的是WGS84 百度用的是BD09
# OpenStreetMap use WGS84, Baidu use BD09
def deg2num_pixel(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg) # 返回一个角度的弧度值
    n = 2.0 ** zoom # 乘方
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    xpixel = int(((lon_deg + 180.0) / 360.0 * n)*256%256 + 1/2)
    ypixel = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n  *256%256+1/2 )
    return (xtile, ytile ,xpixel,ypixel)
def real_coordinate(lat_deg,lon_deg,level):
    (xtile, ytile ,xpixel,ypixel) = deg2num_pixel(lat_deg, lon_deg, level)
    start_x = 3168
    start_y = 1183
    real_coordinate_x = (ytile - start_y) * size + ypixel
    real_coordinate_y = (xtile - start_x) * size + xpixel 
    return(real_coordinate_x,real_coordinate_y)

# 加密解密算法函数和参数
# encryption and decryption function and parameters
xu = 6370996.81
Sp = [1.289059486E7, 8362377.87, 5591021, 3481989.83, 1678043.12, 0]
Hj = [75, 60, 45, 30, 15, 0]
Au = [[1.410526172116255E-8, 8.98305509648872E-6, -1.9939833816331, 200.9824383106796, -187.2403703815547, 91.6087516669843, -23.38765649603339, 2.57121317296198, -0.03801003308653, 1.73379812E7], [ - 7.435856389565537E-9, 8.983055097726239E-6, -0.78625201886289, 96.32687599759846, -1.85204757529826, -59.36935905485877, 47.40033549296737, -16.50741931063887, 2.28786674699375, 1.026014486E7], [ - 3.030883460898826E-8, 8.98305509983578E-6, 0.30071316287616, 59.74293618442277, 7.357984074871, -25.38371002664745, 13.45380521110908, -3.29883767235584, 0.32710905363475, 6856817.37], [ - 1.981981304930552E-8, 8.983055099779535E-6, 0.03278182852591, 40.31678527705744, 0.65659298677277, -4.44255534477492, 0.85341911805263, 0.12923347998204, -0.04625736007561, 4482777.06], [3.09191371068437E-9, 8.983055096812155E-6, 6.995724062E-5, 23.10934304144901, -2.3663490511E-4, -0.6321817810242, -0.00663494467273, 0.03430082397953, -0.00466043876332, 2555164.4], [2.890871144776878E-9, 8.983055095805407E-6, -3.068298E-8, 7.47137025468032, -3.53937994E-6, -0.02145144861037, -1.234426596E-5, 1.0322952773E-4, -3.23890364E-6, 826088.5]]
Qp = [[- 0.0015702102444, 111320.7020616939, 1704480524535203, -10338987376042340, 26112667856603880, -35149669176653700, 26595700718403920, -10725012454188240, 1800819912950474, 82.5], [8.277824516172526E-4, 111320.7020463578, 6.477955746671607E8, -4.082003173641316E9, 1.077490566351142E10, -1.517187553151559E10, 1.205306533862167E10, -5.124939663577472E9, 9.133119359512032E8, 67.5], [0.00337398766765, 111320.7020202162, 4481351.045890365, -2.339375119931662E7, 7.968221547186455E7, -1.159649932797253E8, 9.723671115602145E7, -4.366194633752821E7, 8477230.501135234, 52.5], [0.00220636496208, 111320.7020209128, 51751.86112841131, 3796837.749470245, 992013.7397791013, -1221952.21711287, 1340652.697009075, -620943.6990984312, 144416.9293806241, 37.5], [ - 3.441963504368392E-4, 111320.7020576856, 278.2353980772752, 2485758.690035394, 6070.750963243378, 54821.18345352118, 9540.606633304236, -2710.55326746645, 1405.483844121726, 22.5], [ - 3.218135878613132E-4, 111320.7020701615, 0.00369383431289, 823725.6402795718, 0.46104986909093, 2351.343141331292, 1.58060784298199, 8.77738589078284, 0.37238884252424, 7.45]]

def Convertor(x,y,arr):
    pass
    x_temp = arr[0] + arr[1]*abs(x)
    y_temp = abs(y)/arr[9]
    y_temp1 = arr[2] + arr[3]*y_temp + arr[4]*y_temp*y_temp + arr[5]*y_temp*y_temp*y_temp + arr[6]*y_temp*y_temp*y_temp*y_temp + arr[7]*y_temp*y_temp*y_temp*y_temp*y_temp + arr[8]*y_temp*y_temp*y_temp*y_temp*y_temp*y_temp
    # c = c * (0 > a.lng ? -1 : 1) ternary operator of python like that:
    # python中的三元运算符写法：
    x_Converted = x_temp * (-1 if(0 > x) else 1)
    y_Converted = y_temp1 * (-1 if(0 > y) else 1)
    return[x_Converted,y_Converted]

# BD09经纬度和墨卡托投影的平面坐标的相互转换函数
# the convert function of longitude and latitude degree of BD09 and  plane coordinates of Mercator projection 
def BD092mercotor(lng_BD09,lat_BD09):
    arr = []
    lat_abs = abs(lat_BD09)
    for i in range(len(Hj)):
        if (lat_abs >= Hj[i]):
            arr = Qp[i]
            break
    res = Convertor(lng_BD09,lat_BD09,arr)
    pointX = round(res[0],1)
    pointY = round(res[1],1)
    return (pointX,pointY)
def mercator2BD09(pointX,pointY):
    arr = []
    pointYabs = abs(pointY)
    for i in range(len(Sp)):
        if (pointYabs >= Sp[i]):
            arr = Au[i]
            break
    res = Convertor(pointX,pointY,arr)
    lng_BD09 = round(res[0],6)
    lat_BD09 = round(res[1],6)
    return (lng_BD09,lat_BD09)

# 平面坐标（pointX, pointY）转瓦片坐标（tileX， tileY）和瓦片中的像素坐标（pixelX, pixelY）
# plane coordinates（pointX, pointY）convert to tile coordinates（tileX， tileY）and pixel coordinates（pixelX, pixelY）in tiles
def point2tiles_pixel(pointX,pointY,level):
    tileX = int(pointX * (2 ** (level - 18)) / 256)
    tileY = int(pointY * (2 ** (level - 18)) / 256)
    pixelX = int(pointX * (2 ** (level - 18)) - tileX * 256 + 0.5)
    pixelY = int(pointY * (2 ** (level - 18)) - tileY * 256 + 0.5)
    return(tileX,tileY,pixelX,pixelY)
# 瓦片（tileX， tileY）和像素坐标（pixelX, pixelY）转平面坐标（pointX, pointY）
# tile coordinates（tileX， tileY）and pixel coordinates（pixelX, pixelY）in tiles convert to plane coordinates（pointX, pointY）
def tiles_pixel2point(tileX, tileY, pixelX, pixelY, level):
    pointX = round(((tileX * 256 + pixelX) / (2 ** (level - 18))),1)
    pointY = round(((tileY * 256 + pixelY) / (2 ** (level - 18))),1)
    return(pointX,pointY) 
    

def down_a_map(time_now,start_x,start_y,x_range,y_range,level):

    # start_x = 25265
    # start_y = 9386
    # level = 17
      
    size = 256
    # x_range = 48
    # y_range = 48

    # tile_x,tile_y is (10,)
    tile_x =np.arange(start_x - x_range,start_x , 1)
    tile_y =np.arange(start_y - y_range,start_y , 1)
    # creat a 3 axis array. The size of myMap is (2560,2560,3), type is uint8
    myMap = np.zeros((len(tile_y)*size,len(tile_x)*size,3),dtype = 'uint8' )  
    address = 'http://its.map.baidu.com:8002/traffic/TrafficTileService?' # level=13&x=1580&y=589&time=1507687074027&v=081&smallflow=1&scaler=1
    
    # tiles start from (3158,1173) to (3167,1182), left_bottom to right_top
    # i,j is in 1~10
    for i  in range(len(tile_x)):
        # print(i)
        for j in range(len(tile_y)):
            # print(i,j)
            #url_ij = address+('x=%d&y=%d&z=%d')%(tile_x[i],tile_y[j],level)
            url_ij = address+('level=%d&x=%d&y=%d&time=%d')%(level,tile_x[i],tile_y[j],time_now)
            # print(url_ij)
            temp = showBmap(url_ij)
            # if url_ij can be convert to RGB, temp.size = (256,256)
            # print(temp.size)
            # If this tile is empty, there is no way in picture. 
            # temp = np.zeros((256,256)). temp.size=65536. Otherwise, temp.size=(256,256)
            if temp.size == 65536:
#                print('There is no way in this tile')
                temp_3 =np.zeros((256,256,3))
                temp_3[:,:,0] = temp
                temp_3[:,:,1] = temp
                temp_3[:,:,2] = temp
                temp = temp_3
            # 10*10 tiles convert to a png
            myMap[ (y_range-j-1)*size:(y_range-j)*size ,i*size:(i+1)*size   ,: ] = np.array(temp)
            # position = real_coordinate(tile_x[i],tile_y[j],level) 
            # print(position)
    return myMap


class MyThread(Thread):
    def __init__(self, time_now,start_x,start_y,x_range,y_range,level):
        Thread.__init__(self)
        self.time_now = time_now
        self.start_x = start_x
        self.start_y = start_y
        self.x_range = x_range
        self.y_range = y_range
        self.level = level

    def run(self):
        try_times = 3
        while(try_times>0):
            try:
                self.result = down_a_map(self.time_now,self.start_x,self.start_y,self.x_range,self.y_range,self.level)
                break
            except Exception as e:
                # print(e)
                time.sleep(2)
                try_times -= 1
                if(try_times>0):
                    # print("retry it:" + str(try_times))
                    pass
                else:
                    print("retry fail")
                    print(e)
                    os.kill(os.getpid(), signal.SIGKILL)

                    # pid_need_kill = os.getpid()
                    # instruction =  "/home/dingchaofan/anaconda3/bin/python baidu_roadcondition_py23.py " +str(self.level)+" Y "+str(pid_need_kill)
                    # os.system(instruction)
                    # print("kill process")
                    # os.kill(os.getpid(), signal.SIGKILL)

    def get_result(self):
        return self.result

# 三种merge stage2的保存方式：

# 1.用concatenate方法在np.array上进行合并并保存
def concatenateSave(ssListForVstack,file_name):

    # 用concatenate方法在np.array上进行合并并保存
    time_concatenate1 = int(time.time())
    # print(len(ssListForVstack))
    ss = np.concatenate((ssListForVstack[::-1]), axis=0)
    plt.imsave(file_name+'.png',ss)    

    time_concatenate2 = int(time.time())  
    print("concatenate merge time:" + str(time_concatenate2-time_concatenate1))


# 2.用fromarray方法把np.array转换到PIL并保存，与concatenateSave耗时相当
def fromarraySave(ssListForVstack,file_name):
    time_fromarray1 = int(time.time())

    # print(dt)
    ss = np.concatenate((ssListForVstack[::-1]), axis=0)
    img = Image.fromarray(ss).convert('RGBA')
    img.save(file_name+'fromarray.png')

    time_fromarray2 = int(time.time())
    print("fromarray merge time:" + str(time_fromarray2-time_fromarray1))


# 3.用PIL方法先存再读，再合并，经测试，耗时太久。
def PIL_Save_Merge(ssListForVstack,file_name):
    # 用PIL方法先存再读，再合并
    time_PIL1 = int(time.time())
    img_arr = []
    toImage = Image.new('RGBA',(TileRange*256,TileRange*256))

    for i in range(len(ssListForVstack)):
        img_name = file_name+str(i)+'.png'
        plt.imsave(img_name,ssListForVstack[i])
        fromImge = Image.open(img_name)

        loc = (0, (len(ssListForVstack)-1-i)*threadSize*256)
        toImage.paste(fromImge, loc)
        # 删除临时保存的文件       
        os.remove(img_name)

    toImage.save(file_name+'PILmerged.png')
    time_PIL2 = int(time.time())
    print("time_PIL merge time:" + str(time_PIL2-time_PIL1))

# 由当前时间生成访问url的时间戳和要保存的文件名
def creat_file_name():
    # 生成时间戳timestamp作为读取url的重要参数；size=256是每个瓦片的像素值，不可改变
    size = 256
    #address = 'http://its.map.baidu.com:8002/traffic/TrafficTileService?'#level=13&x=1580&y=589&time=1507687074027&v=081&smallflow=1&scaler=1
    #time_start = 1483200000 #Sun Jan 01 00:00:00 2017
    #time_start = 1472659200.0 #Thu Sep  1 00:00:00 2016
    #time_now = time_start
    time_now = int(time.time())
    time_local = time.localtime(time_now)
    dt = time.strftime("%Y-%m-%d %H:%M:%S",time_local)

    dt_new = dt.replace(' ','_' )
    dt_new = dt_new.replace(':','-' )
    print(dt_new)
    # convert to time array
    timeArray = time.strptime(dt, "%Y-%m-%d %H:%M:%S")
    # convert to timestamp
    timestamp = time.mktime(timeArray)
    timestamp = timestamp *1000

    print(timestamp)
    file_name = dt_new
    return timestamp,file_name

# 多线程下载图片,传入要下载的地图的各个参数值
def Multi_Thread_DownLoad(timestamp,file_name,TitleRight,TitleTop,TileRange,threadNum,mapLevel):
    # 多线程下载图片
    # 保存各个线程的数组
    threadSize = TileRange//threadNum # 每个线程中的tiles行/列个数 注意要是整数int，否则会报错
    ssList = []

    # 多线程下载任务启动
    time_download_start = int(time.time())
    print("download start:" + time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time_download_start)))
    for i in range(threadNum):
        for j in range(threadNum):
            ssTemp = MyThread(timestamp,TitleRight-j*threadSize,TitleTop-i*threadSize,threadSize,threadSize,mapLevel)
            ssList.append(ssTemp)
            
    for index in range(len(ssList)):
        ssList[index].start()
        
    for index in range(len(ssList)):
        ssList[index].join()


    time_merge_start = int(time.time())
    print("download time:" + str(time_merge_start-time_download_start))
    print("merge start:" + time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time_merge_start)))

    ssListForhstack = []
    ssListForVstack = []

    ssListForhstack.append(ssList[-1].get_result())

    for index in range(len(ssList)-2,-1,-1):
        if(index%threadNum == threadNum-1):
            ssListForVstack.append(np.concatenate((ssListForhstack[:]), axis=1))
            ssListForhstack = []
            ssListForhstack.append(ssList[index].get_result())
        else:
            ssListForhstack.append(ssList[index].get_result())
    ssListForVstack.append(np.concatenate((ssListForhstack[:]), axis=1))
    # temp = np.hstack((temp, ssList[index-i].get_result()))
    # temp = np.concatenate((temp, ssList[index-i].get_result()), axis=1)
    # 用concatenate优化vstack和hstack，可以加速矩阵的合并操作
    # np.vstack((a,a))                # 0.000063
    # np.concatenate((a,a), axis=0)   # 0.000040

    time_merge_stage1 = int(time.time())
    print("merge stage 1 finished:" + time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time_merge_stage1)))
    print("merge stage 1 time:" + str(time_merge_stage1-time_merge_start))

    concatenateSave(ssListForVstack,file_name)

    time_finish = int(time.time())
    print("finish start:" + time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time_finish)))
    print("merge time:" + str(time_finish-time_merge_start))
'''
#coding:UTF-8
import time
#获取当前时间
time_now = int(time.time())
#转换成localtime
time_local = time.localtime(time_now)
#转换成新的时间格式(2016-05-09 18:59:20)
dt = time.strftime("%Y-%m-%d %H:%M:%S",time_local)

#转换成时间数组
timeArray = time.strptime(dt, "%Y-%m-%d %H:%M:%S")
#转换成时间戳
timestamp = time.mktime(timeArray)

'''
# sys.argv[0] 文件的路径什么的，本身就有的，不在输入的参数里
# sys.argv[1] 图片等级

'''
# 弃置
# sys.argv[2] 是否结束进程 Y/N
# sys.argv[3] 之前进程的pid
# 给定一个标志位，是否kill之前的进程
# if(sys.argv[2] == 'Y'):
#     # 结束之前的进程，如果有的话
#     os.kill(int(sys.argv[3]), signal.SIGKILL)
#     print("kill a old process")
# else:
#     print("First time to excute this process")
'''
# targetUrl = "https://proxy.horocn.com/api/v2/proxies?order_id=RJ9Y1648150289620460&num=1&format=text&line_separator=win&can_repeat=no"
# resp = requests.get(targetUrl)
# # print(resp.text)
# resip = resp.text.split('\r\n')
# print(resip[0])

def downloadMain(levelParam=14):
    time_process_start = int(time.time())
    print("process start:" + time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time_process_start)))

    # 由当前时间生成访问url的时间戳和要保存的文件名
    timestamp,file_name = creat_file_name()

    # 线程和任务的各个参数
    mapParameter_default14 = [3171,1186,16,2,14]
    # level = int(sys.argv[1])
    level = levelParam 
    baselevel = 14
    levelup = level - baselevel
    TitleRight = mapParameter_default14[0] * (2**levelup)#12684 25368
    TitleTop = mapParameter_default14[1] * (2**levelup)# 4744 9488
    TileRange = mapParameter_default14[2] * (2**levelup)# 64 128 # 每行/列tiles个数
    threadNum = mapParameter_default14[3] # * (2**levelup)# 32 64 # 8*8 64个线程并行
    mapLevel = mapParameter_default14[4] + levelup# 16 17 # 地图等级


    if(level <= 17):
        # 更新线程参数，不能整太多、太频繁，容易被封IP 线程过多会导致[Errno 104] Connection reset by peer
        if(level == 14):
            Multi_Thread_DownLoad(timestamp,file_name,TitleRight,TitleTop,TileRange,threadNum,mapLevel)
        if(level == 16):
            threadNum = 4
            Multi_Thread_DownLoad(timestamp,file_name,TitleRight,TitleTop,TileRange,threadNum,mapLevel)
        if(level == 17):
            threadNum = 4
            Multi_Thread_DownLoad(timestamp,file_name,TitleRight,TitleTop,TileRange,threadNum,mapLevel)
    
    # print(resJsonData)
    print(len(resJsonData))
    # num = 0
    # for item in resJsonData:
    #     if(item[2]>0):
    #         num+=1
    # print(num)
    resData = copy.copy(resJsonData)
    resJsonData.clear()
    jsonData = {
        'jsonName':file_name,
        'data':resData
    }
    
    return jsonData

    

