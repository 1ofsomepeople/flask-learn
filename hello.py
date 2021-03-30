import json
import os
import sys
import time

from flask import Flask, jsonify, render_template, request, url_for
from flask_apscheduler import APScheduler  # 引入APScheduler
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy  # 导入扩展类

import baidu_roadcondition_py23 as download
import data.dataProcess as dataProcess
import dataCsv as datacsv


#任务配置类
class SchedulerConfig(object):
    JOBS = [{
        'id': 'download_interval_job',              # 任务id
        'func': '__main__:download_interval_job',   # 任务执行程序，main函数中的download_interval_job()函数
        'args': None,                               # 传入执行程序的参数
        'trigger': 'interval',                      # 任务执行类型，定时器。通过设置"时间间隔"来运行定时任务
        'seconds': 300,                             # 任务执行间隔，单位秒。每隔300s执行
    }]


def download_interval_job():
    '''定义任务执行程序，功能是更新./data.csv文件.

    '''

    time_now = int(time.time())
    time_local = time.localtime(time_now)
    dt = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
    print(dt, "I'm a scheduler!")
    dataObj = download.downloadMain()
    name = dataObj['jsonName']
    data = dataObj['data']
    datacsv.appendData(name, data)

 
def request_parse(req_data):
    '''解析请求url的参数.

    '''

    # 解析请求数据并以json形式返回
    if (req_data.method) == 'POST':
        data = req_data.json
    elif (req_data.method) == 'GET':
        data = req_data.args
    return data

app = Flask(__name__)                       # 实例化flask
CORS(app, supports_credentials=True)        # 解决跨域问题
app.config.from_object(SchedulerConfig())   # 为实例化的flask引入配置
@app.route("/")
def hello():
    return "Hello World1!"


@app.route('/data/')
def getRealTimeData():
    '''加载实时拥堵数据.

    '''

    dataObj = download.downloadMain()
    return jsonify(dataObj)


@app.route('/data/predict/lr/')
def getPredictDataLr():
    '''由dataCsv中的数据经过lr模型预测生成预测拥堵数据.

    '''

    dataObj = datacsv.getPred('lr')
    return jsonify(dataObj)


@app.route('/data/predict/sage/')
def getPredictDataSage():
    '''由dataCsv中的数据经过Sage模型预测生成预测拥堵数据.

    '''

    dataObj = datacsv.getPred('sage')
    return jsonify(dataObj)


@app.route('/data/history/gt/', methods=["GET", "POST"])
def getHistoryDataGt():

    # 解析接口url的参数
    paramData = request_parse(request)
    dataIndex = int(paramData.get("dataIndex"))
    dataObj = dataProcess.getGtData(dataIndex, 'data.csv')
    return jsonify(dataObj)


@app.route('/data/history/pred/', methods=["GET", "POST"])  # GET 和 POST 都可以
def getHistoryDataPred():

    # 解析接口url的参数
    paramData = request_parse(request)
    # 假设有如下 URL
    # http://10.8.54.48:5000/index?name=john&age=20
    # name = data.get("name")
    # age = data.get("age")
    dataIndex = int(paramData.get("dataIndex"))
    predictType = str(paramData.get("predictType"))
    dataObj = dataProcess.getPredData(dataIndex, predictType, 'data.csv')
    return jsonify(dataObj)


if __name__ == "__main__":
    scheduler = APScheduler()                   # 实例化APScheduler
    scheduler.init_app(app)                     # 把任务列表载入实例flask
    scheduler.start()                           # 启动任务列表
    app.run(use_reloader=False, debug=False)    # 启动flask
