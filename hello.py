import os
import sys

from flask import Flask, render_template, url_for, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy # 导入扩展类
import baidu_roadcondition_py23 as download
import dataCsv as datacsv

from flask_apscheduler import APScheduler # 引入APScheduler
import time
import json

#任务配置类
class SchedulerConfig(object):
    JOBS = [
        {
            'id': 'download_interval_job', # 任务id
            'func': '__main__:download_interval_job', # 任务执行程序
            'args': None, # 执行程序参数
            'trigger': 'interval', # 任务执行类型，定时器
            'seconds': 300, # 任务执行时间，单位秒
        }
    ]

#定义任务执行程序
def download_interval_job():
    time_now = int(time.time())
    time_local = time.localtime(time_now)
    dt = time.strftime("%Y-%m-%d %H:%M:%S",time_local)
    print(dt, "I'm a scheduler!")
    dataObj = download.downloadMain()
    name = dataObj['jsonName']
    data = dataObj['data']
    datacsv.appendData(name,data)
    
app = Flask(__name__)
CORS(app, supports_credentials=True)  # 解决跨域问题

#为实例化的flask引入定时任务配置
app.config.from_object(SchedulerConfig())

@app.route("/")
def hello():
    return "Hello World1!"

@app.route('/data/')
def getRealTimeData():

    dataObj = {}
    for file in os.listdir():
        if(file.endswith(".png")):
            file_path = os.path.join(os.getcwd(), file)
            os.remove(file_path)

    dataObj = download.downloadMain()
    # print('res'+str(len(dataObj['data'])))

    return jsonify(dataObj)

if __name__ == "__main__":
    scheduler = APScheduler()  # 实例化APScheduler
    scheduler.init_app(app)  # 把任务列表载入实例flask
    scheduler.start()  # 启动任务计划
    # 在调试模式下，Flask的重新加载器将加载烧瓶应用程序两次。因此flask总共有两个进程. 重新加载器监视文件系统的更改并在不同的进程中启动真实应用程序
    # 有几种方法可以解决这个问题。我发现效果最好的是禁用重新加载器：app.run(use_reloader=False) 或者关闭调试模式debug mod
    # 解决定时器程序，一个interval运行两次的bug
    app.run(host='0.0.0.0', port=5000, use_reloader=False)