import os
import sys

from flask import Flask, render_template, url_for, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy # 导入扩展类
import baidu_roadcondition_py23 as download

from flask_apscheduler import APScheduler # 引入APScheduler
import time
import json

app = Flask(__name__)
CORS(app, supports_credentials=True)  # 解决跨域问题


def readtempdata():
    jsonName = ''
    data = []
    with open('./tempdata.json', 'r') as f:
        # jsonName = json.load(f).get('jsonName')
        datajson = json.load(f)
    jsonName = datajson['jsonName']
    data = datajson['data']
    return (jsonName,data)
# 加载测试数据
(testjsonName,testdata) = readtempdata()
print(testjsonName,len(testdata))

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
    print('res'+str(len(dataObj['data'])))

    # 写入tempdata.json
    # with open("./tempdata.json", "w") as f:
    #     f.write(json.dumps(dataObj))
    return jsonify(dataObj)

#任务配置类
class SchedulerConfig(object):
    JOBS = [
        {
            'id': 'download_interval_job', # 任务id
            'func': '__main__:download_interval_job', # 任务执行程序
            'args': None, # 执行程序参数
            'trigger': 'interval', # 任务执行类型，定时器
            'seconds': 10, # 任务执行时间，单位秒
        }
    ]
#定义任务执行程序
def download_interval_job():
    time_now = int(time.time())
    time_local = time.localtime(time_now)
    dt = time.strftime("%Y-%m-%d %H:%M:%S",time_local)
    print(dt, "I'm a scheduler!")


#为实例化的flask引入定时任务配置
app.config.from_object(SchedulerConfig())



if __name__ == "__main__":
    scheduler = APScheduler()  # 实例化APScheduler
    scheduler.init_app(app)  # 把任务列表载入实例flask
    scheduler.start()  # 启动任务计划
    app.run(host='0.0.0.0', port=5000)