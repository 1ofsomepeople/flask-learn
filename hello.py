import os
import sys

from flask import Flask, render_template, url_for, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy # 导入扩展类
import baidu_roadcondition_py23 as download


app = Flask(__name__)
CORS(app, supports_credentials=True) 
 
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
    # print(name)
    # for root, dirs, files in os.walk(os.getcwd()): 
    #     print(root) #当前目录路径 
    #     print(files) #当前路径下所有非目录子文件
    
    return jsonify(dataObj)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)