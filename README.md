# 1. 安装虚拟环境
$ pip install virtualenv
首先，安装python虚拟环境
$ cd my_project_dir
$ virtualenv flask-env # falsk-env为虚拟环境目录名，目录名自定义

# 2. 激活虚拟环境
- windows
  - cd flask-env/Scripts
  - activate 激活（需要在cmd）
  - deactivate  停用

- linux/mac
  - source flask-env/bin/activate 激活
  - deactivate 停用

要删除一个虚拟环境，只需删除它的文件夹。rm -rf falsk-env 

# 3. 导出/安装依赖包
requirements.txt用来记录项目所有的依赖包和版本号
pip freeze >requirements.txt
将项目依赖导入
pip install -r requirements.txt
安装项目依赖

pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
安装 pytorch==1.6.0 torchvision==0.7.0 cpu版


# 4. 项目运行和查看
https://www.cnblogs.com/baby123/p/6477429.html
nohup后台运行 设置log输出的文件
nohup python hello.py > ~/project/trafficVis_RD/output.log 2>&1 &
项目进程查看
jobs -l 
ps -aux|grep python 
