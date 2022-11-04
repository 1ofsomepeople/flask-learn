import numpy as np


def metrics(pred, real):
  mae = round(np.abs(pred - real).mean(), 4)
  rmse = round(np.sqrt(np.square(pred-real).mean()), 4)
  return mae, rmse

class MyScaler():
  def __init__(self, max_val) -> None:
    self.max_val = max_val

  def transform(self, data):
    return np.log(data + 1.0)/self.max_val

  def inverse_transform(self, data):
    return np.exp(data * self.max_val) - 1.0


def generate_fake_data():
  source = np.load('data/bus_testdata.npy')
  N, _, t = source.shape
  idx = np.random.randint(low=0, high=t)
  #TODO: 归一化显示
  source = source[:, :, idx]
  # max_val = np.log(1.0 +source.max()) # 当前数据的最大值
  # scaler = MyScaler(max_val)
  # source = scaler.transform(source)
  source = source.tolist()
  data = []
  for i in range(N):
    for j in range(N):
      data.append([i, j, source[i][j]])
  predictTime = "2019-04-02_07-30"
  resObj = {
    "jsonName":predictTime,
    "data":data
  }
  return resObj


def get_his_data(time_index):
  """
    input:
      time_index: 要展示的时刻
    output:
      json: 返回的数据
  """
  source = np.load('data/test_for_vis_data.npy')
  N, _, t = source.shape
  idx = time_index
  source = source[:, :, 2*idx+5] # -32 是最后一天的8:00AM的索引
  # max_val = np.log(1.0 + source.max()) #FIXME: 归一化使用的是当前数据的最大值
  # scaler = MyScaler(max_val)
  # source = scaler.transform(source)
  source = source.tolist()
  data = []
  for i in range(N):
    for j in range(N):
      data.append([i, j, source[i][j]])
  predictTime = "2019-04-02_07-30" #FIXME: 修改时间
  resObj = {
    "jsonName":predictTime,
    "data":data
  }
  return resObj


def get_pred_data(time_index, model):
  """
    input:
      time_index: 要预测的时刻
      model: 预测模型
    output:
      json: 返回的数据
  """
  pred_data = np.load('data/od_pred/bus_30/'+ model + '_vis.npy')
  real_data = np.load('data/test_for_vis_data.npy')
  pred_data = pred_data.squeeze()
  t, N, _ = pred_data.shape
  idx = time_index
  pred = pred_data[2*idx]
  real = real_data[:, :, 2*idx + 5]
  mae, rmse = metrics(pred, real)
  # max_val = np.log(1.0 + source.max()) #FIXME: 归一化使用的是当前数据的最大值
  # scaler = MyScaler(max_val)
  # source = scaler.transform(source)
  pred = pred.tolist()
  data = []
  for i in range(N):
    for j in range(N):
      data.append([i, j, pred[i][j]])
  predictTime = "2019-04-02_07-30" #FIXME: 修改时间
  resObj = {
    "jsonName":predictTime,
    "data":data,
    "scoreMAE":mae,
    "scoreRMSE":rmse,
  }
  return resObj