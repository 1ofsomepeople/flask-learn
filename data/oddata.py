import numpy as np

STATION_DIC = {
  # 23:[116.503107, 39.913065],
  # 22:[116.482481, 39.914231],
  # # '郎家园':[116.476708, 39.914008],
  # 21:[116.471338, 39.914263],
  # 20:[116.464776, 39.913809],
  # 19:[116.454475, 39.914213],
  # 18:[116.449533, 39.914233],
  # 17:[116.436053, 39.914403],
  # 16:[116.422261, 39.914435],
  # 15:[116.407762, 39.913766],
  # 14:[116.398734, 39.913786],
  # 13:[116.384392, 39.913298],
  # # '复兴门内':[116.365456, 39.913015],
  # 12:[116.35699, 39.912949],
  # 11:[116.352452, 39.912947],
  # 10:[116.339085, 39.913174],
  # 9:[116.33088, 39.91328],
  # 8:[116.316968, 39.913395],
  # # '翠微路口':[116.307421, 39.913658],
  # 7:[116.299788, 39.913756],
  # # '东翠路口':[116.294197, 39.913797],
  # 6:[116.288463, 39.913804],
  # 5:[116.284029, 39.913811],
  # 4:[116.272447, 39.913714],
  # 3:[116.257966, 39.913448],
  # 2:[116.242127, 39.913169],
  # 1:[116.240281, 39.918174],
  # 0:[116.239077, 39.921391],
  23:(116.4906351876019, 39.90571750227382),
  22:(116.46981229421405, 39.90717615725094),
  21:(116.45859573583468, 39.90728289609876),
  20:(116.45200969873245, 39.906822723301026),
  19:(116.441703102958, 39.90713935089575),
  18:(116.43677048654843, 39.90708818243652),
  17:(116.42334274544173, 39.90700982647987),
  16:(116.40961252669423, 39.90680074457834),
  15:(116.39514518220952, 39.90601615220736),
  14:(116.38610976246815, 39.90606611105301),
  13:(116.37171736434773, 39.905768842235425),
  12:(116.34421093994041, 39.90589615371753),
  11:(116.33967074021018, 39.90593668234911),
  10:(116.32633702300372, 39.90617947724279),
  9:(116.3181798079248, 39.90621678562706),
  8:(116.30437746724007, 39.906127604467486),
  7:(116.2873252094815, 39.90624058712257),
  6:(116.27604392498202, 39.90622430603992),
  5:(116.2716146501648, 39.90623722844211),
  4:(116.26001177549975, 39.906237122676316),
  3:(116.24545331669, 39.90620798416575),
  2:(116.22951343694095, 39.90619045973707),
  1:(116.22766114192765, 39.911219155500675),
  0:(116.2264529702029, 39.914450096247954)
}
STATION_CODE = {
  '老山公交场站' : 0,
  '老山南路东口' : 1,
  '地铁八宝山站' : 2,
  '玉泉路口西' : 3,
  '永定路口东' : 4,
  '五棵松桥东' : 5,
  '沙沟路口西' : 6,
  '东翠路口' : 6,
  '万寿路口西' : 7,
  '翠微路口' : 7,
  '公主坟' : 8,
  '军事博物馆' : 9,
  '木樨地西' : 10,
  '工会大楼' : 11,
  '南礼士路' : 12,
  '复兴门内' : 12,
  '西单路口东' : 13,
  '天安门西' : 14,
  '天安门东' : 15,
  '东单路口西' : 16,
  '北京站口东' : 17,
  '日坛路' : 18,
  '永安里路口西' : 19,
  '大北窑西' : 20,
  '大北窑东' : 21,
  '郎家园' : 21,
  '八王坟西' : 22,
  '四惠枢纽站' : 23
}
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
      data.append([STATION_DIC[j][0], STATION_DIC[j][1], source[i][j]]) #NOTE: 记录从i站出发到j站的流量
  predictTime = "2019-04-02_07-30"
  resObj = {
    "jsonName":predictTime,
    "data":data
  }
  return resObj


def get_his_data(time_index, predStation):
  """
    input:
      time_index: 要展示的时刻
      predStation: 预测站点
    output:
      resObj: 返回的数据(json格式)
  """
  source = np.load('data/test_for_vis_data.npy')
  N, _, t = source.shape
  idx = time_index
  source = source[:, :, 2*idx+5] # -32 是最后一天的8:00AM的索引
  # max_val = np.log(1.0 + source.max()) #FIXME: 归一化使用的是当前数据的最大值
  # scaler = MyScaler(max_val)
  # source = scaler.transform(source)
  source = source.tolist()
  data1D = []
  data3D = []
  print(predStation)
  origin = STATION_CODE[predStation]
  for destination in range(N):
    data1D.append(source[origin][destination])
  for i in range(N):
    for j in range(N):
      data3D.append([i, j, source[i][j]])
  predictTime = "2016/06/29 08:00:00" #FIXME: 修改时间
  resObj = {
    "jsonName": predictTime,
    "data": {'data1D':data1D, 'data3D':data3D}
  }
  return resObj


def get_pred_data(time_index, model, predStation):
  """
    input:
      time_index: 要预测的时刻
      model: 预测模型
      predStation: 预测站点
    output:
      resObj: 返回的数据(json格式)
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
  data1D = []
  data3D = []
  origin = STATION_CODE[predStation]
  for destination in range(N):
    data1D.append(pred[origin][destination])
  for i in range(N):
    for j in range(N):
      data3D.append([i, j, pred[i][j]])
  predictTime = "2016/06/29 08:00:00" #FIXME: 修改时间
  resObj = {
    "jsonName": predictTime,
    "data": {'data1D':data1D, 'data3D':data3D},
    "scoreMAE": mae,
    "scoreRMSE": rmse,
  }
  return resObj