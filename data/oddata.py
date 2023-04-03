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
STATION_NAME2CODE = {
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
STATION_CODE2NAME = {
  0:'老山公交场站',
  1:'老山南路东口',
  2:'地铁八宝山站',
  3:'玉泉路口西',
  4:'永定路口东',
  5:'五棵松桥东',
  6:'沙沟路口西',
  7:'万寿路口西',
  8:'公主坟',
  9:'军事博物馆',
  10:'木樨地西',
  11:'工会大楼',
  12:'南礼士路',
  13:'西单路口东',
  14:'天安门西',
  15:'天安门东',
  16:'东单路口西',
  17:'北京站口东',
  18:'日坛路',
  19:'永安里路口西',
  20:'大北窑西',
  21:'大北窑东',
  22:'八王坟西',
  23:'四惠枢纽站'
}
TIME_CODE2NAME = {
  0:'08:00',
  1:'08:30',
  2:'09:00',
  3:'09:30',
  4:'10:00',
  5:'10:30',
  6:'11:00',
  7:'11:30',
  8:'12:00',
  9:'12:30',
  10:'13:00',
  11:'13:30',
  12:'14:00',
  13:'14:30',
  14:'15:00',
  15:'15:30',
  16:'16:00',
  17:'16:30',
  18:'17:00',
  19:'17:30',
  20:'18:00',
  21:'18:30',
  22:'19:00',
  23:'19:30',
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
  # print(source.shape)
  origin = STATION_NAME2CODE[predStation]
  idx = time_index
  od_timeview_o = source[origin, :, 5:29] # -32 是最后一天的8:00AM的索引
  od_timeview_d = source[:, origin, 5:29] # -32 是最后一天的8:00AM的索引
  od_spatialview = source[:, :, idx+5] # -32 是最后一天的8:00AM的索引
  # max_val = np.log(1.0 + source.max()) #FIXME: 归一化使用的是当前数据的最大值
  # scaler = MyScaler(max_val)
  # source = scaler.transform(source)
  od_timeview_o = od_timeview_o.tolist()
  od_timeview_d = od_timeview_d.tolist()
  od_spatialview = od_spatialview.tolist()
  data1D_o = []
  data1D_d = []
  data3D = []
  data_map = []
  data_timeview_o = []
  data_timeview_d = []
  # print(predStation)
  for destination in range(N):
    data1D_o.append(od_spatialview[origin][destination])
    data1D_d.append(od_spatialview[destination][origin])
    data_map.append([STATION_DIC[destination][0], STATION_DIC[destination][1], od_spatialview[origin][destination]*30]) # 加入坐标信息的od信息
  for i in range(N):
    for j in range(N):
      data3D.append([STATION_CODE2NAME[i], STATION_CODE2NAME[j], od_spatialview[i][j]])
  for i in range(N):
    for j in range(24):
      data_timeview_o.append([STATION_CODE2NAME[i], TIME_CODE2NAME[j], od_timeview_o[i][j]])
      data_timeview_d.append([STATION_CODE2NAME[i], TIME_CODE2NAME[j], od_timeview_d[i][j]])
  predictTime = TIME_CODE2NAME[idx] #FIXME: 修改时间
  # print(data_timeview_o)
  resObj = {
    "jsonName": predictTime,
    "data": {'data1D_o':data1D_o, 'data1D_d':data1D_d, 'data_map':data_map, 'data3D':data3D, 'data_timeview_o':data_timeview_o, 'data_timeview_d':data_timeview_d},
    "pred_station":predStation
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
  print(pred_data.shape)
  idx = time_index
  pred = pred_data[idx]
  real = real_data[:, :, idx+5]
  mae, rmse = metrics(pred, real)
  # max_val = np.log(1.0 + source.max()) #FIXME: 归一化使用的是当前数据的最大值
  # scaler = MyScaler(max_val)
  # source = scaler.transform(source)
  pred = pred.tolist()
  data1D_o = []
  data3D = []
  data_map = []
  origin = STATION_NAME2CODE[predStation]
  for destination in range(N):
    data1D_o.append(pred[origin][destination])
    data_map.append([STATION_DIC[destination][0], STATION_DIC[destination][1], pred[origin][destination]]) # 加入坐标信息的od信息
  for i in range(N):
    for j in range(N):
      data3D.append([STATION_CODE2NAME[i], STATION_CODE2NAME[j], pred[i][j]])
  predictTime = TIME_CODE2NAME[idx]
  resObj = {
    "jsonName": predictTime,
    "data": {'data1D_o':data1D_o, 'data_map':data_map, 'data3D':data3D},
    "scoreMAE": mae,
    "scoreRMSE": rmse,
  }
  return resObj