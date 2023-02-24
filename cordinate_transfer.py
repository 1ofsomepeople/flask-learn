import math

class LngLatTransfer():

  def __init__(self):
    self.x_pi = 3.14159265358979324 * 3000.0 / 180.0
    self.pi = math.pi  # π
    self.a = 6378245.0  # 长半轴
    self.es = 0.00669342162296594323  # 偏心率平方
    pass

  def GCJ02_to_BD09(self, gcj_lng, gcj_lat):
    """
    实现GCJ02向BD09坐标系的转换
    :param lng: GCJ02坐标系下的经度
    :param lat: GCJ02坐标系下的纬度
    :return: 转换后的BD09下经纬度
    """
    z = math.sqrt(gcj_lng * gcj_lng + gcj_lat * gcj_lat) + 0.00002 * math.sin(gcj_lat * self.x_pi)
    theta = math.atan2(gcj_lat, gcj_lng) + 0.000003 * math.cos(gcj_lng * self.x_pi)
    bd_lng = z * math.cos(theta) + 0.0065
    bd_lat = z * math.sin(theta) + 0.006
    return bd_lng, bd_lat


  def BD09_to_GCJ02(self, bd_lng, bd_lat):
    '''
    实现BD09坐标系向GCJ02坐标系的转换
    :param bd_lng: BD09坐标系下的经度
    :param bd_lat: BD09坐标系下的纬度
    :return: 转换后的GCJ02下经纬度
    '''
    x = bd_lng - 0.0065
    y = bd_lat - 0.006
    z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * self.x_pi)
    theta = math.atan2(y, x) - 0.000003 * math.cos(x * self.x_pi)
    gcj_lng = z * math.cos(theta)
    gcj_lat = z * math.sin(theta)
    return gcj_lng, gcj_lat


  def WGS84_to_GCJ02(self, lng, lat):
    '''
    实现WGS84坐标系向GCJ02坐标系的转换
    :param lng: WGS84坐标系下的经度
    :param lat: WGS84坐标系下的纬度
    :return: 转换后的GCJ02下经纬度
    '''
    dlat = self._transformlat(lng - 105.0, lat - 35.0)
    dlng = self._transformlng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * self.pi
    magic = math.sin(radlat)
    magic = 1 - self.es * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((self.a * (1 - self.es)) / (magic * sqrtmagic) * self.pi)
    dlng = (dlng * 180.0) / (self.a / sqrtmagic * math.cos(radlat) * self.pi)
    gcj_lng = lat + dlat
    gcj_lat = lng + dlng
    return gcj_lng, gcj_lat


  def GCJ02_to_WGS84(self, gcj_lng, gcj_lat):
    '''
    实现GCJ02坐标系向WGS84坐标系的转换
    :param gcj_lng: GCJ02坐标系下的经度
    :param gcj_lat: GCJ02坐标系下的纬度
    :return: 转换后的WGS84下经纬度
    '''
    dlat = self._transformlat(gcj_lng - 105.0, gcj_lat - 35.0)
    dlng = self._transformlng(gcj_lng - 105.0, gcj_lat - 35.0)
    radlat = gcj_lat / 180.0 * self.pi
    magic = math.sin(radlat)
    magic = 1 - self.es * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((self.a * (1 - self.es)) / (magic * sqrtmagic) * self.pi)
    dlng = (dlng * 180.0) / (self.a / sqrtmagic * math.cos(radlat) * self.pi)
    mglat = gcj_lat + dlat
    mglng = gcj_lng + dlng
    lng = gcj_lng * 2 - mglng
    lat = gcj_lat * 2 - mglat
    return lng, lat


  def BD09_to_WGS84(self, bd_lng, bd_lat):
    '''
    实现BD09坐标系向WGS84坐标系的转换
    :param bd_lng: BD09坐标系下的经度
    :param bd_lat: BD09坐标系下的纬度
    :return: 转换后的WGS84下经纬度
    '''
    lng, lat = self.BD09_to_GCJ02(bd_lng, bd_lat)
    return self.GCJ02_to_WGS84(lng, lat)


  def WGS84_to_BD09(self, lng, lat):
    '''
    实现WGS84坐标系向BD09坐标系的转换
    :param lng: WGS84坐标系下的经度
    :param lat: WGS84坐标系下的纬度
    :return: 转换后的BD09下经纬度
    '''
    lng, lat = self.WGS84_to_GCJ02(lng, lat)
    return self.GCJ02_to_BD09(lng, lat)


  def _transformlat(self, lng, lat):
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
          0.1 * lng * lat + 0.2 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * self.pi) + 20.0 *
            math.sin(2.0 * lng * self.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * self.pi) + 40.0 *
            math.sin(lat / 3.0 * self.pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * self.pi) + 320 *
            math.sin(lat * self.pi / 30.0)) * 2.0 / 3.0
    return ret


  def _transformlng(self, lng, lat):
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
          0.1 * lng * lat + 0.1 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * self.pi) + 20.0 *
            math.sin(2.0 * lng * self.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lng * self.pi) + 40.0 *
            math.sin(lng / 3.0 * self.pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lng / 12.0 * self.pi) + 300.0 *
            math.sin(lng / 30.0 * self.pi)) * 2.0 / 3.0
    return ret

  def WGS84_to_WebMercator(self, lng, lat):
    '''
    实现WGS84向web墨卡托的转换
    :param lng: WGS84经度
    :param lat: WGS84纬度
    :return: 转换后的web墨卡托坐标
    '''
    x = lng * 20037508.342789 / 180
    y = math.log(math.tan((90 + lat) * self.pi / 360)) / (self.pi / 180)
    y = y * 20037508.34789 / 180
    return x, y

  def WebMercator_to_WGS84(self, x, y):
    '''
    实现web墨卡托向WGS84的转换
    :param x: web墨卡托x坐标
    :param y: web墨卡托y坐标
    :return: 转换后的WGS84经纬度
    '''
    lng = x / 20037508.34 * 180
    lat = y / 20037508.34 * 180
    lat = 180 / self.pi * (2 * math.atan(math.exp(lat * self.pi / 180)) - self.pi / 2)
    return lng, lat

cordinate_list = [
  # [116.503107, 39.913065],
  # [116.482481, 39.914231],
  [116.476708, 39.914008],
  # [116.471338, 39.914263],
  # [116.464776, 39.913809],
  # [116.454475, 39.914213],
  # [116.449533, 39.914233],
  # [116.436053, 39.914403],
  # [116.422261, 39.914435],
  # [116.407762, 39.913766],
  # [116.398734, 39.913786],
  # [116.384392, 39.913298],
  [116.365456, 39.913015],
  # [116.35699, 39.912949],
  # [116.352452, 39.912947],
  # [116.339085, 39.913174],
  # [116.33088, 39.91328],
  # [116.316968, 39.913395],
  [116.307421, 39.913658],
  # [116.299788, 39.913756],
  [116.294197, 39.913797],
  # [116.288463, 39.913804],
  # [116.284029, 39.913811],
  # [116.272447, 39.913714],
  # [116.257966, 39.913448],
  # [116.242127, 39.913169],
  # [116.240281, 39.918174],
  # [116.239077, 39.921391]
]
answer_wgs84 = []
answer_webmercator = []
transfer = LngLatTransfer()
for cordinate in cordinate_list:
  answer_wgs84.append(transfer.BD09_to_WGS84(cordinate[0], cordinate[1]))
print(answer_wgs84)

[(116.4906351876019, 39.90571750227382),
 (116.46981229421405, 39.90717615725094),
 (116.45859573583468, 39.90728289609876),
 (116.45200969873245, 39.906822723301026),
 (116.441703102958, 39.90713935089575),
 (116.43677048654843, 39.90708818243652),
 (116.42334274544173, 39.90700982647987),
 (116.40961252669423, 39.90680074457834),
 (116.39514518220952, 39.90601615220736),
 (116.38610976246815, 39.90606611105301),
 (116.37171736434773, 39.905768842235425),
 (116.34421093994041, 39.90589615371753),
 (116.33967074021018, 39.90593668234911),
 (116.32633702300372, 39.90617947724279),
 (116.3181798079248, 39.90621678562706),
 (116.30437746724007, 39.906127604467486),
 (116.2873252094815, 39.90624058712257),
 (116.27604392498202, 39.90622430603992),
 (116.2716146501648, 39.90623722844211),
 (116.26001177549975, 39.906237122676316),
 (116.24545331669, 39.90620798416575),
 (116.22951343694095, 39.90619045973707),
 (116.22766114192765, 39.911219155500675),
 (116.2264529702029, 39.914450096247954)]