import os
import sys
import lnglat_mercator_tiles_convertor as convertor
import json

def readpoints():
    pointsFile = './points.json'
    data = []
    tilePoint = {}
    with open(pointsFile, 'r') as f:
        data = json.load(f).get('wgs84points')

    # print('readpoints'+ str(len(data)))
    for point in data:
        # point[0] = round(point[0],6)
        # point[1] = round(point[1],6)
        lng_BD09,lat_BD09 = convertor.wgs84_to_bd09(point[0], point[1])
        pointX,pointY = convertor.BD092mercotor(lng_BD09,lat_BD09)
        tileX,tileY,pixelX,pixelY = convertor.point2tiles_pixel(pointX,pointY,14)

        tileName = str(tileX)+str(tileY)
        if(tileName in tilePoint.keys()):
            tilePoint[tileName].append([pixelX,pixelY,point[0],point[1]])
        else:
            tilePoint[tileName] = []
            tilePoint[tileName].append([pixelX,pixelY,point[0],point[1]])
    return tilePoint

def rgb2hsv(rgb):
    r, g, b = rgb[0], rgb[1], rgb[2]
    m_x = max(r, g, b)
    m_n = min(r, g, b)
    m = m_x - m_n
    if m_x == m_n:
        h = 0
    elif m_x == r:
        if g >= b:
            h = ((g - b) / m) * 60
        else:
            h = ((g - b) / m) * 60 + 360
    elif m_x == g:
        h = ((b - r) / m) * 60 + 120
    elif m_x == b:
        h = ((r - g) / m) * 60 + 240
    if m_x == 0:
        s = 0
    else:
        s = m / m_x
    v = m_x
    H = h / 2
    S = s * 255.0
    V = v * 255.0
    return int(round(H)), int(round(S)), int(round(V))


def hsv2value(hsv):
    h, s, v = hsv[0], hsv[1], hsv[2]
    if 35 <= h <= 99 and 43 <= s <= 255 and 46 <= v <= 255:  # green
        return 3
    elif 0 <= h <= 10 and 43 <= s <= 255 and 46 <= v <= 255:  # red
        return 10
    elif 11 <= h <= 34 and 43 <= s <= 255 and 46 <= v <= 255:  # yellow
        return 7
    elif 0 <= s <= 43 and 46 <= v <= 255:  # white and gray
        return 1
    else:  # black
        return 0

# 将RGB颜色映射到值 灰度化 交通中红色绿色蓝色代表的值不一样权重应该不同
def RGB2Value(R,G,B):
    # 灰度值的加权平均法Gray = 0.299*R + 0.578*G + 0.114*B
    # weight = [0.600,0.100,0.300]
    # value = weight[0]*R + weight[1]*G + weight[2]*B
    # value = round(R,6)
    H,S,V = rgb2hsv([R,G,B])
    value = hsv2value([H,S,V])
    
    return value