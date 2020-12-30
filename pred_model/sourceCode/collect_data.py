# -*- coding: utf-8 -*-
"""
@Time   : 2020/9/13

@Author : Shen Fang
"""
import os
import datetime
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import argparse


def color2value(rgb):
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
            return 20
        elif 0 <= h <= 10 and 43 <= s <= 255 and 46 <= v <= 255:  # red
            return 60
        elif 11 <= h <= 34 and 43 <= s <= 255 and 46 <= v <= 255:  # yellow
            return 40
        elif 0 <= s <= 43 and 46 <= v <= 255:  # white and gray
            return 20  # 10
        else:  # black
            return 20  # 0

    return hsv2value(rgb2hsv(rgb))


def read_coord(file_name):
    coord_data = np.load(file_name)
    coord_data = coord_data[:, [1, 0]]  # 行列交换
    return coord_data


def read_image(image_file):
    image = plt.imread(image_file)
    return image


def rename_file(folder, old_name, new_name):
    os.rename(osp.join(folder, old_name), osp.join(folder, new_name))


def delete_file(folder, file_name):
    os.remove(osp.join(folder, file_name))


def find_lost_time(time_list, year, month, day):
    refer_list = []
    start_time = datetime.datetime(int(year), int(month), int(day))
    delta_time = datetime.timedelta(minutes=5)

    while start_time < datetime.datetime(int(year), int(month), int(day)) + datetime.timedelta(days=1):
        refer_list.append(start_time.strftime("%Y-%m-%d_%H-%M.png"))
        start_time += delta_time

    for refer in refer_list:
        if refer not in time_list:
            print(refer)


class ObtainInfo:
    def __init__(self, coord_file):
        self.coord = read_coord(coord_file).astype(int)

    def delete_image_file(self, image_folder):
        files = sorted(os.listdir(image_folder))
        if ".DS_Store" in files:
            files.remove(".DS_Store")

        for file in files:
            if int(file.split(".")[0].split("-")[3]) % 5 != 0:
                delete_file(image_folder, file)

    def rename_image_file(self, image_folder):
        files = sorted(os.listdir(image_folder))
        if ".DS_Store" in files:
            files.remove(".DS_Store")

        for file in files:
            if len(file) > 20:
                new_file = "-".join(file.split("-")[:4]) + ".png"
                rename_file(image_folder, file, new_file)

    def read_image_data(self, image_file):
        image_data = read_image(image_file)
        color = image_data[self.coord[:,0], self.coord[:, 1]][:, :3]
        color_iter = map(color2value, color)
        color_result = [r for r in color_iter]
        return color_result

    def read_one_day(self, data_folder):
        file_names = sorted(os.listdir(data_folder))

        if ".DS_Store" in file_names:
            file_names.remove(".DS_Store")

        file_names = map(osp.join, [data_folder ] * len(file_names), file_names)

        file_names = [file for file in file_names]

        data_iter = map(self.read_image_data, file_names)

        result_data = []
        i = 1
        for data in data_iter:
            print(i)
            i += 1
            result_data.append(data)

        result_data = np.array(result_data).transpose()  # [N, T]

        return result_data

    def process_one_day(self, data_folder, save_file_name):
        data = self.read_one_day(data_folder)
        print("Time: ", data_folder, "Data Shape: ", data.shape)
        np.save(save_file_name, data)


if __name__ == '__main__':
    # map_image = read_image("level_14.png")
    # patch = map_image[900:2000, 900:]
    # plt.imshow(patch)
    # plt.show()

    parser = argparse.ArgumentParser(description="Process data on each day")
    parser.add_argument("year", type=str, default="2019")
    parser.add_argument("month", type=str)
    parser.add_argument("day", type=str)

    args = parser.parse_args()

    year = args.year
    month = args.month
    day = args.day

    data_folder = "-".join([year, month, day])
    result_file = "".join([year, month, day]) + ".npy"

    obtain = ObtainInfo("image_coord.npy")

    obtain.delete_image_file(data_folder)
    obtain.rename_image_file(data_folder)

    folder = os.listdir(data_folder)
    if ".DS_Store" in folder:
        folder.remove(".DS_Store")

    num_files = len(folder)

    if num_files == 288:
        obtain.process_one_day(data_folder, result_file)
    else:
        print("[Warning]\nNum of Files is not correct: {:d}, left: {:d}".format(num_files, 288-num_files))
        print("Lost Time:")
        find_lost_time(folder, year, month, day)

