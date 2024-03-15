import copy
import os

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import cv2
import networkx as nx


def centroid_coords(poly):
    centroid = poly.centroid
    x, y = centroid.coords[0]
    return x, y


def move_coords(coords, diff_xy):
    new_coords = []
    for j in coords:
        x, y = j
        new_coords.append([x - diff_xy[0], y - diff_xy[1]])  # 进行平移 减去，不是加上
    return new_coords


def input_site(boundary_coords, condition_list, path_files_input, name_exam, canvas_size=1024, input_size=256):
    """

    :param canvas_size: 画布的尺寸
    :param input_size: 最终输出的尺寸
    :param boundary_coords: 场地边界的实际角点
    :param condition_list: [城市编号1-60， 建筑的总面积，平均的层数] m
    :param path_files_input:最终输出的文件储存的位置
    :param name_exam: 最终输出文件的名字(xx.png, xx.npy)
    :return: 储存在指定文件夹的文件png，npy
    """
    canvas_poly = Polygon([(0, 0), (canvas_size, 0), (canvas_size, canvas_size), (0, canvas_size)])  # 画布
    canvas_cen_x, canvas_cen_y = centroid_coords(canvas_poly)  # 计算画布的形心

    # construct boundary polygon # 计算整体平移的距离
    boundary_poly = Polygon(boundary_coords)
    area_site = boundary_poly.area  # 场地的面积
    boundary_cen_x, boundary_cen_y = centroid_coords(boundary_poly)  # 计算场地的形心
    diff_xy = [boundary_cen_x - canvas_cen_x, boundary_cen_y - canvas_cen_y]  # 平移的距离 ######## 只是平移没有缩放

    # move boundary to the center and plot it # 处理边界
    boundary_coords = move_coords(boundary_coords, diff_xy)
    pts_site_bound = np.asarray(boundary_coords, np.int64)

    # 像素化
    scale = int(canvas_size / input_size)
    list_points = pts_site_bound // scale  # 这里需要是整数

    img = np.zeros((input_size, input_size, 1), np.uint8)  # 数据类型必须是int32
    pts = np.array(list_points).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], True, 127, 1)  # 图像，点集，是否闭合，颜色，线条粗细 #边界值为127
    image_in_color = img.squeeze()

    # 仅仅是可视化
    plt.matshow(img, cmap=plt.cm.Blues)
    plt.title('image_in_color')
    plt.show()

    path_png = os.path.join(path_files_input, name_exam + '.png')
    cv2.imwrite(path_png, image_in_color)

    # label
    name_label = name_exam + '.npy'
    np.save(os.path.join(path_files_input, name_label), np.array(condition_list, dtype=object))
    return True


if __name__ == '__main__':
    print('debut')
    input_site(
        [[0, 0], [600, 0], [600, 800], [0, 800]],
        [45, 2100, 10],
        r'\Input_site_files_exam',
        'example1',
        1024, 256, )
    print('end')
