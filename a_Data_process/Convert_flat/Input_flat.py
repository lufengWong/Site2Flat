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


def input_flat(boundary_coords, door_coords, condition_list, path_files_input, name_exam, canvas_size=1024, input_size=256):
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

    # 移动门
    boundary_door = move_coords(door_coords, diff_xy)
    pts_door_bound = np.asarray(boundary_door, np.int64)

    # 边界像素化
    scale = int(canvas_size / input_size)
    list_points = pts_site_bound // scale  # 这里需要是整数

    img = np.zeros((input_size, input_size, 1), np.uint8)  # 数据类型必须是int32
    pts = np.array(list_points).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], True, 127, 1)  # 图像，点集，是否闭合，颜色，线条粗细 #边界值为127
    image_in_color = img.squeeze()

    # 门像素化
    list_points_door = pts_door_bound // scale  # 这里需要是整数
    pts_door = np.array(list_points_door).reshape((-1, 1, 2))
    cv2.polylines(image_in_color, [pts_door], True, 255, 1)  # 图像，点集，是否闭合，颜色，线条粗细 #边界值为127
    image_in_color = image_in_color.squeeze()

    # # 获取边界 和 获取内部区域
    # img_region = np.zeros((canvas_size, canvas_size))
    # cv2.fillPoly(img_region,  # 原图画板
    #              [pts_site_bound],  # 多边形的点
    #              color=1)  # image_0 的内部区域为1
    # img_region = cv2.blur(img_region, (3, 3))  # 进行平滑操作
    #
    # # # 仅仅是可视化
    # # plt.matshow(img_region, cmap=plt.cm.Blues)
    # # plt.title('Site region')
    # # plt.show()

    # # 获取边界
    # img_bound = np.zeros((canvas_size, canvas_size))
    # img_3_channel = np.stack((img_region, img_region, img_region), axis=2) * 255
    # img_3_channel = np.asarray(img_3_channel, dtype=np.uint8)
    # imgray = cv2.cvtColor(img_3_channel, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)  # 获取所有的边界点
    # boundary_site = contours[0]

    # for point in boundary_site:
    #     img_bound[point[0][1], point[0][0]] = 1
    #
    # # img_bound = cv2.blur(img_bound, (5, 5))  # 进行平滑操作
    # kernel = np.ones((3, 3), np.uint8)
    # img_bound = cv2.dilate(img_bound, kernel, iterations=1)  # 进行形态学的操作
    #
    # data_img = np.stack((img_bound, img_bound, img_bound), axis=2)
    # array_in = copy.deepcopy(data_img).astype(np.uint8)
    # #
    # image_in_color = cv2.resize(img,
    #                             dsize=(input_size, input_size),
    #                             fx=1, fy=1,
    #                             interpolation=cv2.INTER_LINEAR)

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
    input_flat(
        [[0, 0], [600, 0], [600, 800], [0, 800]],
        [[0, 100], [0, 300]],
        [45, 2100, 10],
        r'\Input_site_files_exam',
        'example1',
        1024, 256, )
    print('end')
