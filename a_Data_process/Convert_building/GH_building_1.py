import copy
import os
import sys

import numpy as np
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union
import matplotlib.pyplot as plt

import networkx as nx
from PIL import Image

import cv2

from RS_building_regularization_main.main_regularization import boundary_regularization


def find_door(polygon_flat, polygon_public, length_buffer=800, length_door=1200):
    # 功能区域膨胀, 取中间点，然后可以取的话取1200
    polygon_public_buffer = Polygon(np.array(polygon_public)).buffer(length_buffer)
    list_points_shape = polygon_public_buffer.intersection(Polygon(polygon_flat))
    list_point_polygons = list(list_points_shape.exterior.coords)

    # # points_list =list(list_points_shape.coords)
    # for pol in list_points_shape.geoms:
    #     print(list(pol.coords))

    for index_start in range(len(list_point_polygons)):
        index_end = index_start + 1
        if index_start == len(list_point_polygons) - 1:
            index_end = 0

        point_start = list_point_polygons[index_start]
        point_end = list_point_polygons[index_end]

        if LineString([point_start, point_end]).within(Polygon(polygon_flat).exterior):  # 1. 在多边形上
            length_inter = LineString([point_start, point_end]).length
            if length_inter >= length_door:  # 2. 长度大于门的长度
                line = LineString([point_start, point_end])  # 创建一个长度为10的水平线段
                length_line = line.length

                point1 = line.interpolate(length_line / 2 - length_door / 2)  # 取线段上距离起点4长度的点 # 距离起点的距离
                point2 = line.interpolate(length_line / 2 + length_door / 2)  # 取线段上距离起点6长度的点
                segment = LineString([point1, point2])  # 创建一个由两个点组成的线段
                points_door = list(segment.coords)  # 输出结果为LINESTRING (4 0, 6 0) #

    return points_door


def extract_building(path_png, path_save, step_value=32, scale_img_ture=192):
    print('debut')

    name_pic = os.path.basename(path_png).split('.')[0]

    array_img = copy.deepcopy(np.array(Image.open(path_png)))

    # plt.matshow(array_img, cmap=plt.cm.tab10)  # cmap 此时好像没有作用
    # plt.title('array_img')
    # plt.show()

    img_layout = array_img[:, :256, :]

    # plt.matshow(img_layout, cmap=plt.cm.tab10)  # cmap 此时好像没有作用
    # plt.title('img_layout')
    # plt.show()

    # 三个通道的均值
    array_img_gray = np.mean(img_layout, axis=2)

    plt.matshow(array_img_gray, cmap=plt.cm.tab20c)  # cmap 此时好像没有作用
    plt.title('img_layout')
    plt.show()

    # 不同的chanel
    list_spaces = []  # 只需要前四个空间
    list_index_need = [0, 1, 2, 3, 4, 5, 6]
    list_type_space = ['Apartment', 'Elevator', 'Stair', 'Public', 'External', 'ExWall', 'InWall']
    for index_space, type_space in zip(list_index_need, list_type_space):
        # 不同的房间类型
        print(type_space)
        img_canvas = np.zeros((256, 256))
        img_canvas[((index_space - 0.5) * step_value <= array_img_gray) &
                   (array_img_gray <= (index_space + 0.5) * step_value)] = 255  # 一定范围取值

        img_canvas = np.array(img_canvas, dtype=np.uint8)

        if index_space <= 3:
            kernel = np.ones((7, 7), np.uint8)  # 在这里进行开运算 先腐蚀再膨胀
            img_canvas = cv2.morphologyEx(img_canvas, cv2.MORPH_OPEN, kernel)
        # else:
        # img_canvas = cv2.morphologyEx(img_canvas, cv2.MORPH_CLOSE, kernel)
        # img_canvas = cv2.medianBlur(img_canvas, 3)  # 中值滤波

        # # 仅仅是绘图
        # plt.matshow(img_canvas, cmap=plt.cm.tab20b)  # cmap 此时好像没有作用
        # plt.title(str(type_space))
        # plt.show()

        ret, ori_img = cv2.threshold(img_canvas, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ori_img, connectivity=8)

        spaces = []
        for i in range(1, num_labels):
            print(i)
            # 不同的联通域
            img = np.zeros_like(labels)
            index = np.where(labels == i)
            img[index] = 255
            img = np.array(img, dtype=np.uint8)

            # plt.matshow(img, cmap=plt.cm.tab10)  # cmap 此时好像没有作用
            # plt.title(str(type_space) + '_' + str(i))
            # plt.show()

            regularization_contour = boundary_regularization(img, epsilon=0).astype(np.int32) * scale_img_ture  # mm
            print(regularization_contour)
            spaces.append(regularization_contour)

            # buildings_poly = Polygon(np.array(regularization_contour).reshape(-1, 2))
            # plt.fill(*buildings_poly.exterior.xy, color='darkorange')
            # ax = plt.gca()
            # ax.set_aspect(1)
            # plt.show()

        list_spaces.append(spaces)

    list_colors = ['green', 'orange', 'blue', 'red', 'None', 'cyan', 'black']
    for index, spaces_get in enumerate(list_spaces):
        if index <= 3:
            for sp in spaces_get:
                buildings_poly = Polygon(np.array(sp).reshape(-1, 2))
                plt.fill(*buildings_poly.exterior.xy, color=list_colors[index])

        elif index == 5:  # 可视化可以好好利用这些线条
            for sp in spaces_get:
                # plt.Polygon(xy=sp, color=list_colors[index], alpha=0.8)
                buildings_poly = Polygon(np.array(sp).reshape(-1, 2))
                plt.plot(*buildings_poly.exterior.xy, color=list_colors[index], linewidth=1)

        elif index == 6:
            for sp in spaces_get:
                # plt.Polygon(xy=sp, color=list_colors[index], alpha=0.8)
                buildings_poly = Polygon(np.array(sp).reshape(-1, 2))
                plt.fill(*buildings_poly.exterior.xy, color=list_colors[index])
        # pass 放在最后

    plt.show()

    np.save(os.path.join(path_save, name_pic + '.npy'), np.array(list_spaces, dtype=object))
    return list_spaces


def find_all_doors(polygon_flats, polygon_public, length_buffer=800, length_door=1200):
    # 功能区域膨胀, 取中间点，然后可以取的话取1200
    list_apartment_doors = []
    for index, polygon_flat in enumerate(polygon_flats):
        door_this = find_door(polygon_flat, polygon_public, length_buffer=length_buffer, length_door=length_door)
        door_this = np.around(door_this, decimals=0).astype(np.uint64).tolist()
        list_apartment_doors.append({'polygon': polygon_flat, 'door': door_this})

    return list_apartment_doors


def gh_building_layout(list_spaces_all, list_doors, path_save, name_building):
    flats = list_spaces_all[0]
    eles = list_spaces_all[1]
    stairs = list_spaces_all[2]
    exWalls = list_spaces_all[5]
    doors = list_doors

    scale_gh = 1000

    # flats
    data_from = flats
    name_type = 'flats'

    data_type = 'points'
    name_file = 'building_' + str(name_building) + '_' + str(name_type) + '_' + str(data_type) + '.txt'
    with open(os.path.join(path_save, name_file), 'w') as f_points:
        for flat in data_from:
            count_flat_points = len(flat)
            for point in flat:
                f_points.write('%.2f,%.2f,%.2f' % (point[0] / scale_gh, point[1] / scale_gh, 0))
                f_points.write('\n')
    f_points.close()

    data_type = 'count'
    name_file = 'building_' + str(name_building) + '_' + str(name_type) + '_' + str(data_type) + '.txt'
    with open(os.path.join(path_save, name_file), 'w') as f_points:
        for flat in data_from:
            count_flat_points = len(flat)
            f_points.write('%.2f' % count_flat_points)
            f_points.write('\n')
    f_points.close()

    # eles
    data_from = eles
    name_type = 'eles'

    data_type = 'points'
    name_file = 'building_' + str(name_building) + '_' + str(name_type) + '_' + str(data_type) + '.txt'
    with open(os.path.join(path_save, name_file), 'w') as f_points:
        for flat in data_from:
            count_flat_points = len(flat)
            for point in flat:
                f_points.write('%.2f,%.2f,%.2f' % (point[0] / scale_gh, point[1] / scale_gh, 0))
                f_points.write('\n')
    f_points.close()

    data_type = 'count'
    name_file = 'building_' + str(name_building) + '_' + str(name_type) + '_' + str(data_type) + '.txt'
    with open(os.path.join(path_save, name_file), 'w') as f_points:
        for flat in data_from:
            count_flat_points = len(flat)
            f_points.write('%.2f' % count_flat_points)
            f_points.write('\n')
    f_points.close()

    # stairs
    data_from = stairs
    name_type = 'stairs'

    data_type = 'points'
    name_file = 'building_' + str(name_building) + '_' + str(name_type) + '_' + str(data_type) + '.txt'
    with open(os.path.join(path_save, name_file), 'w') as f_points:
        for flat in data_from:
            count_flat_points = len(flat)
            for point in flat:
                f_points.write('%.2f,%.2f,%.2f' % (point[0] / scale_gh, point[1] / scale_gh, 0))
                f_points.write('\n')
    f_points.close()

    data_type = 'count'
    name_file = 'building_' + str(name_building) + '_' + str(name_type) + '_' + str(data_type) + '.txt'
    with open(os.path.join(path_save, name_file), 'w') as f_points:
        for flat in data_from:
            count_flat_points = len(flat)
            f_points.write('%.2f' % count_flat_points)
            f_points.write('\n')
    f_points.close()

    # exWalls
    data_from = exWalls
    name_type = 'exWalls'

    data_type = 'all'
    name_file = 'building_' + str(name_building) + '_' + str(name_type) + '_' + str(data_type) + '.txt'
    with open(os.path.join(path_save, name_file), 'w') as f_points:
        for flat in data_from:
            count_flat_points = len(flat)
            for point in flat:
                f_points.write('%.2f,%.2f,%.2f' % (point[0] / scale_gh, point[1] / scale_gh, 0))
                f_points.write('\n')
    f_points.close()

    # data_type = 'count'
    # name_file = 'building_' + str(name_building) + '_' + str(name_type) + '_' + str(data_type) + '.txt'
    # with open(os.path.join(path_save, name_file), 'w') as f_points:
    #     for flat in data_from:
    #         count_flat_points = len(flat)
    #         f_points.write('%.2f' % count_flat_points)
    #         f_points.write('\n')
    # f_points.close()

    # doors
    data_from = doors
    name_type = 'doors'

    data_type = 'all'
    name_file = 'building_' + str(name_building) + '_' + str(name_type) + '_' + str(data_type) + '.txt'
    with open(os.path.join(path_save, name_file), 'w') as f_points:
        for door_polygon in data_from:
            door_points = door_polygon['door']
            assert len(door_points) == 2, '门的坐标错误'
            for point in door_points:
                f_points.write('%.2f,%.2f,%.2f' % (float(point[0] / scale_gh), point[1] / scale_gh, 0))
                f_points.write('\n')
    f_points.close()


if __name__ == '__main__':
    print('11111')
    # 1
    path_png_1 = r'\1314_1_1.png'
    path_save_1 = r'\example_building_txt'
    list_spaces = extract_building(path_png_1, path_save_1)

    # 2
    flats = list_spaces[0]
    public = list_spaces[3][0]
    list_apartment_doors = find_all_doors(flats, public)

    # 3
    path_save_1 = r'E:\GH-EX'
    gh_building_layout(list_spaces, list_apartment_doors, path_save_1, '1314_1_1')
