import copy
import os
import sys

import numpy as np
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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


def extract_flat(path_png, path_save, step_value=12, scale_img_ture=18 / 256):  # 单位 m
    # print('debut')

    name_pic = os.path.basename(path_png).split('.')[0]

    array_img = copy.deepcopy(np.array(Image.open(path_png)))

    # plt.matshow(array_img, cmap=plt.cm.tab10)  # cmap 此时好像没有作用
    # plt.title('array_img')
    # plt.show()

    img_layout = array_img[:, :256, :]

    # 三个通道的均值
    array_img_gray = np.mean(img_layout, axis=2)

    plt.matshow(array_img_gray, cmap=plt.cm.BrBG)  # cmap 此时好像没有作用
    plt.title('array_img_gray')
    plt.show()

    # 不同的chanel
    list_spaces = []  # 只需要前四个空间
    list_index_need = list(range(0, 18))
    # list_type_space = ['Apartment', 'Elevator', 'Stair', 'Public', 'External', 'ExWall', 'InWall']
    for index_space in list_index_need:  # 不同的房间类型
        # 不同的房间类型
        img_canvas = np.zeros((256, 256))
        img_canvas[((index_space - 0.5) * step_value <= array_img_gray) &
                   (array_img_gray <= (index_space + 0.5) * step_value)] = 255

        img_canvas = np.array(img_canvas, dtype=np.uint8)

        if index_space <= 11:  # 进行连通域的处理
            kernel = np.ones((7, 7), np.uint8)  # 在这里进行开运算 先腐蚀再膨胀
            img_canvas = cv2.morphologyEx(img_canvas, cv2.MORPH_OPEN, kernel)

        # else:
        # img_canvas = cv2.morphologyEx(img_canvas, cv2.MORPH_CLOSE, kernel)
        # img_canvas = cv2.medianBlur(img_canvas, 3)  # 中值滤波

        # 可视化
        # plt.matshow(img_canvas, cmap=plt.cm.tab10)  # cmap 此时好像没有作用
        # plt.title(str(index_space))
        # plt.show()

        ret, ori_img = cv2.threshold(img_canvas, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ori_img, connectivity=8)

        spaces = []
        for i in range(1, num_labels):  # 遍历每一个连通域
            # print(i)
            # 不同的联通域
            img = np.zeros_like(labels)
            index = np.where(labels == i)
            img[index] = 255
            img = np.array(img, dtype=np.uint8)  # 此时是二值化的

            if index_space == 15 or index_space == 17: # 根据分类进行分类
                points_rec = get_rect(np.stack((img, img, img), axis=2)) * scale_img_ture
                spaces.append(points_rec)

            elif index_space == 13:
                print('外轮廓重新找！')
                # 二值化
                # array_layout = img_layout[:, :, 0]
                ret, thresh = cv2.threshold(img, 64, 255, cv2.THRESH_BINARY)

                plt.matshow(thresh, cmap=plt.cm.tab10)
                plt.title('thresh')
                plt.show()

                # 寻找轮廓
                thresh = thresh.astype(np.uint8)  # findContour需要转为int8
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                boundary_contours = (contours[1].squeeze() * scale_img_ture).tolist()

                # regularization_contour = boundary_regularization(img, epsilon=0).astype(np.int32) c  # m
                spaces.append(boundary_contours)

            else:  # 14 也会被检测 但是没用
                regularization_contour = boundary_regularization(img, epsilon=1).astype(np.int32) * scale_img_ture  # m
                spaces.append(regularization_contour.tolist())

            # buildings_poly = Polygon(np.array(regularization_contour).reshape(-1, 2))
            # plt.fill(*buildings_poly.exterior.xy, color='darkorange')
            # ax = plt.gca()
            # ax.set_aspect(1)
            # plt.title(str(index_space))
            # plt.show()


        list_spaces.append(spaces)

    np.save(os.path.join(path_save, name_pic + '.npy'), np.array(list_spaces, dtype=object))

    # 仅仅可视化而已 ########################3
    colors = list(mcolors.XKCD_COLORS.keys())  # 颜色变化

    for index, spaces_get in enumerate(list_spaces):
        if index <= 11:
            for sp in spaces_get:
                buildings_poly = Polygon(np.array(sp).reshape(-1, 2))
                plt.fill(*buildings_poly.exterior.xy, color=mcolors.XKCD_COLORS[colors[index]])

        elif index > 11:  # 可视化可以好好利用这些线条
            for sp in spaces_get:
                # plt.Polygon(xy=sp, color=list_colors[index], alpha=0.8)
                buildings_poly = Polygon(np.array(sp).reshape(-1, 2))
                plt.plot(*buildings_poly.exterior.xy, color=mcolors.XKCD_COLORS[colors[index]], linewidth=1)

    plt.show()

    return list_spaces


def get_center_lines_rec(list_points):
    """
    得到矩形的长中心线
    :param list_points:
    :return:
    """
    import matplotlib.pyplot as plt
    x = [point[0] for point in list_points]
    y = [point[1] for point in list_points]

    x = x + [x[0]]  # x 的闭合列表
    y = y + [y[0]]  # y 的闭合列表
    # plt.plot(x, y)

    list_center_lines = []
    for i in range(2):
        list_center_lines.append([[(x[i] + x[i + 1]) / 2, (y[i] + y[i + 1]) / 2],
                                  [(x[i + 2] + x[i + 3]) / 2, (y[i + 2] + y[i + 3]) / 2]])

    #     plt.plot([(x[i] + x[i + 1]) / 2, (x[i + 2] + x[i + 3]) / 2],
    #              [(y[i] + y[i + 1]) / 2, (y[i + 2] + y[i + 3]) / 2])
    # plt.show()

    if LineString(list_center_lines[0]).length > LineString(list_center_lines[1]).length:
        return list_center_lines[0]
    else:
        return list_center_lines[1]


def get_rect(img):
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 二值化
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # 寻找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 取第一个轮廓
    cnt = contours[0]
    # 拟合最小外接矩形
    rect = cv2.minAreaRect(cnt)
    # 获取四个角点坐标
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    # # 画出矩形
    # cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

    # # 拟合中心线
    # [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    #
    # # 计算直线上两个端点的坐标
    # rows, cols = img.shape[:2]
    # lefty = int((-x * vy / vx) + y)
    # righty = int(((cols - x) * vy / vx) + y)
    #
    # # # 画出中心线
    # # cv2.line(img, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)

    # # 显示图片
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return box


def write_gh_data(path_name_file, list_data):
    example = list_data[0]
    print(type(example))

    if type(example) == type('a'):
        with open(os.path.join(path_name_file), 'w') as f_points:
            for point in list_data:
                f_points.write('%s' % point)
                f_points.write('\n')
        f_points.close()

    if type(example) == type([1,2,3]):
        with open(os.path.join(path_name_file), 'w') as f_points:
            for point in list_data:
                f_points.write('%.2f,%.2f,%.2f' % (point[0], point[1], point[2]))
                f_points.write('\n')
        f_points.close()

    if type(example) == type(1) :
        with open(os.path.join(path_name_file), 'w') as f_points:
            for point in list_data:
                f_points.write('%.2f' % point)
                f_points.write('\n')
        f_points.close()

    return True


def gh_flat_layout(list_spaces_all, path_save, name_flat):
    dict_label_type = {0: 'Living',
                       1: 'Master',
                       2: 'Kitchen',
                       3: 'Bath',
                       4: 'Dining',
                       5: 'Child',
                       6: 'Study',
                       7: 'Second',
                       8: 'Guest',
                       9: 'Balcony',
                       10: 'Entrance',
                       11: 'Storage',
                       12: 'Wall-in',
                       13: 'External area',
                       14: 'Exterior wall',
                       15: 'Front door',
                       16: 'Interior wall',
                       17: 'Interior door'}

    list_points_polygon = []
    list_count_polygons = []
    list_type_polygon = []
    list_label_polygon = []

    list_points_living = []
    list_count_living = []

    list_line_outline = []

    list_line_doors = []

    for index, spaces_get in enumerate(list_spaces_all):
        type_name = dict_label_type[index]

        if index == 15:
            for space in spaces_get:
                line_center = get_center_lines_rec(space)
                for point in line_center:
                    list_line_doors.append([point[0], point[1], 0])

        if index == 17:
            for space in spaces_get:
                line_center = get_center_lines_rec(space)
                for point in line_center:
                    list_line_doors.append([point[0], point[1], 0])

        if index == 13:  # 外墙
            for space in spaces_get:
                for point in space:
                    list_line_outline.append([point[0], point[1], 0])

        if 1 <= index <= 11:
            for space in spaces_get:  # 遍历每一个空间
                list_type_polygon.append(type_name)
                list_label_polygon.append(int(index))
                list_count_polygons.append(len(space))
                for point in space:
                    list_points_polygon.append([point[0], point[1], 0])

        if index == 0:
            print('-----------')
            print(index)
            print(spaces_get)
            for space in spaces_get:  # 遍历每一个空间
                list_count_living.append(len(space))
                for point in space:
                    list_points_living.append([point[0], point[1], 0])
        else:
            pass

    # 储存数据
    # normal rooms
    name_type = 'normal_rooms'
    data_type = 'points'
    data_this = list_points_polygon
    name_file = 'flat_' + str(name_flat) + '_' + str(name_type) + '_' + str(data_type) + '.txt'
    path_name_file_this = os.path.join(path_save, name_file)
    write_gh_data(path_name_file_this, data_this)

    data_type = 'count'
    data_this = list_count_polygons
    name_file = 'flat_' + str(name_flat) + '_' + str(name_type) + '_' + str(data_type) + '.txt'
    path_name_file_this = os.path.join(path_save, name_file)
    write_gh_data(path_name_file_this, data_this)

    data_type = 'type'
    data_this = list_type_polygon
    name_file = 'flat_' + str(name_flat) + '_' + str(name_type) + '_' + str(data_type) + '.txt'
    path_name_file_this = os.path.join(path_save, name_file)
    print('typeeeeeeeeeeeeee')
    write_gh_data(path_name_file_this, data_this)

    data_type = 'label'
    data_this = list_label_polygon
    name_file = 'flat_' + str(name_flat) + '_' + str(name_type) + '_' + str(data_type) + '.txt'
    path_name_file_this = os.path.join(path_save, name_file)
    write_gh_data(path_name_file_this, data_this)

    # living
    name_type = 'living'
    data_type = 'points'
    data_this = list_points_living
    name_file = 'flat_' + str(name_flat) + '_' + str(name_type) + '_' + str(data_type) + '.txt'
    path_name_file_this = os.path.join(path_save, name_file)
    write_gh_data(path_name_file_this, data_this)

    data_type = 'count'
    data_this = list_count_living
    name_file = 'flat_' + str(name_flat) + '_' + str(name_type) + '_' + str(data_type) + '.txt'
    path_name_file_this = os.path.join(path_save, name_file)
    write_gh_data(path_name_file_this, data_this)

    # outline
    name_type = 'outline'
    data_type = 'points'
    data_this = list_line_outline
    name_file = 'flat_' + str(name_flat) + '_' + str(name_type) + '_' + str(data_type) + '.txt'
    path_name_file_this = os.path.join(path_save, name_file)
    write_gh_data(path_name_file_this, data_this)

    # outline
    name_type = 'doors'
    data_type = 'lines'
    data_this = list_line_doors
    name_file = 'flat_' + str(name_flat) + '_' + str(name_type) + '_' + str(data_type) + '.txt'
    path_name_file_this = os.path.join(path_save, name_file)
    write_gh_data(path_name_file_this, data_this)

    print('debut')


if __name__ == '__main__':

    path_png_1 = r'\2164.png'
    path_save_1 = r'\example_flats_txt'
    list_spaces = extract_flat(path_png_1, path_save_1)

    # flats = list_spaces[0]
    # public = list_spaces[3][0]
    #
    # list_apartment_doors = find_all_doors(flats, public)

    path_save_gh_1 = r'E:\GH-EX'
    name_flat_1 = '2164'
    gh_flat_layout(list_spaces, path_save_gh_1, name_flat_1)
