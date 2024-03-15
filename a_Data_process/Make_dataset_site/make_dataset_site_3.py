# -*- coding: utf-8 -*-
import os

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import json
import cv2
import networkx as nx


# find centroid of a polygon
def centroid_coords(poly):
    centroid = poly.centroid
    x, y = centroid.coords[0]
    return x, y


# parallel moving all coords to the center of the graph
def move_coords(coords, diff_xy):
    new_coords = []
    for j in coords:
        x, y = j
        new_coords.append([x - diff_xy[0], y - diff_xy[1]])  # 进行平移 减去，不是加上
    return new_coords


# function of plotting
def plot_community(community, canvas_size, path_png_data_save):
    # 放到1024里边的时候尺寸是没有变化的坐标点和原来的坐标点是一致的 （只存在一个论文上的缩放比例）
    # 第一个通道 边界
    # 第二个通道 场地区域
    # 第三个通道 区域以及层高

    canvas_poly = Polygon([(0, 0), (canvas_size, 0), (canvas_size, canvas_size), (0, canvas_size)])  # 画布
    canvas_cen_x, canvas_cen_y = centroid_coords(canvas_poly)  # 计算画布的形心

    # if coords transfer is needed, read data from raw data, else read data from _rp

    boundary_coords = community['boundary']  # 场地边界
    buildings = community['buildings']
    _id = community['_id']
    city_ = community['city']

    # construct boundary polygon # 计算整体平移的距离
    boundary_poly = Polygon(boundary_coords)
    area_site = boundary_poly.area  # 场地的面积
    boundary_cen_x, boundary_cen_y = centroid_coords(boundary_poly)  # 计算场地的形心
    diff_xy = [boundary_cen_x - canvas_cen_x, boundary_cen_y - canvas_cen_y]  # 平移的距离 ######## 只是平移没有缩放

    # move boundary to the center and plot it # 处理边界
    boundary_coords = move_coords(boundary_coords, diff_xy)
    pts_site_bound = np.asarray(boundary_coords, np.int64)

    # 获取边界 和 获取内部区域
    img_region = np.zeros((canvas_size, canvas_size))
    cv2.fillPoly(img_region,  # 原图画板
                 [pts_site_bound],  # 多边形的点
                 color=1)  # image_0 的内部区域为1
    img_region = cv2.blur(img_region, (3, 3))  # 进行平滑操作

    # # 仅仅是可视化
    # plt.matshow(img_region, cmap=plt.cm.Blues)
    # plt.title('Site region')
    # plt.show()

    # 获取边界
    img_bound = np.zeros((canvas_size, canvas_size))
    img_3_channel = np.stack((img_region, img_region, img_region), axis=2) * 255
    img_3_channel = np.asarray(img_3_channel, dtype=np.uint8)
    imgray = cv2.cvtColor(img_3_channel, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)  # 获取所有的边界点
    boundary_site = contours[0]

    for point in boundary_site:
        img_bound[point[0][1], point[0][0]] = 1

        # # 8 邻域
        # # 横竖 斜线
        # if point[0][1] + 1 < canvas_size:
        #     img_bound[point[0][1] + 1, point[0][0]] = 1
        # if point[0][0] + 1 < canvas_size:
        #     img_bound[point[0][1], point[0][0] + 1] = 1
        # if point[0][1] - 1 >= 0:
        #     img_bound[point[0][1] - 1, point[0][0]] = 1
        # if point[0][0] - 1 >= 0:
        #     img_bound[point[0][1], point[0][0] - 1] = 1
        # # 斜线
        # if point[0][1] + 1 < canvas_size and point[0][0] - 1 >= 0:
        #     img_bound[point[0][1] + 1, point[0][0] - 1] = 1
        #
        # if point[0][1] + 1 < canvas_size and point[0][0] + 1 < canvas_size:
        #     img_bound[point[0][1] + 1, point[0][0] + 1] = 1
        #
        # if point[0][1] - 1 >= 0 and point[0][0] - 1 >= 0:
        #     img_bound[point[0][1] - 1, point[0][0] - 1] = 1
        #
        # if point[0][1] - 1 >= 0 and point[0][0] + 1 < canvas_size:
        #     img_bound[point[0][1] - 1, point[0][0] + 1] = 1

    # img_bound = cv2.blur(img_bound, (5, 5))  # 进行平滑操作
    kernel = np.ones((3, 3), np.uint8)
    img_bound = cv2.dilate(img_bound, kernel, iterations=1)  # 进行形态学的操作

    # # 仅仅是可视化
    # plt.matshow(img_bound, cmap=plt.cm.Blues)
    # plt.title('Site boundary')
    # plt.show()

    # 先处理掉被错误分割的建筑单体
    # 1 依次将多边形膨胀，如果与其他多边形相交且高层相同则合并
    list_building_coords = [buildings[i]['coords'] for i in range(len(buildings))]

    # 删除不合理的数据
    area_buildings = sum([Polygon(build).area for build in list_building_coords])
    percent_building_sit = area_buildings / area_site
    if percent_building_sit <= 0.1:
        return False

    list_building_height = [buildings[i]['floor'] for i in range(len(buildings))]
    list_combine_1 = []
    for index_female, (building_coord_female, building_height_female) \
            in enumerate(zip(list_building_coords, list_building_height)):
        for index_male, (building_coord_male, building_height_male) \
                in enumerate(zip(list_building_coords, list_building_height)):

            if index_male > index_female:  # 小对比大的
                poly_male = Polygon(building_coord_male).buffer(0.01)
                poly_female = Polygon(building_coord_female).buffer(0.01)
                if poly_male.intersects(poly_female) and building_height_male == building_height_female:
                    list_combine_1.append([index_female, index_male])
            else:
                pass

    # 连通分析
    l = list_combine_1
    G = nx.Graph()
    # 将节点添加到Graph
    G.add_nodes_from(sum(l, []))
    # 从节点列表创建边
    q = [[(s[i], s[i + 1]) for i in range(len(s) - 1)] for s in l]
    for i in q:
        # 向Graph添加边
        G.add_edges_from(i)
    # 查找每个组件的图形和列表节点中的所有连接组件
    k = [list(i) for i in nx.connected_components(G)]
    # print(k)
    list_combine = k

    # 制作新的 buildings_new
    buildings_new = []
    for building_combine in list_combine:
        list_polygons = [Polygon(list_building_coords[index_building]) for index_building in building_combine]
        build_new = unary_union(list_polygons)

        # plt.fill(*build_new.exterior.xy, color='red')
        # plt.show()

        list_new_ = []
        if build_new.geom_type != 'Polygon':  # 去掉其他类型的 # 可改为 geom_type
            continue
        build_new_coords = list(build_new.exterior.xy)
        for x, y in zip(build_new_coords[0].tolist(), build_new_coords[1].tolist()):
            list_new_.append([x, y])
        build_new_coords = list_new_

        # 添加组合后的
        buildings_new.append(dict(coords=build_new_coords, floor=list_building_height[building_combine[0]]))
    # plt.show()

    # 添加没有变化的
    buildings_dealt = list(set(sum(list_combine, [])))
    for index_build in range(len(list_building_coords)):
        if index_build in buildings_dealt:
            pass
        else:
            buildings_new.append(dict(coords=list_building_coords[index_build],
                                      floor=list_building_height[index_build]))

    buildings = buildings_new
    # 完成处理坏数据

    img_region_building = np.zeros((canvas_size, canvas_size))
    for i in range(len(buildings)):  # 遍历所有的建筑
        building_coords = buildings[i]['coords']
        building_height = buildings[i]['floor']

        # building 的面积
        building_area_this = Polygon(building_coords).area
        # print('building area: m**2')  # 此时1个格子就是实际的1m 实际的坐标对应像素的坐标
        # print(building_area_this)
        # move buildings to the center
        building_coords = move_coords(building_coords, diff_xy)  # 相对形心进行平移
        pts_build_bound = np.asarray(building_coords, np.int64)

        img_region_building_this = np.zeros((canvas_size, canvas_size))  # 区域为0
        cv2.fillPoly(img_region_building_this,  # 原图画板
                     [pts_build_bound],  # 多边形的点
                     color=int(building_height))  # image_0 的内部区域为1

        img_region_building += img_region_building_this

    # plt.matshow(img_region_building, cmap=plt.cm.Blues)
    # plt.title('Building region')
    # plt.show()

    # plt.matshow(img_bound + img_region_building, cmap=plt.cm.Blues)
    # plt.title('Site layout')
    # plt.show()

    data_img = np.stack((img_bound, img_region, img_region_building), axis=2)
    name_png = os.path.join(path_png_data_save, city_ + '_' + _id + '.png')
    cv2.imwrite(name_png, data_img)

    # plt.matshow(data_img[:, :, 2], cmap=plt.cm.Blues)
    # plt.title('2')
    # plt.show()
    return True


if __name__ == '__main__':
    # _id_to_draw = '61ef8a8b32b5d4672152cf77'
    # _id_to_draw = '61c30e4cbf4bd2130d036c64'
    # _id_to_draw ='61c30e4cbf4bd2130d036ca8'
    # _id_to_draw ='61c30e4dbf4bd2130d036eb9'
    _id_to_draw = '61c30e4dbf4bd2130d036eb9'

    path_json = r'\ReCo_json.json'
    canvas_size = 1024
    path_save = r'Png_debut'

    # list_city = []  # 1-60
    #
    # plotting = {}
    # with open(path_json, encoding='utf-8') as f:
    #     data = json.load(f)
    #
    #     for example in data:
    #         # list_city.append(int(example['city'].split('_')[1]))
    #
    #         if example['_id'] == _id_to_draw:
    #             plotting = example
    #             break
    #
    #         # print(example)
    #         # plotting = example
    #         # print(plotting)
    #         #
    #         # for key, value in plotting.items():
    #         #     print(key + ':')
    #         #     print(value)
    #
    # plot_community(plotting, canvas_size, path_save)
    # plt.show()
    #
    # # print(list(set(list_city)))  # 1-60

    count_continue = 33929
    sum_all = 0
    with open(path_json, encoding='utf-8') as f:
        data = json.load(f)
        for index, example in enumerate(data):
            if index >= count_continue:
                print(index, '/', len(data))
                result = plot_community(example, canvas_size, path_save)
                # if result:
                #     sum_all += 1
                #     # print(sum_all)
