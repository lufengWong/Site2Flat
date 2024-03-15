import copy
import os
import sys
#
# sys.path.append("..")
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# sys.path.append(os.getcwd())

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import cv2
import networkx as nx
from PIL import Image
from sklearn.cluster import KMeans, DBSCAN

from RS_building_regularization_main.main_regularization import boundary_regularization


def contours_in(contours, Index_contour, size_img=256):
    p = np.zeros(shape=(size_img, size_img))
    cv2.drawContours(p, contours, Index_contour, 255, -1)
    # 绘制的目标图像，输入的轮廓组，指明第几个轮廓, 轮廓的颜色，负值表示轮廓内部填充

    # 仅仅是可视化
    # plt.matshow(p, cmap=plt.cm.tab10)
    # plt.title('region')
    # plt.show()

    a = np.where(p == 255)[0].reshape(-1, 1)
    b = np.where(p == 255)[1].reshape(-1, 1)
    coordinate = np.concatenate([a, b], axis=1).tolist()
    inside = [tuple(x) for x in coordinate]
    return inside


def extract_site(path_png, path_save, step_floor_value=6.0, length_ture_img=1024):
    name_pic = os.path.basename(path_png).split('.')[0]

    array_img = copy.deepcopy(np.array(Image.open(path_png)))
    img_layout = array_img[:, :256, :]

    # 变为1024的，便于操作
    img_layout = cv2.resize(img_layout,
                           dsize=(length_ture_img, length_ture_img),
                           fx=1, fy=1,
                           interpolation=cv2.INTER_NEAREST)

    # 删除边界, 新的数据集是没有边界的
    pixel_value_boundary = 127  # 边界的像素值为127
    img_layout[img_layout == pixel_value_boundary] = 0
    # ############################

    array_layout = np.mean(img_layout, axis=2, dtype=np.uint16)
    img_layout = copy.deepcopy(np.stack((array_layout, array_layout, array_layout), axis=2))

    # 仅仅是可视化 # 生成的像素
    plt.matshow(array_layout, cmap=plt.cm.tab10)
    plt.title('array_layout')
    plt.show()

    # BR
    ori_img = copy.deepcopy(img_layout.astype(np.uint8))
    ori_img = cv2.medianBlur(ori_img, 5)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY) # 此时都还在

    # plt.matshow(ori_img, cmap=plt.cm.tab10)
    # plt.title('ori_img')
    # plt.show()

    ret, ori_img = cv2.threshold(ori_img, 0, 255, cv2.THRESH_BINARY) # | cv2.THRESH_OTSU

    # plt.matshow(ori_img, cmap=plt.cm.tab10)
    # plt.title('ori_img_binary')
    # plt.show()

    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ori_img, connectivity=4)

    # labels 其实就是区域
    list_buildings = []
    # 遍历联通域
    for i in range(1, num_labels):
        img = np.zeros_like(labels)
        index = np.where(labels == i)
        img[index] = 255
        img = np.array(img, dtype=np.uint8)

        # # 仅仅是可视化
        # plt.matshow(img, cmap=plt.cm.tab10)
        # plt.title('img')
        # plt.show()

        regularization_contour = boundary_regularization(img, epsilon=6).astype(np.int32)  # 找到了边界

        # 需要找到内部的高度
        list_height = [array_layout[x, y] for (x, y) in zip(index[0].tolist(), index[1].tolist())]

        list_height_pure = [height for height in list_height if (height !=0 and height % step_floor_value == 0)]
        # maxlabel = max(list_height, key=list_height.count) # 寻找列表中出现最多的值
        maxlabel = np.mean(list_height_pure)
        height_floor = maxlabel / step_floor_value
        floor_int = int(np.around(height_floor, decimals=0))

        # 数据储存
        list_buildings.append({'polygon': regularization_contour, 'floor': floor_int})

    np.save(os.path.join(path_save, name_pic + '.npy'), np.array(list_buildings, dtype=object))

    return list_buildings


def merge_buildings(list_buildings_poly,):
    print('debut')
    list_area = []
    list_centroid = []
    for building in list_buildings_poly:
        print(building['polygon'], building['floor'])
        buildings_poly = Polygon(np.array(building['polygon']).reshape(-1, 2))
        list_area.append(buildings_poly.area)
        list_centroid.append(np.array(buildings_poly.centroid.xy).reshape(-1,2).squeeze().tolist())
        plt.fill(*buildings_poly.exterior.xy, color='darkorange', alpha = building['floor'] * 0.03)

    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()

    # # 根据面积进行聚类
    # # 选择中值的几何为代表几何
    # clustering = DBSCAN(eps=0.5, min_samples=2).fit(np.array(list_area).reshape(-1, 1))
    # list_label = clustering.labels_.tolist()
    # set_label = set(list_label)
    # for label in set_label:
    #     index_need = [i for i, x in enumerate(list_label) if x == label]
    #     area_need = [list_area[i] for i in index_need]
    #     sort_area = area_need.sort()
    #     median_exist = len(area_need) // 2 + 1
    #     are_selected = sort_area[median_exist]

    print('debuge')


if __name__ == '__main__':


    list_buildings_1 = extract_site(
        r'\city_11_61ef8a9332b5d4672152e27e.png',
        r'\Convert_site\Txt_polygon',
    )
    # merge_buildings(list_buildings_1)
