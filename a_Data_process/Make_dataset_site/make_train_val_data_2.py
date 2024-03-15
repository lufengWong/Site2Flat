# -*- coding: utf-8 -*-
import os
import copy
import random
import shutil

import numpy as np
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import json
import cv2

from PIL import Image

import cv2 as cv


# 制作输入输出和标签
# 输入的图像是 边界
# 输入的条件是 城市，建筑总面积，建筑平均高度
# 输出是建筑平面
# 图像尺寸是 256

def multichannel2AB_connection(path_pic, path_target, img_size_input, num_count):
    """
    转为目标格式
    :param path_pic:
    :param path_target_pics:
    :return:
    """

    # print(num_count)

    name_pic = os.path.basename(path_pic)
    city_num = int(name_pic.split('_')[1])

    array_img = np.array(Image.open(path_pic))

    # 制作输入图像
    building_bd = array_img[:, :, 2]   # 对有数值的地方进行了操作 空白地方是255这个挺好的
    input_img = np.stack((building_bd, building_bd, building_bd), axis=2)
    array_in = copy.deepcopy(input_img).astype(np.uint8)

    image_in_color = cv2.resize(array_in,
                                dsize=(img_size_input, img_size_input),
                                fx=1, fy=1,
                                interpolation=cv2.INTER_LINEAR) * 127  # 边界值为127

    # # 仅可视化
    # plt.matshow(image_in_color[:, :, 0], cmap=plt.cm.tab20b)  # Set3
    # plt.title('boundary')
    # plt.show()

    # 制作输出图像
    building_region = array_img[:, :, 0] * 6  # 对有数值的地方进行了操作 空白地方是255这个挺好的
    output_img = np.stack((building_region, building_region, building_region), axis=2)
    array_out = copy.deepcopy(output_img).astype(np.uint8)

    image_out_color = cv2.resize(array_out,
                                 dsize=(img_size_input, img_size_input),
                                 fx=1, fy=1,
                                 interpolation=cv2.INTER_NEAREST)

    #  添加边界，此时才添加，省得resize出问题 0731 不要边界便于识别 0802
    # for ch in range(3):
    #     image_out_color[:, :, ch][image_in_color[:, :, 0] == 127] = 127

    # # 仅可视化
    # plt.matshow(image_out_color[:, :, 0], cmap=plt.cm.tab10)  # Set3
    # plt.show()

    # cv 显示灰度图
    # image_out_color[:, :, 0][image_out_color[:, :, 0]==0] = 255
    # cv2.imshow('gray',image_out_color[:, :, 0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # pair
    image_1_2 = np.concatenate([image_out_color, image_in_color], 1)

    # # 仅仅是可视化
    # plt.matshow(image_1_2[:,:,0], cmap=plt.cm.tab10)  # Set3
    # plt.show()

    # save
    path_target_pics_img = os.path.join(path_target, 'image')
    path_new_pic = os.path.join(path_target_pics_img, name_pic)
    cv.imwrite(path_new_pic, image_1_2)

    # 制作条件txt
    list_txt = [city_num]
    # 建筑的总面积
    num_nozero = np.count_nonzero(array_img[:, :, 0])  # 实际建筑的面积
    area_build_sum = num_nozero * 1 * 1
    list_txt.append(area_build_sum)
    # 平均的层数
    sum_height = np.sum(array_img[:, :, 0])
    mean_height = sum_height // area_build_sum
    list_txt.append(mean_height)
    # save
    path_label = os.path.join(path_target, 'label')
    name_label = name_pic.split('.')[0] + '.npy'
    np.save(os.path.join(path_label, name_label), np.array(list_txt, dtype=object))

    # print('end')


if __name__ == '__main__':


    path_pngs_debut = r'\Png_debut'
    path_dataset_pics = r'\Used'  # 用于训练的文件夹
    file_img = 'image'
    file_label = 'label'
    img_size_need = 256
    percent_train = 0.8

    # 文件夹
    if os.path.exists(path_dataset_pics):
        shutil.rmtree(path_dataset_pics)
    os.mkdir(path_dataset_pics)

    for folder_1 in ['train', 'val']:

        path_train_pics_img = os.path.join(path_dataset_pics, folder_1)
        if os.path.exists(path_train_pics_img):
            shutil.rmtree(path_train_pics_img)
        os.mkdir(path_train_pics_img)

        path_train_img = os.path.join(path_train_pics_img, file_img)
        path_train_label = os.path.join(path_train_pics_img, file_label)

        if os.path.exists(path_train_img):
            shutil.rmtree(path_train_img)
        os.mkdir(path_train_img)

        if os.path.exists(path_train_label):
            shutil.rmtree(path_train_label)
        os.mkdir(path_train_label)
    # 开始转换
    for count, png in enumerate(os.listdir(path_pngs_debut)):

        print(count, '/', len(os.listdir(path_pngs_debut)))

        png_path = os.path.join(path_pngs_debut, png)

        if random.random() <= percent_train:  # train
            path_target_train = os.path.join(path_dataset_pics, 'train')
            multichannel2AB_connection(png_path, path_target_train, img_size_need, count)
        else:
            path_target_train = os.path.join(path_dataset_pics, 'val')
            multichannel2AB_connection(png_path, path_target_train, img_size_need, count)




