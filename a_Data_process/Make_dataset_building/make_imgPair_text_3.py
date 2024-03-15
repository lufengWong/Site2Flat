# -*- coding: utf-8 -*-
# @Time    : 2023/7/19 11:38
# @Author  : Lufeng Wang
# @WeChat  : tofind404
# @File    : make_imgPair_text.py
# @Software: PyCharm
import copy
import random

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import shutil
# import utils
import os

from tqdm import tqdm

import matplotlib.patches as mpatches

import cv2 as cv

import utils


def show_array(array_img, name):
    """
    矩阵， 图像
    :param array_img:
    :param name:
    :return:
    """
    im = plt.imshow(array_img, cmap='rainbow')

    values = np.unique(array_img.ravel())
    # get the colors of the values, according to the
    # colormap used by imshow
    colors = [im.cmap(im.norm(value)) for value in values]
    # create a patch (proxy artist) for every color
    patches = [mpatches.Patch(color=colors[i], label="Label {l}".format(l=int(values[i]))) for i in range(len(values))]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    # plt.grid(True)
    plt.title(name)


def get_apartment_area(data):
    """
    获取户型面积的列表，获取电梯个数列表
    :param data:
    :return:
    """
    # apartments
    number_apartments = list(set(data[data < utils.value_step]))  # 1,2,3

    apartments_area = [int(len(np.where(data == id_rec)[0]) * (utils.size_grid ** 2) / (1000 ** 2)) for id_rec in
                       number_apartments]

    # elevator[1,1] [2]
    number_elevator = list(set(data[data < utils.value_step * 2]) - set(number_apartments))
    elevator_count = [int(len(np.where(data == id_rec)[0]) * (utils.size_grid ** 2) / (1000 ** 2) // ((
                                                                                                                  utils.elevator_min_length / 1000) ** 2))
                      if len(np.where(data == id_rec)[0]) * (utils.size_grid ** 2) / (1000 ** 2) // ((
                                                                                                                 utils.elevator_min_length / 1000) ** 2) >= 1
                      else 1
                      for id_rec in number_elevator]  # 保证不会为0

    # stair
    number_stair = list(set(data[data < utils.value_step * 3])
                        - set(data[data < utils.value_step * 2]))
    # 可能有多个1
    stair_count = [int(len(np.where(data == id_rec)[0]) * (utils.size_grid ** 2) / (1000 ** 2) // utils.stair_area)
                      if len(np.where(data == id_rec)[0]) * (utils.size_grid ** 2) / (1000 ** 2) // utils.stair_area >= 1
                      else 1
                      for id_rec in number_stair]  # 保证不会为0

    return [stair_count, elevator_count, apartments_area]


if __name__ == '__main__':

    # step1 先分照片 ##################################
    path_pics_all = r'\dataset_png_all_clear'

    path_train_debut = r'\train_debut'
    path_train = r'\train'
    path_val_debut = r'\val_debut'
    path_val = r'\val'
    # path_tst_debut = r'F:\pix2pix\dataset_building\zjkj\tst_debut'
    # path_tst = r'F:\pix2pix\dataset_building\test'

    # path_train_debut = path_train
    # # path_tst_debut = path_tst
    # path_val_debut = path_val

    # if os.path.exists(path_train_debut):
    #     shutil.rmtree(path_train_debut)
    # os.mkdir(path_train_debut)
    #
    # if os.path.exists(path_val_debut):
    #     shutil.rmtree(path_val_debut)
    # os.mkdir(path_val_debut)
    #
    # # if os.path.exists(path_tst_debut):
    # #     shutil.rmtree(path_tst_debut)
    # # os.mkdir(path_tst_debut)
    #
    # for name in tqdm([pth_path for pth_path in os.listdir(path_pics_all)]):
    #     possibility = random.random()
    #     if possibility <= 0.8:
    #         shutil.copy(os.path.join(path_pics_all, name), os.path.join(path_train_debut, name))
    #     elif possibility > 0.8:
    #         shutil.copy(os.path.join(path_pics_all, name), os.path.join(path_val_debut, name))
    #     # else:
    #     #     shutil.copy(os.path.join(path_pics_all, name), os.path.join(path_tst_debut, name))
    #
    # # step2 数据增强
    # path_files = path_train  # 不同的数据文件夹进行增强
    # for name in tqdm([pth_path for pth_path in os.listdir(path_files)]):
    #     path_png = os.path.join(path_files, name)
    #     name_split = name.split('.')[0]
    #
    #     img = Image.open(path_png)
    #
    #     # transform_ver = transforms.RandomVerticalFlip(p=1)
    #     # img_ver = transform_ver(img)
    #     # name_ver = str(name_split) + '_ver.png'
    #     # img_ver.save(os.path.join(path_files, name_ver))
    #     list_angle = [90, 180, 270]
    #     for angle_rot in list_angle:
    #         transform_ro_45 = transforms.RandomRotation(degrees=(angle_rot, angle_rot))
    #         img_ro_45 = transform_ro_45(img)
    #         name_ro_45 = str(name_split) + '_' + str(angle_rot) + '.png'
    #         img_ro_45.save(os.path.join(path_files, name_ro_45))
    #
    #     transform_hor = transforms.RandomHorizontalFlip(p=1)
    #     img_hor = transform_hor(img)
    #     name_hor = str(name_split) + '_hor.png'
    #     img_hor.save(os.path.join(path_files, name_hor))

    # step3 转换为目标格式 #####################################
    path_train_debut_pics = path_val_debut  # 不同的文件
    path_train_pics = path_val  # 用于训练的文件夹
    file_img = 'image'
    file_label = 'label'

    if os.path.exists(path_train_pics):
        shutil.rmtree(path_train_pics)
    os.mkdir(path_train_pics)

    path_train_pics_img = os.path.join(path_train_pics, file_img)
    if os.path.exists(path_train_pics_img ):
        shutil.rmtree(path_train_pics_img )
    os.mkdir(path_train_pics_img )

    path_train_pics_label = os.path.join(path_train_pics, file_label)
    if os.path.exists(path_train_pics_label):
        shutil.rmtree(path_train_pics_label)
    os.mkdir(path_train_pics_label)

    floor_plans_names = [pth_path for pth_path in os.listdir(path_train_debut_pics)]

    percent_train = 1  # train 已经自己分为了两部分了

    list_img_array = []
    for name in tqdm(floor_plans_names[0:]):

        if random.random() <= percent_train:

            find_error = False

            path_pic = os.path.join(path_train_debut_pics, name)
            img = Image.open(path_pic)
            # img.show()
            array_img = np.array(img)

            if any(np.array_equal(array_img[:, :, 1], i) for i in list_img_array):
                print('repeat')
                find_error = True

            # 做一个基本的判断 要含有所有的要素
            for i in range(0, 7):
                if not array_img[:, :, 1].__contains__(i):
                    print('error')
                    find_error = True

            if not find_error:
                list_img_array.append(array_img[:, :, 1])  # 存入库中

                # list_index_change = [6, 5]
                # list_value_dilate = [300, 200]
                # list_ratio_dilate = [3, 3]

                # 在此处统计面积 ################
                # 100*channel2 + channel3 合成为一个矩阵 # np.int64
                array_txt = 100 * array_img[:, :, 1].astype(np.int64) + array_img[:, :, 2].astype(np.int64)

                [num_stair, num_elevator, area_flat] = get_apartment_area(array_txt)

                list_txt = [0] * 3
                list_txt[0] = sum(num_stair)
                list_txt[1] = sum(num_elevator)
                list_txt[2] = len(area_flat)  # 改为个数

                path_label = os.path.join(path_train_pics, file_label)
                name_label = str(name).split()[0].split('.')[0]
                np.save(os.path.join(path_label, name_label + '.npy'),
                        np.array(list_txt, dtype=object))

                # 制作图像 #######################
                array_inter = np.ones((array_img.shape[0], array_img.shape[1]))
                array_inter = array_inter * 4  # value 150  # 默认值为150

                list_index_change = [6, 2, 3, 5, 1, 0]
                # list_value_dilate = [300, 100, 0, 200, 250]
                list_value_dilate = [6, 2, 3, 5, 1, 0]

                list_ratio_dilate = [3, 5, 5, 3, 1, 1]

                # 改变value值
                for index, value, ratio in zip(list_index_change, list_value_dilate, list_ratio_dilate):
                    array_temp = np.zeros((array_img.shape[0], array_img.shape[1]))  # 建立一个全是空的矩阵
                    array_temp[array_img[:, :, 1] == index] = 1  # 找出对应不同索引的位置
                    array_temp = array_temp.astype(np.uint8)

                    kernel = np.ones((ratio, ratio), np.uint8)
                    img_dilate = cv.dilate(array_temp, kernel, iterations=1)  # 进行形态学的操作
                    array_inter[img_dilate == 1] = value  # 将其替换为相应的值

                # 输出 (内部的区域)
                array_inter = array_inter.astype(np.uint8) * 32  # ################ 32 ############3
                image_inter_color = np.stack((array_inter, array_inter, array_inter), axis=2)
                # # 仅仅是可视化
                # plt.matshow(array_inter, cmap=plt.cm.tab20c) # Set3
                # plt.show()

                # image_inter_gray = cv.cvtColor(array_inter, cv.COLOR_GRAY2BGR)
                # image_inter_color = cv.applyColorMap(image_inter_gray, cv.COLORMAP_RAINBOW)
                # cv.imwrite('12.png', image_np_2)
                # img = Image.open('12.png')
                # img.show()

                # 输入（轮廓）
                array_img_0 = copy.deepcopy(array_img[:, :, 0]) * 1  # ##### 可以改变使用不同的颜色

                kernel = np.ones((3, 3), np.uint8)
                array_img_0 = cv.dilate(array_img_0, kernel, iterations=1)  # 进行形态学的操作

                array_out = copy.deepcopy(array_img_0).astype(np.uint8)
                image_out_color = np.stack((array_out, array_out, array_out), axis=2)

                # # 仅仅是可视化
                # plt.matshow(array_out, cmap=plt.cm.tab20c)  # Set3
                # plt.show()

                image_1_2 = np.concatenate([image_inter_color, image_out_color], 1)

                # # 仅仅是可视化
                # plt.matshow(image_1_2[:,:,0], cmap=plt.cm.tab20c)  # Set3
                # plt.show()

                path_pics = os.path.join(path_train_pics, file_img)
                path_new_pic = os.path.join(path_pics, name)
                cv.imwrite(path_new_pic, image_1_2)
