import os
import pickle
import random
import shutil
import numpy as np

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import numpy as np
import shutil
import pickle
import utils
import os

import sys

from tqdm import tqdm

import matplotlib.patches as mpatches

from scipy import misc
import matplotlib.pyplot as pyplot

import cv2 as cv

from torchvision import transforms


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


if __name__ == '__main__':

    # 先分照片 ##################################
    path_pics_all = r'F:\pix2pix\dataset_building\clear_all'

    # path_train_debut = r'F:\pix2pix\dataset_building\zjkj\train_debut'
    path_train = r'F:\pix2pix\dataset_building\train'
    # path_val_debut = r'F:\pix2pix\dataset_building\zjkj\val_debut'
    path_val = r'F:\pix2pix\dataset_building\val'
    # path_tst_debut = r'F:\pix2pix\dataset_building\zjkj\tst_debut'
    path_tst = r'F:\pix2pix\dataset_building\test'

    path_train_debut = path_train
    path_tst_debut = path_tst
    path_val_debut = path_val

    if os.path.exists(path_train_debut):
        shutil.rmtree(path_train_debut)
    os.mkdir(path_train_debut)

    if os.path.exists(path_val_debut):
        shutil.rmtree(path_val_debut)
    os.mkdir(path_val_debut)

    if os.path.exists(path_tst_debut):
        shutil.rmtree(path_tst_debut)
    os.mkdir(path_tst_debut)

    for name in tqdm([pth_path for pth_path in os.listdir(path_pics_all)]):
        possibility = random.random()
        if possibility < 0.8:
            shutil.copy(os.path.join(path_pics_all, name), os.path.join(path_train_debut, name))
        elif possibility > 0.9:
            shutil.copy(os.path.join(path_pics_all, name), os.path.join(path_val_debut, name))
        else:
            shutil.copy(os.path.join(path_pics_all, name), os.path.join(path_tst_debut, name))

    # 数据增强
    path_files = path_tst_debut
    for name in tqdm([pth_path for pth_path in os.listdir(path_files)]):
        path_png = os.path.join(path_files, name)
        name_split = name.split('.')[0]

        img = Image.open(path_png)

        # transform_ver = transforms.RandomVerticalFlip(p=1)
        # img_ver = transform_ver(img)
        # name_ver = str(name_split) + '_ver.png'
        # img_ver.save(os.path.join(path_files, name_ver))
        list_angle = [90, 180, 270]
        for angle_rot in list_angle:
            transform_ro_45 = transforms.RandomRotation(degrees=(angle_rot, angle_rot))
            img_ro_45 = transform_ro_45(img)
            name_ro_45 = str(name_split) + '_' + str(angle_rot) + '.png'
            img_ro_45.save(os.path.join(path_files, name_ro_45))

        transform_hor = transforms.RandomHorizontalFlip(p=1)
        img_hor = transform_hor(img)
        name_hor = str(name_split) + '_hor.png'
        img_hor.save(os.path.join(path_files, name_hor))

    # 转换为目格式 #####################################

    path_train_debut_pics = path_tst_debut
    path_train_pics = path_tst

    if os.path.exists(path_train_pics):
        shutil.rmtree(path_train_pics)
    os.mkdir(path_train_pics)

    floor_plans_names = [pth_path for pth_path in os.listdir(path_train_debut_pics)]

    percent_train = 1

    list_img_array = []
    for name in tqdm(floor_plans_names[0:]):

        if random.random() <= percent_train:

            find_error = False

            path_pic = os.path.join(path_train_debut_pics, name)
            img = Image.open(path_pic)
            # img.show()
            array_img = np.array(img)
            # print(array_img.shape)

            if any(np.array_equal(array_img[:, :, 1], i) for i in list_img_array):
                print('repeat')
                find_error = True

            # 做一个基本的判断 要含有所有的要素
            for i in range(0, 7):
                if not array_img[:, :, 1].__contains__(i):
                    print('error')
                    find_error = True

            if not find_error:
                list_img_array.append(array_img[:, :, 1])
                array_inter = np.ones((array_img.shape[0], array_img.shape[1]))
                array_inter = array_inter * 150  # value 150

                list_index_change = [6, 2, 3, 5, 1]
                list_value_dilate = [300, 100, 0, 200, 250]
                list_ratio_dilate = [3, 5, 5, 3, 5]

                # list_index_change = [6, 5]
                # list_value_dilate = [300, 200]
                # list_ratio_dilate = [3, 3]

                for index, value, ratio in zip(list_index_change, list_value_dilate, list_ratio_dilate):
                    array_temp = np.zeros((array_img.shape[0], array_img.shape[1]))
                    array_temp[array_img[:, :, 1] == index] = 1
                    array_temp = array_temp.astype(np.uint8)
                    kernel = np.ones((ratio, ratio), np.uint8)
                    img_dilate = cv.dilate(array_temp, kernel, iterations=1)
                    array_inter[img_dilate == 1] = value

                # # array_inter[array_img[:, :, 1] == 3] = 150
                # array_inter[array_img[:, :, 1] == 2] = 100
                # array_inter[array_img[:, :, 1] == 1] = 50

                array_inter = array_inter.astype(np.uint8)
                image_inter_gray = cv.cvtColor(array_inter, cv.COLOR_GRAY2BGR)
                image_inter_color = cv.applyColorMap(image_inter_gray, cv.COLORMAP_RAINBOW)
                # cv.imwrite('12.png', image_np_2)
                # img = Image.open('12.png')
                # img.show()

                array_inter = np.ones((array_img.shape[0], array_img.shape[1]))
                array_inter = array_inter * 150  # value 150
                array_inter[array_img[:, :, 3] == 255] = 200

                array_inter = array_inter.astype(np.uint8)

                image_out_gray = cv.cvtColor(array_inter, cv.COLOR_GRAY2BGR)
                image_out_color = cv.applyColorMap(image_out_gray, cv.COLORMAP_RAINBOW)

                # cv.imwrite('13.png', image_out_2)
                # img = Image.open('13.png')
                # img.show()

                image_1_2 = np.concatenate([image_inter_color, image_out_color], 1)
                path_new_pic = os.path.join(path_train_pics, name)
                cv.imwrite(path_new_pic, image_1_2)
