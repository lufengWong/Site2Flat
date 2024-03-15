import copy
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
# import utils
import os

import sys
import multiprocessing as mp
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


def multichannel2AB_connection(path_pics, path_target_pics, num_count):
    """
    转为目标格式
    :param path_pics:
    :param path_target_pics:
    :return:
    """
    # # 颜色策略

    print(num_count)

    path_pic = path_pics
    name_i = os.path.basename(path_pics)
    array_img = np.array(Image.open(path_pic))  # 放大倍数

    # 轮廓
    array_img_0 = copy.deepcopy(array_img[:, :, 0]) * 1
    image_out_color = np.stack((array_img_0, array_img_0, array_img_0), axis=2)
    # image_out_color = cv.applyColorMap(image_out_color, cv.COLORMAP_OCEAN)

    # 分区
    expend_color = 12  # ###############
    array_img_1 = copy.deepcopy(array_img[:, :, 1]) * expend_color
    image_inter_color = np.stack((array_img_1, array_img_1, array_img_1), axis=2)

    # pair
    image_1_2 = np.concatenate([image_inter_color, image_out_color], 1)

    # # 仅仅是可视化
    # plt.matshow(image_1_2[:,:,0], cmap=plt.cm.rainbow)  # Set3
    # plt.show()

    # save
    path_target_pics_img = os.path.join(path_target_pics, 'image')
    path_new_pic = os.path.join(path_target_pics_img, name_i)
    cv.imwrite(path_new_pic, image_1_2)

    # condition
    # 门的位置 客厅面积  卧室面积 ...
    array_channel4 = copy.deepcopy(array_img[:, :, 3])
    index_interior = np.where(array_channel4 == 255)  # 内部区域，然后找质心
    centroid = [int(np.mean(index_interior[0])), int(np.mean(index_interior[1]))]

    # # 寻找门的坐标
    # array_channel1 = copy.deepcopy(array_img[:, :, 1])
    # index_front_door = np.where(array_channel1 == 15)
    # v_direction, h_direction = index_front_door
    # num_piex_v = len(list(set(v_direction)))
    # mean_piex_v = np.mean(v_direction)
    # num_piex_h = len(list(set(h_direction)))
    # mean_piex_h = np.mean(h_direction)
    # # 判断门所在的位置
    # door_edge = 0
    # if num_piex_v >= num_piex_h:  # 为竖直
    #     if mean_piex_h <= centroid[0]:
    #         door_edge = 3
    #     else:
    #         door_edge = 1
    # elif num_piex_v < num_piex_h:
    #     if mean_piex_v <= centroid[1]:
    #         door_edge = 4
    #     else:
    #         door_edge = 2

    # 面积
    pixel2length_rplan = 18 / 256  # m
    img_channel_1 = copy.deepcopy(np.array(array_img[:, :, 1])).astype(np.uint16)
    img_channel_2 = copy.deepcopy(np.array(array_img[:, :, 2])).astype(np.uint16) # 4 5 7
    room_count = np.unique(img_channel_2)  # 没有意义 房间从1开始 [0,1,2,3,4,5,6,7]
    array_100channel1_plus_channel2 = 100 * img_channel_1 + img_channel_2
    data = array_100channel1_plus_channel2

    # 客厅的面积 大于等于1 小于100
    index_max = 100
    index_min = 1
    number_living = list(set(data[data < index_max])-set(data[data < index_min]))
    area_living = [int(len(np.where(data == id_rec)[0]) * (pixel2length_rplan ** 2)) for id_rec in number_living]
    area_living_sum = sum(area_living)

    # 主卧
    index_max = 200
    index_min = 100
    number_living = list(set(data[data < index_max]) - set(data[data < index_min]))
    area_living = [int(len(np.where(data == id_rec)[0]) * (pixel2length_rplan ** 2)) for id_rec in number_living]
    area_mast_room_sum = sum(area_living)

    # 客厅的面积 大于等于1 小于100
    index_max = 300
    index_min = 200
    number_living = list(set(data[data < index_max]) - set(data[data < index_min]))
    area_living = [int(len(np.where(data == id_rec)[0]) * (pixel2length_rplan ** 2)) for id_rec in number_living]
    area_kitchen_sum = sum(area_living)

    list_txt = [area_living_sum, area_mast_room_sum, area_kitchen_sum]
    path_label = os.path.join(path_target_pics, 'label')
    name_label = name_i.split('.')[0]+'.npy'

    # print(list_txt)
    np.save(os.path.join(path_label, name_label), np.array(list_txt, dtype=object))

    return True


def write2pair_pool(all_debut_dir, pkl_dir):
    """
    并行处理pkl
    """
    # train_data_path = [os.path.join(all_debut_dir, path) for path in os.listdir(all_debut_dir)]

    train_data_names_path = [os.path.join(all_debut_dir, name) for name in os.listdir(all_debut_dir)]
    target_path = [pkl_dir for i in range(len(train_data_names_path))]
    print(f'Number of dataset: {len(train_data_names_path)}')

    pool = mp.Pool(12)
    result = [pool.apply_async(multichannel2AB_connection, args=(all_debut_dir_1, pkl_dir_1, num_count))
              for all_debut_dir_1, pkl_dir_1, num_count in
              zip(train_data_names_path, target_path, range(len(target_path)))]

    [p.get() for p in result]


def divide_pics(pics_all, train_folder, val_folder, test_folder, divide_ratio):
    """
    分图片为训练集、测试集、验证集
    :return:
    """
    for path_i in [train_folder, val_folder, test_folder]:

        if os.path.exists(path_i):
            shutil.rmtree(path_i)
        os.mkdir(path_i)

    for name in tqdm([pth_path for pth_path in os.listdir(pics_all)]):
        possibility = random.random()
        if possibility < divide_ratio[0]:
            shutil.copy(os.path.join(pics_all, name), os.path.join(train_folder, name))
        elif possibility > divide_ratio[0] + divide_ratio[1]:
            shutil.copy(os.path.join(pics_all, name), os.path.join(val_folder, name))
        else:
            shutil.copy(os.path.join(pics_all, name), os.path.join(test_folder, name))


if __name__ == '__main__':
    # 测试主函数 #
    # path_pic = r'C:\Users\18328\Desktop\TrabSD-GAN-git\Make_dataset_flat\RPLAN_example\1.png'
    # multichannel2AB_connection(path_pic,
    #                            r'C:\Users\18328\Desktop\TrabSD-GAN-git\Make_dataset_flat\dataset_rplan',
    #                            1)

    # 批量的数据转换
    # path_pics_all_1 = r'C:\Users\18328\Desktop\TrabSD-GAN-git\Make_dataset_flat\RPLAN_example'
    # target_pics_all_1 = r'C:\Users\18328\Desktop\TrabSD-GAN-git\Make_dataset_flat\dataset_rplan'
    # write2pair_pool(path_pics_all_1, target_pics_all_1)

    # step3 转换为目标格式 #####################################
    path_train_debut_pics = r'\floorplan_dataset'  # 不同的文件
    path_train_pics = r'\train' # 用于训练的文件夹
    path_val_pics = r'\val'  # 用于训练的文件夹
    file_img = 'image'
    file_label = 'label'

    if os.path.exists(path_train_pics):
        shutil.rmtree(path_train_pics)
    os.mkdir(path_train_pics)

    if os.path.exists(path_val_pics):
        shutil.rmtree(path_val_pics)
    os.mkdir(path_val_pics)

    path_train_pics_img = os.path.join(path_train_pics, file_img)
    if os.path.exists(path_train_pics_img):
        shutil.rmtree(path_train_pics_img)
    os.mkdir(path_train_pics_img)

    path_train_pics_label = os.path.join(path_train_pics, file_label)
    if os.path.exists(path_train_pics_label):
        shutil.rmtree(path_train_pics_label)
    os.mkdir(path_train_pics_label)

    path_train_pics_img = os.path.join(path_val_pics, file_img)
    if os.path.exists(path_train_pics_img):
        shutil.rmtree(path_train_pics_img)
    os.mkdir(path_train_pics_img)

    path_train_pics_label = os.path.join(path_val_pics, file_label)
    if os.path.exists(path_train_pics_label):
        shutil.rmtree(path_train_pics_label)
    os.mkdir(path_train_pics_label)

    floor_plans_names = [pth_path for pth_path in os.listdir(path_train_debut_pics)]

    percent_train = 0.8

    for count, name in tqdm(enumerate(floor_plans_names[0:])):
        path_pic = os.path.join(path_train_debut_pics, name)

        if random.random() <= percent_train:
            multichannel2AB_connection(path_pic,
                                       path_train_pics,
                                       count)
        else:
            multichannel2AB_connection(path_pic,
                                       path_val_pics,
                                       count)

    # #
    # path_train_1 = r'F:\z-hxs-Gan\dataset_colorBlock\train'
    # path_val_1 = r'F:\z-hxs-Gan\dataset_colorBlock\val'
    # # path_test_1 = r'F:\z-hxs-Gan\dataset_colorBlock\test'
    #
    #
    # divide_pics(target_pics_all_1, path_train_1, path_val_1, path_test_1, [0.3, 0.3, 0.4])
    #
    # # pool = multiprocessing.Pool(12)
    # # result = [pool.apply_async(), args =]
    #
    #
    # # # 数据处理
    # # path_files = path_pics_all_1
    # # path_files_new = ''
    # #
    # # for name in tqdm([pth_path for pth_path in os.listdir(path_files)]):
    # #     path_png = os.path.join(path_files, name)
    # #     name_split = name.split('.')[0]
    # #
    # #     img = Image.open(path_png)
    # #
    # #     # transform_ver = transforms.RandomVerticalFlip(p=1)
    # #     # img_ver = transform_ver(img)
    # #     # name_ver = str(name_split) + '_ver.png'
    # #     # img_ver.save(os.path.join(path_files, name_ver))
    # #     list_angle = [90, 180, 270]
    # #     for angle_rot in list_angle:
    # #         transform_ro_45 = transforms.RandomRotation(degrees=(angle_rot, angle_rot))
    # #         img_ro_45 = transform_ro_45(img)
    # #         name_ro_45 = str(name_split) + '_' + str(angle_rot) + '.png'
    # #         img_ro_45.save(os.path.join(path_files, name_ro_45))
    # #
    # #     transform_hor = transforms.RandomHorizontalFlip(p=1)
    # #     img_hor = transform_hor(img)
    # #     name_hor = str(name_split) + '_hor.png'
    # #     img_hor.save(os.path.join(path_files, name_hor))
    #
    # # # 转换为目格式 #####################################
