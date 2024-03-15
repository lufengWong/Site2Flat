# -*- coding: utf-8 -*-
# @Time    : 2023/3/8 23:48
# @Author  : Lufeng Wang
# @WeChat  : tofind404
# @File    : utils.py
# @Software: PyCharm


size_pix = 256
size_grid = 192
value_step = 100

elevator_min_length = 1900  # 近似正方形 2400
stair_area = (2.6+0.4) * (7.6+0.4)  # 需要整除取整

list_reverse = [0, 1]  # 正反对称 (左右)
list_angles = [0, 1, 2, 3]  # 旋转的角度0 90 180 270 逆时针

# list_channel_1_plus_3_rot = [
#     np.rot90(np.fliplr(dict_data_100channel1_plus_channel2[list_ids_candidate[index]]), rot[1]) if rot[0] == 1
#     else np.rot90(dict_data_100channel1_plus_channel2[list_ids_candidate[index]], rot[1])
#     for index, rot in zip(regions_chosen[0], regions_chosen[1])]

png_suffix = '.png'
graph_suffix = '.npy'

room_label = [
    (0, 'Apartment'),
    (1, 'Ladder'),  # elevator
    (2, 'Lift'),  # stair
    (3, 'Public'),

    (4, 'External'),
    (5, 'ExteriorWall'),
    (6, 'InteriorWall'),
]

category = [category for category in room_label if category[1] not in {'External', 'ExteriorWall', 'InteriorWall'}]

num_category = len(category)  # 数据长度
num_info_boundary = 3  # boundary inside sum_category

mask_size = 2

def log(file, msg='', is_print=True):
    if is_print:
        print(msg)
    file.write(msg + '\n')
    file.flush()
