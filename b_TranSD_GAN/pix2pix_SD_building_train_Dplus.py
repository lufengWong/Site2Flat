import os
import shutil
import copy

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image
from torch import functional as F
from tqdm import tqdm

from evaluate import SegmentationMetric
from PIL import Image

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import mean_squared_error as MSE

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'Model'))

from Model.Model_Unet_all_condition_tran_Dplus import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(" -- 使用GPU进行训练 -- ")


def log(file, msg='', is_print=True):
    """
    储存数据
    """
    if is_print:
        print(msg)
    file.write(msg + '\n')
    file.flush()
    ##############


class MyDataset(Dataset):
    def __init__(self, root, subfolder, transform=None):
        """
        自定义数据集初始化
        :param root: 数据文件根目录
        :param subfolder: 数据文件子目录
        :param transform: 预处理方法
        """
        super(MyDataset, self).__init__()
        self.path = os.path.join(root, subfolder)

        # image
        self.path_imgs = os.path.join(self.path, 'image')
        self.image_list = [x for x in os.listdir(self.path_imgs)]
        # label
        self.path_labels = os.path.join(self.path, 'label')
        self.label_list = [x for x in os.listdir(self.path_labels)]

        self.transform = transform

    def __len__(self):
        """
        以便可以len(dataset)形式返回数据大小
        :return: 数据集大小
        """
        return len(self.image_list)

    def __getitem__(self, item):
        """
        支持索引以便dataset可迭代获取
        :param item: 索引
        :return: 索引对应的数据单元
        """
        image_path = os.path.join(self.path_imgs, self.image_list[item])
        image = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR -> RGB
        if self.transform is not None:
            image = self.transform(image)

        # Dataset每个数据单元要求返回一个数据一个标签 此处标签无意义（但不能直接设为None）
        # lable = 'NONE'
        lable_path = os.path.join(self.path_labels, self.label_list[item])
        lable = np.load(lable_path, allow_pickle=True).tolist()  # BGR -> RGB
        lable = torch.tensor(lable).float()  # 转换数据
        return image, lable


## 加载数据（Facades） ##
def loadData(root, subfolder, batch_size, shuffle=True):
    """
    加载数据以返回DataLoader类型
    :param root: 数据文件根目录
    :param subfolder: 数据文件子目录
    :param batch_size: 批处理样本大小
    :param shuffle: 是否打乱数据（默认为是）
    :return: DataLoader类型的可迭代数据
    """
    # 数据预处理方式
    transform = transforms.Compose([
        transforms.ToTensor(),  # (H, W, C) -> (C, H, W) & (0, 255) -> (0, 1)
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # (0, 1) -> (-1, 1)
    ])
    # 创建Dataset对象
    dataset = MyDataset(root, subfolder, transform=transform)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


## 训练判别器 ##
def D_train(D: Discriminator, G: GeneratorUNet_ConTran, X, Label, BCELoss, optimizer_D):
    """
    训练判别器
    :param D: 判别器
    :param G: 生成器
    :param X: 未分隔的数据
    :param BCELoss: 二分交叉熵损失函数
    :param optimizer_D: 判别器优化器
    :return: 判别器的损失值
    """
    Label = Label.to(device)

    # 标签转实物（右转左）
    image_size = X.size(3) // 2
    x = X[:, :, :, image_size:].to(device)  # 标签图（右半部分）
    y = X[:, :, :, :image_size].to(device)  # 实物图（左半部分）
    xy = torch.cat([x, y], dim=1)  # 在channel维重叠 xy!=X  # 长度变为了维度，看两张图片，input和ground truth
    # 梯度初始化为0
    D.zero_grad()

    # 在真数据上 #################
    # D_output_r = D(xy).squeeze()  # 删除为1的维度  # 输出path30*30的概率特征图
    # 两个输出 ##
    D_output_r, D_output_label_r = D(xy)
    D_output_r = D_output_r.squeeze()  # 删除为1的维度  # 输出path30*30的概率特征图

    D_real_loss = BCELoss(D_output_r, torch.ones(D_output_r.size()).to(device)).to(device)  # 和全是1的差距 代入1就是一半了
    conSin_real = F.cosine_similarity(Label, D_output_label_r).to(device)  # [-1,1] # 因为实际情况都为正值，在同一个象限？
    conSin_real = F.tanh(conSin_real+1.0).to(device)
    D_real_label_loss = BCELoss(conSin_real, torch.ones(conSin_real.size()).to(device))

    # 在假数据上 #################
    G_output = G(x, Label)
    X_fake = torch.cat([x, G_output], dim=1)
    # D_output_f = D(X_fake).squeeze()  # path输出概率
    # 两个输出 ##
    D_output_f, D_output_label_f = D(X_fake)
    D_output_f = D_output_f.squeeze()

    D_fake_loss = BCELoss(D_output_f, torch.zeros(D_output_f.size()).to(device))  # 和全是0的差距
    conSin_fake = F.cosine_similarity(Label, D_output_label_f).to(device)
    conSin_fake = torch.mean(F.tanh(conSin_fake + 1.0).to(device))
    D_fake_label_loss = BCELoss(conSin_fake, torch.zeros(conSin_fake.size()).to(device))

    # 反向传播并优化 ############################
    D_loss = (D_real_loss + D_fake_loss) * 0.5 + (D_real_label_loss + D_fake_label_loss) * 0.5  # 判断器
    D_loss.backward()
    optimizer_D.step()

    return D_loss.data.item()


def D_val(D: Discriminator, G: GeneratorUNet_ConTran, X, Label, BCELoss, optimizer_D):
    """
    训练判别器
    :param D: 判别器
    :param G: 生成器
    :param X: 未分隔的数据
    :param BCELoss: 二分交叉熵损失函数
    :param optimizer_D: 判别器优化器
    :return: 判别器的损失值
    """
    Label = Label.to(device)
    # 标签转实物（右转左）
    image_size = X.size(3) // 2
    x = X[:, :, :, image_size:].to(device)  # 标签图（右半部分）
    y = X[:, :, :, :image_size].to(device)  # 实物图（左半部分）
    xy = torch.cat([x, y], dim=1)  # 在channel维重叠 xy!=X  # 长度变为了维度，看两张图片，input和ground truth
    # 梯度初始化为0
    D.zero_grad()

    # 在真数据上 #################
    # D_output_r = D(xy).squeeze()  # 删除为1的维度  # 输出path30*30的概率特征图
    # 两个输出 ##
    D_output_r, D_output_label_r = D(xy)
    D_output_r = D_output_r.squeeze()  # 删除为1的维度  # 输出path30*30的概率特征图

    D_real_loss = BCELoss(D_output_r, torch.ones(D_output_r.size()).to(device)).to(device)  # 和全是1的差距 代入1就是一半了
    conSin_real = F.cosine_similarity(Label, D_output_label_r).to(device)
    conSin_real = torch.mean(F.tanh(conSin_real + 1.0).to(device))
    D_real_label_loss = BCELoss(conSin_real, torch.ones(conSin_real.size()).to(device)).to(device)

    # 在假数据上 #################
    G_output = G(x, Label)
    X_fake = torch.cat([x, G_output], dim=1)
    # D_output_f = D(X_fake).squeeze()  # path输出概率
    # 两个输出 ##
    D_output_f, D_output_label_f = D(X_fake)
    D_output_f = D_output_f.squeeze()

    D_fake_loss = BCELoss(D_output_f, torch.zeros(D_output_f.size()).to(device))  # 和全是0的差距
    conSin_fake = F.cosine_similarity(Label, D_output_label_f).to(device)
    conSin_fake = torch.mean(F.tanh(conSin_fake + 1.0).to(device))
    D_fake_label_loss = BCELoss(conSin_fake, torch.zeros(conSin_fake.size()).to(device)).to(device)

    # 反向传播并优化 ############################
    D_loss = (D_real_loss + D_fake_loss) * 0.5 + (D_real_label_loss + D_fake_label_loss) * 0.5  # 判断器
    # D_loss.backward()
    # optimizer_D.step()

    return D_loss.data.item()


## 训练生成器 ##
def G_train(D: Discriminator, G: GeneratorUNet_ConTran, X, Label, BCELoss, L1, optimizer_G,
            lamb=100, lamb2=100, ):
    """
    训练生成器
    :param D: 判别器
    :param G: 生成器
    :param X: 未分隔的数据
    :param BCELoss: 二分交叉熵损失函数
    :param L1: L1正则化函数
    :param optimizer_G: 生成器优化器
    :param lamb: L1正则化的权重
    :return: 生成器的损失值
    """
    Label = Label.to(device)
    # 标签转实物（右转左）
    image_size = X.size(3) // 2
    x = X[:, :, :, image_size:].to(device)  # 标签图（右半部分）
    y = X[:, :, :, :image_size].to(device)  # 实物图（左半部分）

    # 梯度初始化为0
    G.zero_grad()
    # 生成图像
    G_output = G(x, Label)

    # 计算图像指标 #####################
    array_y = y.cpu().detach().numpy()
    array_G = G_output.cpu().detach().numpy()

    value_this_batch_MSE = MSE(array_y, array_G)
    value_this_batch_MAE = np.mean(np.abs(array_y - array_G))

    value_this_batch_SSIM = SSIM(array_y, array_G, data_range=1 - (-1), channel_axis=True)
    value_this_batch_PSNR = PSNR(array_y, array_G, data_range=1 - (-1))

    # 损失
    X_fake = torch.cat([x, G_output], dim=1)  # FAKE
    D_output_f, D_output_label_f = D(X_fake)
    D_output_f = D_output_f.squeeze()

    G_BCE_loss = BCELoss(D_output_f, torch.ones(D_output_f.size()).to(device))  # 最优为0 # D判断为1则证明无需返回损失
    G_L1_Loss = L1(G_output, y)

    conSin_real_G = F.cosine_similarity(Label, D_output_label_f).to(device)
    conSin_real_G = torch.mean(F.tanh(1 - conSin_real_G).to(device))  # 求和求平均即可
    G_coSin_BCE_loss = BCELoss(conSin_real_G, torch.ones(conSin_real_G.size()).to(device))  # 最优为0 # D判断为1则证明无需返回损失

    # 反向传播并优化
    G_loss = G_BCE_loss + lamb * G_L1_Loss + lamb2 * G_coSin_BCE_loss
    G_loss.backward()
    optimizer_G.step()

    return G_loss.data.item(), value_this_batch_MSE, value_this_batch_MAE, value_this_batch_SSIM, value_this_batch_PSNR


def G_val(D: Discriminator, G: GeneratorUNet_ConTran, X, Label, BCELoss, L1, optimizer_G,
          lamb=100, lamb2=100, ):
    """
    训练生成器
    :param D: 判别器
    :param G: 生成器
    :param X: 未分隔的数据
    :param BCELoss: 二分交叉熵损失函数
    :param L1: L1正则化函数
    :param optimizer_G: 生成器优化器
    :param lamb: L1正则化的权重
    :return: 生成器的损失值
    """
    Label = Label.to(device)
    # 标签转实物（右转左）
    image_size = X.size(3) // 2
    x = X[:, :, :, image_size:].to(device)  # 标签图（右半部分）
    y = X[:, :, :, :image_size].to(device)  # 实物图（左半部分）

    # 梯度初始化为0
    G.zero_grad()
    # 生成图像
    G_output = G(x, Label)

    # 计算图像指标 #####################
    array_y = y.cpu().detach().numpy()
    array_G = G_output.cpu().detach().numpy()

    value_this_batch_MSE = MSE(array_y, array_G)
    value_this_batch_MAE = np.mean(np.abs(array_y - array_G))

    value_this_batch_SSIM = SSIM(array_y, array_G, data_range=1 - (-1), channel_axis=True)
    value_this_batch_PSNR = PSNR(array_y, array_G, data_range=1 - (-1))

    # 损失
    X_fake = torch.cat([x, G_output], dim=1)  # FAKE
    D_output_f, D_output_label_f = D(X_fake)
    D_output_f = D_output_f.squeeze()

    G_BCE_loss = BCELoss(D_output_f, torch.ones(D_output_f.size()).to(device))  # 最优为0 # D判断为1则证明无需返回损失
    G_L1_Loss = L1(G_output, y)

    conSin_real_G = F.cosine_similarity(Label, D_output_label_f).to(device)
    conSin_real_G = torch.mean(F.tanh(1 - conSin_real_G).to(device))  # 求和求平均即可
    G_coSin_BCE_loss = BCELoss(conSin_real_G, torch.ones(conSin_real_G.size()).to(device))  # 最优为0 # D判断为1则证明无需返回损失

    # 反向传播并优化
    G_loss = G_BCE_loss + lamb * G_L1_Loss + lamb2 * G_coSin_BCE_loss
    # G_loss.backward()
    # optimizer_G.step()

    return G_loss.data.item(), value_this_batch_MSE, value_this_batch_MAE, value_this_batch_SSIM, value_this_batch_PSNR


## 主函数：训练Pix2Pix网络 ##
def main():
    # 过程数据储存
    stage = 'flat'  # change ##########
    project_name = '0924_SD-GAN-D_2D_epoch200_flat_rightLF' # change  ##########

    # 制作文件夹
    save_path = os.path.join(r'F:\u-GAN-SD-train-val-test', stage)
    if os.path.exists(save_path):
        pass
        # shutil.rmtree(save_path)
        # os.mkdir(save_path)
    else:
        os.mkdir(save_path)

    save_path = os.path.join(save_path, project_name)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        os.mkdir(save_path)
    else:
        os.mkdir(save_path)

    # log 文件
    log_file = os.path.join(save_path, 'log_' + project_name + '.txt')  # log 文件
    log_file = open(log_file, 'w')

    # 数据来源
    root = r'\dataset_flat'  # change
    subfolder_train = 'train'  # 'train'
    subfolder_val = 'val'  # 'val'
    batch_size = 32
    train_loader = loadData(root, subfolder_train, batch_size, shuffle=True)  # 16 * 25 = 400
    len_condition_numbers = 3  # 改为了3 个

    # 定义网络参数
    in_ch, out_ch = 3, 3  # 输入输出图片通道数
    ngf, ndf = 64, 64  # 生成数、判别器第一层卷积通道数 原始Pix2Pix
    image_size = 256  # 图片大小

    # 定义训练参数
    lr_G, lr_D = 0.0002, 0.0001  # G、D的学习速率  # 0.0002， # 0.0002 0.0004, 0.0002
    beta1 = 0.5  # momentum term of Adam（一般用的是0.9）
    lamb = 100  # 在生成器的目标函数中L1正则化的权重
    epochs = 200  # 训练迭代次数 300
    lamb2 = 10  # 在生成器的目标函数中conSin正则化的权重 100→10

    # 声明生成器、判别器 ##################
    G = GeneratorUNet_ConTran(in_ch, out_ch, image_size, len_condition_numbers).to(device)
    D = Discriminator(in_ch, out_ch, len_condition_numbers, ndf).to(device)

    # 目标函数 & 优化器
    BCELoss = nn.BCELoss().to(device)  # 优化器 # 都放在了GPU
    L1 = nn.L1Loss().to(device)  # Pix2Pix论文中在传统GAN目标函数加上了L1
    optimizer_G = optim.Adam(G.parameters(), lr=lr_G, betas=(beta1, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=lr_D, betas=(beta1, 0.999))

    ##############

    # 输入数据 & ground-truth & 初始生成器的输出
    X, Label = next(iter(train_loader), )
    g = G(X[:, :, :, image_size:].to(device), Label.to(device))
    save_image(X[:, :, :, image_size:], os.path.join(save_path, 'sample_0_input.png'))
    save_image(X[:, :, :, :image_size], os.path.join(save_path, 'sample_0_ground-truth.png'))
    save_image(g.view(batch_size, in_ch, image_size, image_size), os.path.join(save_path, 'sample_0_generate.png'))

    # 训练 ####################################
    G.train()  # （区分.eval）
    D.train()  # （ .train不启用BatchNorm、Dropout）

    D_Loss, G_Loss, Epochs = [], [], range(1, epochs + 1)  # 对一次epoch的loss数据操作
    D_Loss_val, G_Loss_val, Epochs_val = [], [], []  # 对一次# epoch的loss数据操作

    for epoch in (range(epochs)):

        log(log_file, f'Epoch: %d' % epoch)

        D_losses, G_losses, batch, d_l, g_l = [], [], 0, 0, 0  # 对一次batch的loss数据操作
        d_l_sum = 0
        g_l_sum = 0
        Quan_MSE, Quan_MAE, Quan_SSIM, Quan_PSNR = [], [], [], []

        for X, Label in tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}', ascii=True,
                             total=len(train_loader)):  # 一共 400/16 = 25 batch

            # 每次epoch最大为10
            batch += 1

            # 训练Discriminator并保存loss # 原来是5
            if batch // 2 == 0:  # 鉴别器少更新
                D_losses.append(D_train(D, G, X, Label, BCELoss, optimizer_D))

            # 训练Generator并保存损失
            loss_batch, mse_batch, mae_batch, ssim_batch, psnr_batch = G_train(D, G, X, Label, BCELoss, L1, optimizer_G,
                                                                               lamb, lamb2)

            G_losses.append(loss_batch)

            # 图像评价
            Quan_MSE.append(mse_batch)
            Quan_MAE.append(mae_batch)
            Quan_SSIM.append(ssim_batch)
            Quan_PSNR.append(psnr_batch)

            # 打印每次batch的平均loss
            d_l, g_l = np.array(D_losses).mean(), np.array(G_losses).mean()
            # 加和
            d_l_sum += d_l
            g_l_sum += g_l

            if batch == len(train_loader):  # 也可以减少缩进等一个epoch 内的batch循环完
                avg_d_l = d_l_sum / len(train_loader)
                avg_g_l = g_l_sum / len(train_loader)

                avg_mse = np.mean(Quan_MSE)
                avg_mae = np.mean(Quan_MAE)
                avg_ssim = np.mean(Quan_SSIM)
                avg_psnr = np.mean(Quan_PSNR)

        # 保存每次epoch的train_loss
        D_Loss.append(avg_d_l)
        G_Loss.append(avg_g_l)

        log(log_file, f'Train_D_Loss: % .5f' % avg_d_l)
        log(log_file, f'Train_G_Loss: % .5f' % avg_g_l)
        log(log_file, f'Train_MSE: % .5f' % avg_mse)
        log(log_file, f'Train_MAE: % .5f' % avg_mae)
        log(log_file, f'Train_SSIM: % .5f' % avg_ssim)
        log(log_file, f'Train_PSNR: % .5f' % avg_psnr)

        # val ###############################
        # 测试每5次epoch的生成效果
        if (epoch + 1) % 5 == 0:
            G.eval()
            D.eval()

            g_l_sum_val = 0
            d_l_sum_val = 0
            Quan_MSE_val, Quan_MAE_val, Quan_SSIM_val, Quan_PSNR_val = [], [], [], []

            val_loader = loadData(root, subfolder_val, batch_size, shuffle=True)  # 16 * 25 = 400
            D_losses, G_losses, batch, d_l, g_l = [], [], 0, 0, 0  # 对一次batch的loss数据操作

            for X, Label in val_loader:  # 一共 400/16 次 = 25 batch

                with torch.no_grad():

                    D_losses.append(D_val(D, G, X, Label, BCELoss, optimizer_D))  # D 评价

                    loss_batch_val, mse_batch_val, mae_batch_val, ssim_batch_val, psnr_batch_val \
                        = G_val(D, G, X, Label, BCELoss, L1, optimizer_G, lamb, lamb2)  # G 评价

                    G_losses.append(loss_batch_val)
                    # 图像评价
                    Quan_MSE_val.append(mse_batch_val)
                    Quan_MAE_val.append(mae_batch_val)
                    Quan_SSIM_val.append(ssim_batch_val)
                    Quan_PSNR_val.append(psnr_batch_val)

                    d_l_val, g_l_val = np.array(D_losses).mean(), np.array(G_losses).mean()
                    # 加和
                    d_l_sum_val += d_l_val
                    g_l_sum_val += g_l_val

            g_l_val_avg = g_l_sum_val / len(val_loader)
            d_l_val_avg = d_l_sum_val / len(val_loader)

            avg_mse_val = np.mean(Quan_MSE_val)
            avg_mae_val = np.mean(Quan_MAE_val)
            avg_ssim_val = np.mean(Quan_SSIM_val)
            avg_psnr_val = np.mean(Quan_PSNR_val)

            Epochs_val.append(epoch + 1)
            G_Loss_val.append(g_l_val_avg)
            D_Loss_val.append(d_l_val_avg)

            log(log_file, f'Val_D_Loss: % .3f' % d_l_val_avg)
            log(log_file, f'Val_G_Loss: % .3f' % g_l_val_avg)

            log(log_file, f'Val_MSE: % .5f' % avg_mse_val)
            log(log_file, f'Val_MAE: % .5f' % avg_mae_val)
            log(log_file, f'Val_SSIM: % .5f' % avg_ssim_val)
            log(log_file, f'Val_PSNR: % .5f' % avg_psnr_val)

            # print('Val [%d / %d]:  loss_d_avg_val= %.3f  loss_g_avg_val= %.3f' % (
            #     epoch + 1, epochs, d_l_val_avg, g_l_val_avg))

            # 使用val生成图片
            X, _ = next(iter(val_loader))  # 第一个batch
            save_image(X[:, :, :, image_size:], os.path.join(save_path, 'sample_' + str(epoch + 1) + '_input.png'))
            save_image(X[:, :, :, :image_size], os.path.join(save_path, 'sample_' + str(epoch + 1) + '_ground-truth.png'))
            g = G(X[:, :, :, image_size:].to(device), Label.to(device))
            save_image(g.view(batch_size, in_ch, image_size, image_size),
                       os.path.join(save_path, 'sample_' + str(epoch + 1) + '_generate.png'))

            G.train()
            D.train()

        # 保存训练结果
        if (epoch + 1) % 25 == 0:
            torch.save(G.state_dict(), os.path.join(save_path, 'params_generator' + '_%s.pth' % (int(epoch + 1),)))
            torch.save(D.state_dict(),
                       os.path.join(save_path, 'params_discriminator_params' + '_%s.pth' % (int(epoch + 1),)))
            '''
            G = torch.load('generator.pth')
            D = torch.load('discriminator.pth')
            '''
            print('Copy the parameters!')

    print("Done!")

    # 画出train-loss图

    plt.plot(Epochs, D_Loss, label='Discriminator Losses')
    plt.plot(Epochs, np.array(G_Loss) / 100, label='Generator Losses / 100')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss_train.png'))
    plt.show()

    # 画出val-loss图
    # G的val-loss因为包含L1 相比D的loss太大了 画图效果不好 所以除以100
    plt.plot(Epochs_val, D_Loss_val, label='Discriminator Losses')
    plt.plot(Epochs_val, np.array(G_Loss_val) / 100, label='Generator Losses / 100')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss_val.png'))
    plt.show()



if __name__ == '__main__':
    # # 运行 ##
    main()

