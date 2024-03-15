import torch.nn as nn
import torch.nn.functional as F
import torch

import os
import sys

from Transformer_for_SD_UNet import SpatialTransformer

from tensorboardX import SummaryWriter


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)  # 增加尺寸
        x = torch.cat((x, skip_input), 1)

        return x


class UNetMiddle(nn.Module):
    def __init__(self, in_channel, out_channel, normalize=True, dropout=0.0):
        super(UNetMiddle, self).__init__()
        layers = [nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channel))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp4(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp4, self).__init__()

    def forward(self, x, skip_input):
        x = torch.cat((x, skip_input), 1)

        return x


class EmbeddingNum(nn.Module):
    def __init__(self, length_vector, size_feature):
        super(EmbeddingNum, self).__init__()

        self.size_ = size_feature

        layers = [
            nn.BatchNorm1d(length_vector),

            nn.Linear(length_vector, 8 * 8),
            nn.ReLU(True),

            nn.Linear(8 * 8, 16 ** 2),
            nn.ReLU(True),

            nn.Linear(16 ** 2, size_feature ** 2),
            nn.ReLU(True),

        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, self.size_, self.size_)
        return x


class GeneratorUNet_ConTran(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, pic_sizes=256, length_num=16):
        super(GeneratorUNet_ConTran, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

        # self.middle = UNetMiddle(512, 512)
        # self.up4 = UNetUp4(512, 512)

        self.feature_size = pic_sizes // (2 ** 8)
        self.con = EmbeddingNum(length_num, self.feature_size)  # length 16

    def forward(self, x, c):
        """
        x 图片 (batch, channel, h, w)
        c 条件 (batch, n) 以为向量
        """

        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        attention_tr = SpatialTransformer(512, 3, 3, self.feature_size).to(device)  # 图片的通道数 head layer
        middle = attention_tr(d8, self.con(c))

        # u4 = self.up4(middle, d4)

        # #################
        # u1 = self.up1(d8, d7)
        u1 = self.up1(middle, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


##############################
#        Discriminator
##############################


## 辨别器 PatchGAN（其实就是卷积网络而已） ##
class Discriminator(nn.Module):
    def __init__(self, in_ch, out_ch, length_label, ndf=64):
        """
        定义判别器的网络结构
        :param in_ch: 输入数据的通道数
        :param ndf: 第一层卷积的通道数 number of discriminator's first conv filters
        """
        super(Discriminator, self).__init__()

        # 不是输出一个表示真假概率的实数，而是一个N*N的Patch矩阵（此处为30*30），其中每一块对应输入数据的一小块
        # in_ch + out_ch 是为将对应真假数据同时输入
        # 256 * 256（输入）
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_ch + out_ch, ndf, kernel_size=4, stride=2, padding=1),
            # 输入图片已正则化 不需BatchNorm
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 128 * 128
        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 64 * 64
        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 32 * 32
        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 31 * 31
        self.layer5 = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )
        # 30 * 30（输出的Patch大小） # 局部细节

        # ########### D_vector
        self.D_vec = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),  # C 16 * 16
                                   nn.BatchNorm2d(ndf * 8),
                                   nn.LeakyReLU(0.2, inplace=True),

                                   nn.Conv2d(ndf * 8, 1, kernel_size=1, stride=1, padding=0),
                                   nn.BatchNorm2d(1),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   ) # 1 16 * 16

        self.outVec = nn.Linear(16**2, length_label)

    def forward(self, X):
        """
        判别器模块正向传播
        :param X: 输入判别器的数据
        :return: 判别器的输出
        """
        layer1_out = self.layer1(X)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)
        layer5_out = self.layer5(layer4_out)

        # D-Vec
        Vec_con = self.D_vec(layer3_out)  # 还是保持批次的维度
        Vec_con = Vec_con.view(-1, 16**2)
        Vec_out = self.outVec(Vec_con)

        return layer5_out, Vec_out


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    attention_1 = GeneratorUNet_ConTran(3, 3, 256, 16).to(device)

    img_input_1 = torch.randn(32, 3, 256, 256).to(device)
    num_input_1 = torch.randn(32, 16).to(device)

    m = attention_1(img_input_1, num_input_1)


    print(m.size())


