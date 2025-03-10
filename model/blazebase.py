import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from settings import config as cfg
class ClassHead(nn.Module):
    def __init__(self, inp_chns, out_chns, config, activation='None'):
        super().__init__()

        if activation == 'None':
            self.activation = nn.Identity()
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)

        
        self.n_classes = config.NUM_CLASSES
        self.n_max = config.N_MAX

        self.input_dim = 2048  # 输入特征张量的维度
        self.hidden_dim = 128  # 中间层的维度
        self.num_classes = self.n_classes  # 输出类别的数量

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.num_classes)
        self.dropout = nn.Dropout(0.2)

        # self.pool = nn.AdaptiveAvgPool2d((36, 64))

    def forward(self, x): #torch.Size([2, 2304, 180, 320])
        # x 的形状: [B * T_per_seg, C, H, W]
        B_T, F = x.shape
        batch=cfg.BATCH_SIZE       

        # 全连接层
        x = self.fc1(x)  # [B * T_per_seg, hidden_dim]
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)  # [B * T_per_seg, num_classes]

        # 将 T_per_seg 维度压缩到一起
        x = x.view(batch, -1, self.num_classes)  # [B, T_per_seg, num_classes]
        x = x.mean(dim=1)  # [B, num_classes]  # 在 T_per_seg 维度上取平均

        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, inp_chns, out_chns, kernel_size, stride=1) -> None:
        super().__init__()
    
        self.conv_1 =  nn.Conv2d(inp_chns, inp_chns, kernel_size, stride=stride, groups=inp_chns, padding=kernel_size//2, bias=False)
        self.conv_2 = nn.Conv2d(inp_chns, out_chns, 1, bias=False)

    def forward(self, x):
        return self.conv_2(self.conv_1(x))    


class SqueezeChannels(nn.Module):
    def __init__(self, inp_chns, out_chns) -> None:
        super().__init__()

        if inp_chns != out_chns:
            self.conv = nn.Conv2d(inp_chns, out_chns, 1, bias=False)
        else:
            self.conv = nn.Identity()

    def forward(self, x):
        return self.conv(x)


class DecoderConv(nn.Module):
    def __init__(self, inp_chns, out_chns, block_num, sampler=None) -> None:
        super().__init__()

        self.sampler = sampler
# DepthwiseSeparableConv 层的逐点卷积部分负责将输入通道数调整为输出通道数
        if self.sampler == 'down':
            self.conv_b = DepthwiseSeparableConv(inp_chns, out_chns, 3, stride=2)
        else:
            self.conv_b = DepthwiseSeparableConv(inp_chns, out_chns, 3)
        
        self.conv = nn.ModuleList()
        for i in range(block_num):
            self.conv.append(DepthwiseSeparableConv(out_chns, out_chns, kernel_size=3))

    def forward(self, x):
        x = F.relu(self.conv_b(x))

        for i in range(len(self.conv)):
            x = F.relu(x + self.conv[i](x))

        if self.sampler == 'up':
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        
        return x


class ChannelPadding(nn.Module):
    def __init__(self, channels):
        super(ChannelPadding, self).__init__()
        self.channels = channels

    def forward(self, x):
        pad_shape = (0, 0, 0, 0, 0, self.channels - x.size(1))
        out = nn.functional.pad(x, pad_shape, 'constant', 0)

        return out

class BlazeBlock(nn.Module):
    def __init__(self, inp_channel, out_channel, block_num=3):
        super(BlazeBlock, self).__init__()

        self.downsample_a = DepthwiseSeparableConv(inp_channel, out_channel, 3, stride=2)   

        if inp_channel != out_channel:
            self.downsample_b = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                ChannelPadding(channels=out_channel)
            )
        else:
            self.downsample_b = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv = nn.ModuleList()
        for i in range(block_num):
            self.conv.append(DepthwiseSeparableConv(out_channel, out_channel, kernel_size=3))
    def forward(self, x):
        da = self.downsample_a(x)
        db = self.downsample_b(x)
        if da.size(2) != db.size(2) or da.size(3) != db.size(3):
    # 计算高度和宽度的填充量
            padding_height = (0, 0)  # 初始化高度方向不填充
            padding_width = (0, 0)   # 初始化宽度方向不填充

            # 根据高度差调整填充
            if da.size(2) > db.size(2):
                padding_height = (0, da.size(2) - db.size(2))  # 在db的底部填充
            elif da.size(2) < db.size(2):
                padding_height = (db.size(2) - da.size(2), 0)  # 在da的底部填充

            # 根据宽度差调整填充
            if da.size(3) > db.size(3):
                padding_width = (0, da.size(3) - db.size(3))  # 在db的右侧填充
            elif da.size(3) < db.size(3):
                padding_width = (db.size(3) - da.size(3), 0)  # 在da的右侧填充
            # print("Padding db:", padding_height, padding_width)

    # 应用填充
            if padding_height != (0, 0) or padding_width != (0, 0):
                db = F.pad(db, padding_width + padding_height, 'constant', 0)

            # print("db size after padding:", db.size())
            # 应用填充
            # db = F.pad(db, padding_height + padding_width, 'constant', 0)
        if da.size() != db.size():
            print(1)
            raise ValueError(f"通道数不匹配: da={da.size()}, db={db.size()}")
        x = F.relu(da + db)

        for i in range(len(self.conv)):
            x = F.relu(x + self.conv[i](x))
        return x
  
