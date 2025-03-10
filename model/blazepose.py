import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from .blazebase import BlazeBlock, DecoderConv, ClassHead
import os

from mambapy.mamba import Mamba, MambaConfig

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

    def forward(self, x):
        # avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        a=self.avg_pool(x)
        b=self.fc1(a)
        c=self.relu1(b)
        d=self.fc2(c)
        avg_out=d
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return F.sigmoid(out) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_attn = torch.cat([avg_out, max_out], dim=1)
        x_attn = self.conv1(x_attn)
        return self.sigmoid(x_attn) * x

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class BlazePose(nn.Module):
    def __init__(self, config):
        super(BlazePose, self).__init__()
        

        self.n_joints = config.NUM_JOINTS
        self.inp_chn = config.MODEL.INPUT_CHANNEL    
        self.cbam = CBAM(128)       
# 添加 Mamba 模块
        self.mamba_config = MambaConfig(d_model=192, n_layers=2)
        self.mamba = Mamba(self.mamba_config)
        # 添加全连接层，将金字塔池化输出映射到 2048 维
        self.fc = nn.Linear(128 * (1 + 4 + 16), 2048)  # 192 是通道数，1+4+16 是金字塔池化的输出维度
        self.conv3 = nn.Sequential(
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
    )
        # 添加金字塔池化模块
        self.spp = SpatialPyramidPooling(levels=[1, 2, 4])  # 实例化 SPP 模块
        self._define_layers()        


    def _define_layers(self):     
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.inp_chn, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.backbone1 = nn.ModuleList([
            BlazeBlock(16, 32, 3),
            BlazeBlock(32, 64, 4),
            BlazeBlock(64, 128, 5),
            BlazeBlock(128, 192, 6),
        ])
        self.segmentation_decoder = nn.ModuleList([
            DecoderConv(192 , 192 , 2),
            DecoderConv(192  *2, 128 , 2, sampler='up'),
            DecoderConv(128  *2, 64 , 2, sampler='up'),
            DecoderConv(64  *2, 32 , 2, sampler='up'),
        ])

    def forward(self, x):
        device = torch.device(os.environ["DEVICE"] if "DEVICE" in os.environ else "cpu")
        
        H, W = x.shape[-2:]  #torch.Size([B, 2, 720, 1280])
        B = x.shape[0]
        x = self.conv1(x) #torch.Size([4, 16, 360, 640])
        feature_maps = []
        for i, layer in enumerate(self.backbone1):
            x = layer(x)  #torch.Size([2, 768, 360, 640]) torch.Size([2, 1152, 180, 320])
            # torch.Size([2, 1536, 90, 160])# torch.Size([2, 2304, 45, 80])torch.Size([2, 3072, 23, 40])
            feature_maps.append(x)

        f5 = feature_maps[-1] #torch.Size([4, 192, 23, 40]) torch.Size([2, 3072, 23, 40])
        feature_maps = feature_maps[::-1]
        # 在backbone1和decoder之间加入CBAM
        # f5 = self.cbam(f5)
        B, C, H, W = f5.shape
        f5 = f5.reshape(B, C, -1)  # 形状变为 (B, C, H * W)
        f5 = f5.permute(0, 2, 1)   # 形状变为 (B, H * W, C)
        f5 = self.mamba(f5)        # 输入 Mamba 模型
        f5 = f5.permute(0, 2, 1)   # 形状恢复为 (B, C, H * W)
        f5 = f5.reshape(B, C, H, W)  # 形状恢复为 (B, C, H, W)
        # x = f5
        x = f5 #torch.Size([4, 192, 40, 23])
        for i, seg in enumerate(self.segmentation_decoder):
            x_seg = seg(x)              #torch.Size([2, 192, 23, 40]) torch.Size([2, 9216, 23, 40])
            # torch.Size([2, 2304, 46, 80]) torch.Size([2, 3072, 90, 160])
            if x_seg.size()[2:]!=feature_maps[i].size()[2:]:
                x_seg = F.interpolate(x_seg, size=feature_maps[i].shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x_seg, feature_maps[i]], dim=1) #torch.Size([2, 192, 23, 40])
        x = self.conv2(x)# [b 4,160 90]
        x = self.conv3(x)
        # x = self.cbam(x)
        # # 添加金字塔池化
        x_spp = self.spp(x)  # 输出形状为 [B, C, (1 + 4 + 16) * 192]
        x_spp = x_spp.view(B, -1)  # 展平为 [B, C * (1 + 4 + 16)]
        x_spp = self.fc(x_spp)  # 映射到 [B, 2048]

        return x_spp  # 输出形状为 [B, 2048]

# 定义金字塔池化模块
class SpatialPyramidPooling(nn.Module):
    def __init__(self, levels=[1, 2, 4]):
        super(SpatialPyramidPooling, self).__init__()
        self.levels = levels

    def forward(self, x):
        B, C, H, W = x.shape
        pooled_features = []
        for level in self.levels:
            # 计算每个金字塔层的输出大小
            pool_size = (H // level, W // level)
            if pool_size[0] == 0 or pool_size[1] == 0:
                raise ValueError(f"Pool size too small for input size {H}x{W} at level {level}")
            # 使用固定池化尺寸
            pooled = F.avg_pool2d(x, kernel_size=pool_size, stride=pool_size)
            pooled = pooled.view(B, C, -1)  # 展平
            pooled_features.append(pooled)
        # 将所有池化结果拼接在一起
        return torch.cat(pooled_features, dim=2)
