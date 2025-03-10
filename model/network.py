import torch
from torch import nn
from torch.nn import functional as F
from .blazepose import BlazePose, ClassHead
import matplotlib.pyplot as plt

class EEG(nn.Module):
    def __init__(self, config):
        super(EEG, self).__init__()

        self.n_joints = config.NUM_JOINTS

        self.blaze_pose = BlazePose(config)

        self.enable_eros = config.EROS
        inp_chn = config.MODEL.INPUT_CHANNEL
        width, height = config.MODEL.IMAGE_SIZE
        self.hm_width, self.hm_height = config.MODEL.HEATMAP_SIZE

        kernal_size = config.DATASET.EROS.KERNEL_SIZE
        decay_base = config.DATASET.EROS.DECAY_BASE

        self.classification_head = ClassHead(32, 2, config,activation='leaky_relu')
        
        self.temporal_shift = TemporalShift(fold_div=3)


    def forward(self, x, prev_buffer=None, prev_key=None, batch_first=False):
        T, B, C, H, W = x.shape  # 输入形状为 [T, B, C, H, W]
        N = 8  # 时间段数
        T_per_seg = T // N  # 每段时间内的帧数
        
        # 1. 将输入从 [T, B, C, H, W] 转换为 [B, T, C, H, W]
        x = x.permute(1, 0, 2, 3, 4).contiguous()  # 形状变为 [B, T, C, H, W]
        x_segments = x.chunk(N, dim=1)

        features = []
        for seg in x_segments:
        # 将每段时间的特征展平为 [B * T_per_seg, C, H, W]
            seg = seg.reshape(B * T_per_seg, C, H, W)
            # 通过 BlazePose 提取特征，输出形状为 [B * T_per_seg, F]，其中 F=2048
            feat = self.blaze_pose(seg)  # 输出形状为 [B * T_per_seg, 2048]
            # 将特征重新调整为 [B, T_per_seg, F]
            feat = feat.view(B, T_per_seg, -1)
            features.append(feat)# 将特征在时间维度上堆叠
        features_stacked = torch.stack(features, dim=2)  # [B, T_per_seg, N, F, H', W']
        # print("features_stacked.size():")
        # print(features_stacked.size())
        # 应用TSM模块
        features_tsm = self.temporal_shift(features_stacked)  # [B, T_per_seg, N, F, H', W']
        # 特征聚合，例如取平均
        features_agg = features_tsm.mean(dim=2)  # [B, T_per_seg, F, H', W']
         # 将时间维度和批量维度合并
        features_agg = features_agg.view(B * T_per_seg, -1)# 送入分类头进行分类
        x = self.classification_head(features_agg)
        return x

class TemporalShift(nn.Module):
    def __init__(self, fold_div=4):
        super(TemporalShift, self).__init__()
        self.fold_div = fold_div

    def forward(self, x):
        # x shape: [B, T_per_seg, N, F, H', W']
        B, T, N, F = x.size()
        fold = F // self.fold_div

        x1 = x[:, :, :, :fold]  # 第一部分：左移
        x2 = x[:, :, :, fold:2*fold]  # 第二部分：右移
        x3 = x[:, :, :, 2*fold:2*fold + fold]  # 第三部分：保持不变
        x_remaining = x[:, :, :, 2*fold + fold:]  # 剩余通道s
        # 对x1向后平移一帧，对x2向前平移一帧
        shift_left = torch.roll(x1, shifts=1, dims=1)  # 左移
        shift_right = torch.roll(x2, shifts=-1, dims=1)  # 右移
        
        # 组合回特征
        out = torch.cat([shift_left, shift_right, x3, x_remaining], dim=3)
        return out



class TemporalShift_Global(nn.Module):
    def __init__(self, fold_div=4):
        super().__init__()
        self.fold_div = fold_div

    def forward(self, x):
        # x shape: [B, T, F]
        B, T, F = x.size()
        fold = F // self.fold_div
        
        x1 = x[:, :, :fold]
        x2 = x[:, :, fold:2*fold]
        x3 = x[:, :, 2*fold:3*fold]
        x_remaining = x[:, :, 3*fold:]

        shift_left = torch.roll(x1, shifts=1, dims=1)
        shift_right = torch.roll(x2, shifts=-1, dims=1)

        return torch.cat([shift_left, shift_right, x3, x_remaining], dim=2)
    
class EEG(nn.Module):
    """Option2: 先局部后跨段时间移位，使用首段特征"""
    def __init__(self, config):
        super().__init__()
        self.n_segments = 8
        self.blaze_pose = BlazePose(config)
        self.tsm_local = TemporalShift(fold_div=3)
        self.tsm_inter = TemporalShift_InterBins(fold_div=3)
        self.classification_head = ClassHead_Option2(
            inp_dim=2048, 
            hidden_dim=128, 
            num_classes=config.NUM_CLASSES
        )

    def forward(self, x, prev_buffer=None, prev_key=None, batch_first=False):
        # 输入形状: [T=48, B=4, C=2, H=720, W=1280]
        T, B, C, H, W = x.shape
        T_per_seg = T // self.n_segments  # 6
        
        # 重塑输入为连续内存（关键修复）
        x = x.permute(1, 0, 2, 3, 4).contiguous()  # [B, T, C, H, W]
        x = x.reshape(B, self.n_segments, T_per_seg, C, H, W)  # [4,8,6,2,720,1280]
        
        all_features = []
        for seg_idx in range(self.n_segments):
            # --- 关键修复：确保张量连续 ---
            seg = x[:, seg_idx].contiguous()  # [4,6,2,720,1280]
            seg_flat = seg.view(B * T_per_seg, C, H, W)  # [24,2,720,1280]
            
            feat = self.blaze_pose(seg_flat)  # [24,2048]
            feat = feat.view(B, T_per_seg, -1)  # [4,6,2048]
            all_features.append(feat)
        
        features = torch.stack(all_features, dim=1)  # [4,8,6,2048]
        
        # 时间移位
        features = self.tsm_local(features)  # 局部
        features = self.tsm_inter(features)  # 跨段
        
        # 取首段特征
        first_seg = features[:, 0]  # [4,6,2048]
        features_agg = first_seg.mean(dim=1)  # [4,2048]
        
        return self.classification_head(features_agg)  # [4,38]

class TemporalShift_InterBins(nn.Module):
    def __init__(self, fold_div=4):
        super().__init__()
        self.fold_div = fold_div

    def forward(self, x):
        # 输入形状: [B, N_segments, T_per_seg, F]
        B, N, T, F = x.size()
        x = x.permute(0, 2, 1, 3)  # [B, T_per_seg, N, F]
        x = x.reshape(B * T, N, F)  # 合并时间维度
        
        fold = F // self.fold_div
        x1 = x[:, :, :fold]
        x2 = x[:, :, fold:2*fold]
        x3 = x[:, :, 2*fold:3*fold]
        x_remaining = x[:, :, 3*fold:]

        # 跨段移位
        shift_left = torch.roll(x1, shifts=1, dims=1)  # 跨段左移
        shift_right = torch.roll(x2, shifts=-1, dims=1)  # 跨段右移

        out = torch.cat([shift_left, shift_right, x3, x_remaining], dim=2)
        out = out.view(B, T, N, F).permute(0, 2, 1, 3)  # 恢复原始维度
        return out  # 输出形状: [B, N, T_per_seg, F]
       

class ClassHead_Option2(nn.Module):
    """分类头（Option2专用）"""
    def __init__(self, inp_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(inp_dim, hidden_dim)  # 2048 → 128
        self.fc2 = nn.Linear(hidden_dim, num_classes)  # 128 → 38
        self.act = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        # 输入形状: [B,2048]
        x = self.act(self.fc1(x))  # [4,128]
        return self.fc2(x)  # [4,38]


