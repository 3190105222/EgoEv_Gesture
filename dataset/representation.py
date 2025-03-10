import cv2
import numpy as np
import torch

from typing import Any
import matplotlib.pyplot as plt

# 用于图像尺寸变换
class ResizeTransform:
    def __init__(self, cfg, height, width):
        self.source_height = height
        self.source_width = width

        self.taget_width = cfg.MODEL.IMAGE_SIZE[0]
        self.taget_height = cfg.MODEL.IMAGE_SIZE[1]

        self.sx = (self.taget_width / self.source_width)
        self.sy = (self.taget_height / self.source_height)

    def __call__(self, x, y) -> Any:
        x = x.astype(np.float32)
        y = y.astype(np.float32)

        x = x * self.sx
        y = y * self.sy

        return x, y

    @property
    def height(self):
        return self.taget_height
    
    @property
    def width(self):
        return self.taget_width


class LNES:  #同一个动作的正负是挨着的
    def __init__(self, cfg, height, width):
        self.resize_transform = ResizeTransform(cfg, height, width)
        
        lnes_config = cfg.DATASET.LNES
        self.windows_time_ms = lnes_config.WINDOWS_TIME_MS

    def __call__(self, data_batch) -> Any:
        windows_time_ms = self.windows_time_ms
        
        if data_batch.shape[-1] == 4:
            original_xs, original_ys, original_ps, original_ts = data_batch.T
        else:
            raise ValueError('Invalid data_batch shape')

        original_ts = original_ts.astype(np.float32)
        # ts = (ts[-1] - ts) * 1e-3 # microseconds to milliseconds 

                # 新增代码开始
        num_windows = int(np.ceil((original_ts.max() - original_ts[0]) / self.windows_time_ms))
        lnes_list = []
        xs_list = []
        ys_list = []

        for i in range(num_windows):
            start_time = original_ts[0] + i * windows_time_ms
            end_time = start_time + windows_time_ms
            selected_indices = (original_ts >= start_time) & (original_ts < end_time)
            
            xs = original_xs[selected_indices]
            ys = original_ys[selected_indices]
            ts = original_ts[selected_indices]
            ps = original_ps[selected_indices].astype(np.int32)

            xs, ys = self.resize_transform(xs, ys)
            width, height = self.resize_transform.width, self.resize_transform.height
            xs = xs.astype(np.int32)
            ys = ys.astype(np.int32)

            # 重新计算时间比例
            ts = (ts - start_time) / windows_time_ms
            # 重新生成lnes表示
            lnes = np.zeros((height, width, 2))
            lnes[ys, xs, ps] = 1.0 - ts  #(720, 1280, 2)
            lnes_list.append(lnes)

        return lnes_list
        


    def visualize(cls, lnes: np.ndarray):
        if isinstance(lnes, torch.Tensor):
            lnes = lnes.permute(1, 2, 0)
            lnes = lnes.cpu().numpy()   

        lnes = lnes.copy() * 255
        lnes = lnes.astype(np.uint8)
                
        h, w = lnes.shape[:2]
    
        b = lnes[..., :1]
        r = lnes[..., 1:]
        g = np.zeros((h, w, 1), dtype=np.uint8)

        lnes = np.concatenate([r, g, b], axis=2).astype(np.uint8)

        return lnes

