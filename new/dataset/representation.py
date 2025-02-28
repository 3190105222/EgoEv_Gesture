import cv2
import numpy as np
import torch

from typing import Any
import matplotlib.pyplot as plt

def add_salt_and_pepper(image, low_th, high_th):
    saltpepper_noise = np.zeros_like(image)
    cv2.randu(saltpepper_noise, 0, 255)

    image[saltpepper_noise < low_th] = 0
    image[saltpepper_noise > high_th] = 255


class Rectangle:
    def __init__(self, x_tl, y_tl, width, height):
        self.x_tl = x_tl
        self.y_tl = y_tl
        self.width = width
        self.height = height

    def intersect(self, rect):
        x_tl = max(self.x_tl, rect.x_tl)
        y_tl = max(self.y_tl, rect.y_tl)
        x_br = min(self.x_tl + self.width, rect.x_tl + rect.width)
        y_br = min(self.y_tl + self.height, rect.y_tl + rect.height)
        if x_tl < x_br and y_tl < y_br:
            return Rectangle(x_tl, y_tl, x_br - x_tl, y_br - y_tl)
        return None

    __and__ = intersect

    def equal(self, rect):
        x_tl_diff = self.x_tl - rect.x_tl
        y_tl_diff = self.y_tl - rect.y_tl
        x_br_diff = self.width - rect.width
        y_br_diff = self.height - rect.height
        diff = x_tl_diff + y_tl_diff + x_br_diff + y_br_diff
        if diff == 0:
            return True
        return False

    __eq__ = equal

'''```
EROS 类：

表示一个侵蚀操作，用于图像处理中的形态学操作。
属性：
kernel_size：结构元素的大小。
frame_height 和 frame_width：图像的尺寸。
decay_base：衰减基础值。
方法：
get_frame：获取当前的侵蚀图像。
update：根据给定的速度（vx, vy）更新侵蚀图像。
reset_frame：重置侵蚀图像。
'''
class EROS:

    def __init__(self, kernel_size, height, width, decay_base=0.3):
        self.kernel_size = kernel_size
        self.frame_height = height
        self.frame_width = width
        self.decay_base = decay_base
        self._image = np.zeros((height, width), dtype=np.uint8)

        if self.kernel_size % 2 != 0:
            self.kernel_size += 1

    def get_frame(self):
        return self._image

    def update(self, vx, vy):
        odecay = self.decay_base ** (1.0 / self.kernel_size)
        half_kernel = int(self.kernel_size / 2)
        roi_full = Rectangle(0, 0, self.frame_width, self.frame_height)
        roi_raw = Rectangle(0, 0, self.kernel_size, self.kernel_size)

        roi_raw.x_tl = vx - half_kernel
        roi_raw.y_tl = vy - half_kernel
        roi_valid = roi_raw & roi_full

        if roi_valid is None:
            return True
        roi = [roi_valid.y_tl, roi_valid.y_tl + roi_valid.height, roi_valid.x_tl, roi_valid.x_tl + roi_valid.width]
        update_mask = np.ones((roi[1] - roi[0], roi[3] - roi[2]), dtype=np.float) * odecay
        self._image[roi[0]:roi[1], roi[2]:roi[3]] = np.multiply(self._image[roi[0]:roi[1], roi[2]:roi[3]],
                                                                update_mask).astype(np.uint8)
        self._image[vy, vx] = 255

        return roi_raw != roi_valid
    def reset_frame(self):
        pass

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
            # xs_list.append(xs)
            # ys_list.append(ys)
        # 在第0个维度上拼接所有lnes表示
        # concatenated_lnes = np.concatenate(lnes_list, axis=0)
        # concatenated_xs = np.concatenate(xs_list, axis=0)
        # concatenated_ys = np.concatenate(ys_list, axis=0)
        # data = {
        #     'input': lnes_list,
        # #     'coord_x': xs_list,
        # #     'coord_y': ys_list,
        # }
        
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


class EventFrame:
    def __init__(self, cfg, height, width):
        self.height = height
        self.width = width

        self.resize_transform = ResizeTransform(cfg, height, width)

    def __call__(self, data_batch) -> Any:
        if data_batch.shape[-1] == 6:
            xs, ys, ts, ps, fs, segmentation = data_batch.T
        elif data_batch.shape[-1] == 5:
            xs, ys, ts, ps, fs = data_batch.T
        elif data_batch.shape[-1] == 4:
            xs, ys, ts, ps = data_batch.T
        else:
            raise ValueError('Invalid data_batch shape')

        ts = ts.astype(np.float32)
        ts = (ts[-1] - ts) * 1e-3 # microseconds to milliseconds 

        xs, ys = self.resize_transform(xs, ys)
        width, height = self.resize_transform.width, self.resize_transform.height

        xs = xs.astype(np.int32)
        ys = ys.astype(np.int32)
        ps = ps.astype(np.int32)

        ef = np.zeros((height, width, 2))
        ef[ys, xs, ps] = 1.0

        data = {
            'input': ef,
            'coord_x': xs,
            'coord_y': ys,
        }

        if data_batch.shape[-1] >= 5:
            data['frame_index'] = fs[-1]            
        
        if data_batch.shape[-1] == 6:
            data['segmentation_indices'] = segmentation.astype(np.uint8)

        return data

    def visualize(cls, ef: np.ndarray):
        if isinstance(ef, torch.Tensor):
            ef = ef.permute(1, 2, 0)
            ef = ef.cpu().numpy()   

        ef = ef.copy() * 255
        ef = ef.astype(np.uint8)
                
        h, w = ef.shape[:2]
    
        b = ef[..., :1]
        r = ef[..., 1:]
        g = np.zeros((h, w, 1), dtype=np.uint8)

        ef = np.concatenate([r, g, b], axis=2).astype(np.uint8)

        return ef
