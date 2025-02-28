import sys; sys.path.append('../')
# import cv2
# import random
# import json
import logging
# import pickle
import numpy as np
# import os
# import torch
# from dataset.dataset_utils import h5py_File
from dataset.transforms import rotate_points

from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)


class RealEventStream(Dataset):
    # 从JSON文件中加载元数据，
    # 获取图像的高度和宽度。同时，加载事件流文件、本地姿态文件和全局姿态文件
    def __init__(self, data_path, cfg, is_train):
        super().__init__()

        self.data_path = data_path
        self.height = 720#meta['height']
        self.width = 1280#meta['width']
        
        self.is_train = is_train
        # self.stream_path = self.data_path / 'events.h5'
        self.fin = None 

        self.max_frame_time = cfg.DATASET.REAL.MAX_FRAME_TIME_IN_MS
        self.is_train = is_train
    
    def generate_class(self, npy_filepath):
        # npy_path = Path(npy_filepath)
        npy_filename = Path(npy_filepath).stem
        x, y = map(int, npy_filename.replace('act', '').split('_')[:2])
        if x >37:
            x = x - 38
        return x





    def generate_txt_filename(self, npy_filepath):
        # 从.npy文件名中提取x和y
        npy_path = Path(npy_filepath)
        npy_filename = Path(npy_filepath).stem
        directory_path = npy_path.parent
        x, y = map(int, npy_filename.replace('act', '').split('_')[:2])
        # 根据x的值生成a, b, c
        if x < 38:
            a = x + 1
            b = 0
        else:
            a = x - 37
            b = 1
        c = y+1
        # 格式化为a-b-c.txt
        txt_filename = f"{a:02d}-{b}-{c}.txt"
        return directory_path / 'label' /txt_filename 
    
    def generate_seg_filename(self, npy_filepath):
        # 从.npy文件名中提取x和y
        npy_path = Path(npy_filepath)
        npy_filename = Path(npy_filepath).stem
        directory_path = npy_path.parent
        x, y = map(int, npy_filename.replace('act', '').split('_')[:2])
        # 根据x的值生成a, b, c
        if x < 38:
            a = x + 1
            b = 0
        else:
            a = x - 37
            b = 1
        c = y+1
        # 格式化为a-b-c.txt
        seg_filename = f"{a:02d}-{b}-{c}.npy"
        return directory_path / 'mask' / seg_filename  
    
    def __len__(self):
        data = np.load(self.data_path)
        return len(data)

    def __getitem__(self, idx):
        
        # data_path = self.data_paths[idx]  # 使用索引idx来获取正确的数据文件路径
        # txt_file = self.generate_txt_filename(self.data_path)
        # seg_file = self.generate_seg_filename(self.data_path)
        # txt_file = os.path.join(data_path, txt_filename)

        # Load the numpy file
        data = np.load(self.data_path)
        data[:,3] = (data[:,3] - data[0,3])/1000
        class1 = self.generate_class(self.data_path)
        # Load the corresponding label from the text file
        # with open(txt_file, 'r') as f:
        #     label = f.read().strip()

        # seg_gt = np.load(seg_file) #(5, 720, 1280)

         # 创建一个1280x720的黑色图片
        # image = Image.new('L', (1280, 720), 0)
                # 将字符串数据分割成行
        # lines = label.strip().split('\n')

        # 将图片转换为像素数组，方便操作
        # pixels = image.load()
        # for i in range(22, 42):
        #     # 分割每行数据
        #     parts = lines[i].split()
        #     # 确保每行都有至少3个部分
        #     if len(parts) >= 3:
        #         x = int(parts[1])  # 获取x坐标
        #         y = int(parts[2])  # 获取y坐标

        #     # 确保坐标在图片范围内
        #     if 0 <= x < 1280 and 0 <= y < 720:
        #         # 将坐标周围的4x4区域设置为255
        #         for dx in range(-1, 2):
        #             for dy in range(-1, 2):
        #                 pixels[x + dx, y + dy] = 255


        # seg_gt  =0  
        return data,class1# label,seg_gt
    
'''
# 用于获取指定索引的事件批量。根据是否为训练模式，随机或使用固定的最大帧时间。
#然后，从事件流中获取事件数据，并将其转换为 NumPy 数组
    def get_event_batch(self, idx, num_events):
        if self.is_train:
            max_frame_time = random.randint(15, self.max_frame_time)
        else:
            max_frame_time = self.max_frame_time
        
        frame_time = 0
        data_batches = []
        while frame_time < max_frame_time:
            data_batch = self.fin[idx: idx + num_events] 
            ts = data_batch[:, 2] 
            
            if not len(ts): 
                break

            ts = (ts[-1] - ts[0]) * 1e-3 # microseconds to milliseconds 

            data_batches.append(data_batch)

            frame_time += ts
            idx += num_events

        if len(data_batches) == 0:
            raise StopIteration
        
        data_batches_np = np.concatenate(data_batches, axis=0)
        del data_batches

        return data_batches_np


# 用于获取指定索引的注释信息。这包括RGB帧索引、全局到板空间的变换矩阵、分割掩码等。
# 如果某些文件不存在，将返回默认值。
    # def get_annoation(self, index):
    #     index = int(index)
    #     try:
    #         anno = self.pose_list[index] # TODO: Fix this -1 one frame offset
    #     except IndexError:
    #         return {
    #         'rgb_frame_index': -1,
    #         'ego_j2d': None
    #     }

    #     ego_j2d = anno['ego_j2d']

    #     return {
    #         'rgb_frame_index': self.frame_start_index + index,
    #         'ego_j2d': ego_j2d,
    #     }
  '''