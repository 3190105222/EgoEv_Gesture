import sys; sys.path.append('../')
# import cv2
# import random
# import json
import logging
# import pickle
import numpy as np
# import os
# import torch


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


        # Load the numpy file
        data = np.load(self.data_path)
        data[:,3] = (data[:,3] - data[0,3])/1000
        class1 = self.generate_class(self.data_path)

        return data,class1
