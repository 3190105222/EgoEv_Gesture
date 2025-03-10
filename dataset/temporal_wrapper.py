import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from dataset import transforms
from settings import config
import matplotlib.pyplot as plt

class TemoralWrapper(Dataset):
    def __init__(self, dataset, augment=False):
        super().__init__()
        self.dataset = dataset
        self.augment = augment

    
    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, idx):
        # 假设 self.dataset[idx] 返回一个包含 'x' 和 'hms' 键的字典
        dataset = self.dataset[idx]
        data = dataset['x'] #torch.Size([44, 720, 1280])
        class1 = dataset['class']
        # labels = dataset['j2d'] #torch.Size([21*m,3])
        # seg_gt = dataset['seg']
        n_max =96   #46 #
        m_max = 210 #105  #
        # Padding operation for data
        n, height, width = data.shape
        acts = (n/2-3)//5+1
        nreal = 10 * acts -4  #0 #
        data = data[:int(nreal),:,:]
        mreal = 21 * acts #0 #
        # labels = labels[:int(mreal),:]
        n, height, width = data.shape
        assert data.shape[0] % 2 == 0, "数据集中的图像数量必须是偶数"

        if n<n_max:
            pad_height = n_max - n
            padded_data = np.pad(data, ((0, pad_height), (0, 0), (0, 0)), mode='constant')
        else:
            padded_data = data[:n_max]       

        
        if isinstance(padded_data, np.ndarray):
            padded_data = torch.from_numpy(padded_data.copy())
       
        data_dict = {
            'x': padded_data,

            'class':class1
        }
        return data_dict


    def apply_transform(self, data, transform, flip_lr):
        # 应用仿射变换和左右翻转
        transformed_data = [transforms.functional.affine(data[i], transform, scale=1, translate=(0, 0)) for i in range(len(data))]
        if flip_lr:
            transformed_data = [transforms.functional.hflip(item) for item in transformed_data]
        return transformed_data
