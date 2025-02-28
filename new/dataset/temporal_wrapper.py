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
            'x': padded_data,  # 填充后的数据 torch.Size([146, 720, 1280])
            # 'j2d': padded_labels,  # 原始数据（如果需要） torch.Size([15, 180, 320])
            # 'seg':padded_seg
            'class':class1
        }
        return data_dict
    # def __getitem__(self, idx):
    #     # 获取原始数据和标签
    #     dataset = self.dataset[idx]
    #     original_data = dataset['x']
    #     hms = dataset['hms']

    #     if self.augment:
    #         # 应用随机变换和翻转
    #         transform = self.get_random_transform()
    #         flip_lr = self.get_random_flip_lr()
    #         # 假设变换和翻转应用于数据集中的每个项目
    #         original_data = [self.apply_transform(item, transform, flip_lr) for item in original_data]


    def apply_transform(self, data, transform, flip_lr):
        # 应用仿射变换和左右翻转
        transformed_data = [transforms.functional.affine(data[i], transform, scale=1, translate=(0, 0)) for i in range(len(data))]
        if flip_lr:
            transformed_data = [transforms.functional.hflip(item) for item in transformed_data]
        return transformed_data

# 使用示例
# dataset = TemoralWrapper(your_dataset, augment=True)
# for data, mask, labels in dataset:
#     pass

'''
class TemoralWrapper(Dataset):
    def __init__(self, dataset, timesteps, augment) -> None:
        super().__init__()

        self.dataset = dataset
        self.timesteps = timesteps
        self.augment = augment  

    def get_random_transform(self):
        target_width, target_height = config.MODEL.IMAGE_SIZE
        
        center_shift_x = target_width / 2 + target_width * 0.1 * random.uniform(-1, 1)
        center_shift_y = target_height / 2 + target_height * 0.1 * random.uniform(-1, 1)
        center = np.array([center_shift_x, center_shift_y])
        scale = 1 + random.uniform(-1, 1) * config.DATASET.SCALE_FACTOR
        rot = random.uniform(-1, 1) * config.DATASET.ROT_FACTOR
        
        output_size = np.array((target_width, target_height))

        A = transforms.get_affine_transform(center, scale, rot, output_size)

        return A
    
    def get_random_flip_lr(self):
        if random.random() < 0.5:
            flip = True
        else:
            flip = False
        return flip
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        start_index = max(0, idx - self.timesteps)
        end_index = idx
        
        end_index += self.timesteps - (end_index - start_index)

        kwargs = {'start_index': idx}
        if self.augment:
            kwargs['augment'] = True	
        
            # kwargs['A'] = self.get_random_transform()
            # kwargs['flip_lr'] = self.get_random_flip_lr()

            kwargs['flip_axis'] = self.get_random_flip_axis()    
            
        data = []
        for i in range(start_index, end_index): 
            kwargs['offset_index'] = i - start_index
            # data.append(self.dataset[i, kwargs])
            data_item = self.dataset[i]
            data.append(data_item)
            
        return data

    def visualize(self, *args, **kwargs):
        return self.dataset.visualize(*args, **kwargs)

    def evaluate_joints(self, *args, **kwargs):
        return self.dataset.evaluate_joints(*args, **kwargs)
    
    def evaluate_dataset(self, *args, **kwargs):
        return self.dataset.evaluate_dataset(*args, **kwargs)
'''