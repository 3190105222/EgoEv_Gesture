import cv2
import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import OrderedDict
from dataset import transforms
from scipy.ndimage import gaussian_filter, map_coordinates
from dataset.metrics import compute_3d_errors_batch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)


class Joints3DDataset(Dataset):
    def __init__(self, cfg, root, is_train):
        super().__init__()
    
        self.is_train = is_train
        self.root = root

        self.target_type = cfg.MODEL.TARGET_TYPE
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.sigma = cfg.MODEL.SIGMA

        self.cfg = cfg
        self.db = []

        self.p_dropout=0.95


    def __len__(self,):
        raise NotImplementedError

    def transform(self, data,class1, is_train):    
        if is_train:
            # 应用数据增强
            transform_params = self.get_random_transform_params()
            transformed_data = self.apply_augmentation(data,**transform_params)
        else:
            # 如果不是训练模式，不应用增强
            transformed_data = data 
        
        return {'x': transformed_data, 'class': class1}  

    def apply_augmentation(self, data, do_flip, dropout_mask, do_rotate, rotation_matrix,
                       do_scale, scale_factor, do_translate, translate_x, translate_y,
                       do_noise, noise_std, do_crop, crop_ratio, do_elastic,
                       elastic_alpha, elastic_sigma):
        if do_flip:
            data = self.flip_lr_batch(data)

        if do_rotate and rotation_matrix is not None and not do_flip: #旋转就不翻转
            data = self.rotate_image_batch(data, rotation_matrix)
        # if do_scale and scale_factor is not None:
        #     data = self.random_scale_batch(data, scale_factor)
        if do_translate and translate_x is not None and translate_y is not None and not do_rotate: #旋转就不平移
            data = self.random_translate_batch(data, translate_x, translate_y)        
        # if do_crop and crop_ratio is not None:
        #     data = self.random_crop_batch(data, crop_ratio)
        # if do_noise and noise_std is not None:
        #     data = self.add_noise_batch(data, noise_std)
        if dropout_mask:
            data = self.random_dropout_batch(data, self.p_dropout)
        # if do_elastic and elastic_alpha is not None and elastic_sigma is not None:
        #     data = self.elastic_deformation_batch(data, elastic_alpha, elastic_sigma)
            
        return data
    
    def random_scale_batch(self, data, scale_factor):
        batch_size, height, width = data.shape
        scaled_batch = np.zeros_like(data)
        
        for i in range(batch_size):
            scaled_img = cv2.resize(data[i], None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
            scaled_height, scaled_width = scaled_img.shape
            
            if scaled_height > height or scaled_width > width:
                scaled_img = cv2.resize(scaled_img, (width, height), interpolation=cv2.INTER_LINEAR)
            else:
                padded_img = np.zeros((height, width), dtype=data.dtype)
                padded_img[:scaled_height, :scaled_width] = scaled_img
                scaled_img = padded_img
            
            scaled_batch[i] = scaled_img
        
        return scaled_batch

    def random_translate_batch(self, data, translate_x, translate_y):
        batch_size, height, width = data.shape
        translated_batch = np.zeros_like(data)
        
        translation_matrix = np.float32([[1, 0, translate_x],
                                        [0, 1, translate_y]])
        
        for i in range(batch_size):
            translated_img = cv2.warpAffine(data[i], translation_matrix, (width, height), flags=cv2.INTER_LINEAR)
            translated_batch[i] = translated_img
        
        return translated_batch

    def add_noise_batch(self, data, noise_std):
        noise = np.random.normal(0, noise_std, data.shape)
        noisy_data = data + noise
        noisy_data = np.clip(noisy_data, 0, 255).astype(data.dtype)
        return noisy_data

    def random_crop_batch(self, data, crop_ratio):
        batch_size, height, width = data.shape
        crop_height = int(height * crop_ratio)
        crop_width = int(width * crop_ratio)
        cropped_batch = np.zeros_like(data)
        
        for i in range(batch_size):
            y_start = np.random.randint(0, height - crop_height)
            x_start = np.random.randint(0, width - crop_width)
            cropped_img = data[i, y_start:y_start+crop_height, x_start:x_start+crop_width]
             # 将裁剪后的图像填充到原始尺寸
            padded_img = np.zeros((height, width), dtype=data.dtype)
            y_start_pad = (height - crop_height) // 2
            x_start_pad = (width - crop_width) // 2
            padded_img[y_start_pad:y_start_pad+crop_height, x_start_pad:x_start_pad+crop_width] = cropped_img
            
            cropped_batch[i] = padded_img
        
        return cropped_batch
    

    
    def rotate_image_batch(self,data, rotation_matrix):
        batch_size, height, width = data.shape
        rotated_batch = np.zeros_like(data)
        
        for i in range(batch_size):
            M = rotation_matrix[:2, :3]  # Extract 2x2 rotation matrix for image
            dst_img = cv2.warpAffine(data[i], M, (width,height ), flags=cv2.INTER_AREA)
            rotated_batch[i] = dst_img
        
        return rotated_batch


    def random_dropout_batch(self,data,p_dropout):
        n, h, w = data.shape
        for i in range(n):
            # 使用 Sobel 算子检测边缘
            sobel_x = cv2.Sobel(data[i], cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(data[i], cv2.CV_64F, 0, 1, ksize=3)
            edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            # 根据边缘强度生成掩码
            edge_mask = edge_magnitude > np.percentile(edge_magnitude, 50)
            # 生成随机掩码
            random_mask = np.random.rand(h, w) < p_dropout
            # 结合边缘掩码和随机掩码
            combined_mask = np.logical_and(edge_mask, random_mask)
            # 应用掩码
            data[i, :, :] = data[i, :, :] * combined_mask
        return data
# n,h,w = data.shape
#         dropout = np.random.rand(h,w) < p_dropout
#         for i in range(n):
#             data[i,:,:] = data[i,:,:] * dropout
#         return data

    
    def flip_lr_batch(self,data):
        flip_types = ['horizontal', 'vertical', 'both', 'none']
        probabilities = [0.45, 0.45, 0.1, 0]  # 对应上述翻转类型的概率

        # 随机选择翻转类型
        flip_type = np.random.choice(flip_types, p=probabilities)

        if flip_type == 'horizontal':
            # 水平翻转
            transformed_data = np.flip(data, axis=2)
        elif flip_type == 'vertical':
            # 竖直翻转
            transformed_data = np.flip(data, axis=1)
        elif flip_type == 'both':
            # 同时进行水平和竖直翻转
            transformed_data = np.flip(data, axis=(1, 2))

        return transformed_data
        
     
    def get_random_transform_params(self, p_flip=0, p_dropout=0, p_rotate=0,
                                p_scale=0, p_translate=0, p_noise=0,
                                p_crop=0, p_elastic=0,
                                scale_range=(0.8, 1.2), translate_range=(-50, 50),
                                noise_std=5, crop_ratio=0.9,
                                elastic_alpha=50, elastic_sigma=5,
                                rotation_range=(-45, 45)):
        params = {}
        params['do_flip'] = np.random.rand() < p_flip
        params['dropout_mask'] = np.random.rand() < p_dropout
        params['do_rotate'] = np.random.rand() < p_rotate
        params['do_scale'] = np.random.rand() < p_scale
        params['do_translate'] = np.random.rand() < p_translate
        params['do_noise'] = np.random.rand() < p_noise
        params['do_crop'] = np.random.rand() < p_crop
        params['do_elastic'] = np.random.rand() < p_elastic
        params['translate_x'] = 0
        params['translate_y'] = 0
        params['noise_std'] = 0
        params['scale_factor'] = 1.0  # 默认不缩放
        params['crop_ratio'] = 1.0    # 默认不裁剪
        params['elastic_alpha'] = 0   # 默认不进行弹性变形
        params['elastic_sigma'] = 0   # 默认不进行弹性变形
        if params['do_rotate']:
        # 随机生成旋转角度
            angle_degrees = np.random.uniform(rotation_range[0], rotation_range[1])
            params['rotation_matrix'] = self.get_rotation_matrix(angle_degrees)
        else:
            params['rotation_matrix'] = None
        if params['do_scale']:
            params['scale_factor'] = np.random.uniform(scale_range[0], scale_range[1])
        if params['do_translate']:
            params['translate_x'] = np.random.randint(translate_range[0], translate_range[1])
            params['translate_y'] = np.random.randint(translate_range[0], translate_range[1])
        if params['do_noise']:
            params['noise_std'] = noise_std
        if params['do_crop']:
            params['crop_ratio'] = crop_ratio
        if params['do_elastic']:
            params['elastic_alpha'] = elastic_alpha
            params['elastic_sigma'] = elastic_sigma
        
        return params
    
    def get_rotation_matrix(self,angle_degrees):
        angle_rad = np.radians(angle_degrees)
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])
        return rotation_matrix
    
    def elastic_deformation_batch(self, data, alpha, sigma):
        batch_size, height, width = data.shape
        deformed_batch = np.zeros_like(data)
        
        dx = gaussian_filter((np.random.rand(height, width) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(height, width) * 2 - 1), sigma) * alpha
        
        for i in range(batch_size):
            x, y = np.meshgrid(np.arange(width), np.arange(height))
            indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
            deformed_img = map_coordinates(data[i], indices, order=1).reshape(height, width)
            deformed_batch[i] = deformed_img
        
        return deformed_batch