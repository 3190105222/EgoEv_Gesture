import cv2
import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import OrderedDict
from dataset import transforms

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
            do_flip, dropout_mask, rotation_matrix = self.get_random_transform_params()
            transformed_data = self.apply_augmentation(data,do_flip, dropout_mask, rotation_matrix)
        else:
            # 如果不是训练模式，不应用增强
            transformed_data = data 
        
        return {'x': transformed_data, 'class': class1}  

    def apply_augmentation(self, data,do_flip, dropout_mask, rotation_matrix):
        if do_flip==1:
            transformed_data = self.flip_lr_batch(data)
            return transformed_data 
        if dropout_mask==1:
            transformed_data = self.random_dropout_batch(data,self.p_dropout)
            return transformed_data 
        if rotation_matrix is not None:
            transformed_data = self.rotate_image_batch(data, rotation_matrix)
            return transformed_data 
        return data 
    
    def rotate_image_batch(self,data, rotation_matrix):
        batch_size, height, width = data.shape
        rotated_batch = np.zeros_like(data)
        
        for i in range(batch_size):
            M = rotation_matrix[:2, :3]  # Extract 2x2 rotation matrix for image
            dst_img = cv2.warpAffine(data[i], M, (width,height ), flags=cv2.INTER_AREA)
            rotated_batch[i] = dst_img
        
        return rotated_batch


    def random_dropout_batch(self,data,p_dropout):
        n,h,w = data.shape
        dropout = np.random.rand(h,w) < p_dropout
        for i in range(n):
            data[i,:,:] = data[i,:,:] * dropout
        return data

    
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
        
     
    def get_random_transform_params(self, p_flip=0.2, p_dropout=0.2, p_rotate=0.2, max_rotation_angle=10):
        do_flip = np.random.rand() < p_flip
        dropout_mask = np.random.rand() < p_dropout
        do_rotate = np.random.rand() < p_rotate
        rotation_angle = np.random.uniform(-max_rotation_angle, max_rotation_angle) if do_rotate else 0
        rotation_matrix = self.get_rotation_matrix(rotation_angle)
        return do_flip, dropout_mask, rotation_matrix
    
    def get_rotation_matrix(self,angle_degrees):
        angle_rad = np.radians(angle_degrees)
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])
        return rotation_matrix

    def generate_target(self, joints):
        heatmap_size = self.heatmap_size #320 180
        sigma = self.sigma   #2
        image_size = self.image_size #1280 720
        num_joints = joints.shape[0] #105
        num_sets = joints.shape[0] // 21 #5 
        target = np.zeros((num_joints, heatmap_size[1], heatmap_size[0]), dtype=np.float32)
        tmp_size = sigma * 3
        feat_stride = image_size / heatmap_size
# tmp_size 是高斯分布的半径，feat_stride 是图像尺寸与热图尺寸之间的比例。
        for joint_id in range(num_joints):
            num_joint = joint_id // 21
            
            # mu_x 和 mu_y 表示高斯分布的均值（中心）在热图坐标系中的位置
            mu_x = int(joints[joint_id][1] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][2] / feat_stride[1] + 0.5)

            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

            if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] or br[0] < 0 or br[1] < 0:
                continue  # 如果超出边界，则跳过该关节的热图生成

            # Generate gaussian using matrix operations
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

            # 计算热图范围内的高斯核的坐标
            g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
            img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

            # 将高斯核赋值到热图的相应位置
            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]],
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
            )

        return target      
    def save_heatmaps_as_images(self, target, output_dir='./'):
        target = np.asarray(target)
        
        num_heatmaps = target.shape[0]
        
        heatmap_size_y, heatmap_size_x = target.shape[1], target.shape[2]
        
        for i in range(num_heatmaps):
            fig, ax = plt.subplots(figsize=(heatmap_size_x / 100, heatmap_size_y / 100))
            
            ax.imshow(target[i], cmap='jet')  
            ax.set_title(f'Heatmap {i+1}')
            ax.axis('off')
            
            filename = f'{output_dir}target_{i+1}.jpg'
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
    '''
# 这个方法用于生成高斯热图，它接收关节位置和可见性作为输入，
# 并返回热图和权重。它首先初始化热图和权重，然后对于每个关节，
# 根据其位置生成高斯分布，并将其放置在热图上的相应位置。
    def generate_target(self, joints):
  
        num_sets = joints.shape[0] // 21
        target = np.zeros((num_sets,
                        self.heatmap_size[1],
                        self.heatmap_size[0]),
                        dtype=np.float32)
        array_to_visualize= np.zeros((self.heatmap_size[1],
                        self.heatmap_size[0]),
                        dtype=np.float32)
        tmp_size = self.sigma * 3
        feat_stride = self.image_size / self.heatmap_size

        for joint_id in range(len(joints)):
            num_joint = joint_id//21
            
            # mu_x 和 mu_y 表示高斯分布的均值（中心）在热图坐标系中的位置。ul 和 br 分别代表高斯分布的左上角和右下角在热图坐标系中的坐标
            mu_x = int(joints[joint_id][1] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][2] / feat_stride[1] + 0.5)

            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

            # if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
            #         or br[0] < 0 or br[1] < 0:
            if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] or br[0] < 0 or br[1] < 0:
                continue  # 如果超出边界，则跳过该关节的热图生成

            # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

            # 计算热图范围内的高斯核的坐标
            # g_x 和 g_y 是高斯分布内部的坐标，而 img_x 和 img_y 是热图上的坐标。
            g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
            img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

            # 将高斯核赋值到热图的相应位置
            target[num_joint][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
            
            if num_joint==0:
                array_to_visualize[img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target
    
   
 
         '''
# 这个方法（看起来像是一个未完成的实现或一个替代方法）用于生成包含3D位置信息的热图。
# 它与 generate_target 类似，但额外地将3D位置信息编码到热图的通道中。
    def generate_location_maps(self, j2d, vis_j2d):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target = np.zeros((self.num_joints,
                           self.heatmap_size[1],
                           self.heatmap_size[0], 3),
                          dtype=np.float32)

        tmp_size = self.sigma * 3
        feat_stride = self.image_size / self.heatmap_size

        for joint_id in range(self.num_joints):
            mu_x = int(j2d[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(j2d[joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                vis_j2d[joint_id] = 0
                continue

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

            v = vis_j2d[joint_id, 0]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

                # target[joint_id, :, :, 0] *= j3d[joint_id, 0]
                # target[joint_id, :, :, 1] *= j3d[joint_id, 1]
                # target[joint_id, :, :, 2] *= j3d[joint_id, 2]

        target = torch.from_numpy(target)

        return target, vis_j2d

   
# 这些方法用于评估数据集的性能，计算平均误差（MPJPE 和 PAMPJPE）：

# 它们接收配置、帧索引、真实和预测的3D关节位置以及可见性掩码。
# 计算每个序列或关节的误差，并返回以字典形式的错误度量。
#     @classmethod
#     def evaluate_dataset(cls, cfg, frame_indices, all_gt_j3ds, all_preds_j3d, all_vis_j3d):
#         sequence_range = {
#             'walk': [0, 3500],
#             'crouch': [3500, 6500],
#             'pushup': [6500, 9000],
#             'boxing': [9000, 12350],
#             'kick': [12350, 15200],
#             'dance': [15200, 17800],
#             'inter. with env': [17800, 20700],
#             'crawl': [20700, 23800],
#             'sports': [23800, 33000],
#             'jump': [33000, 200000], # max frame index
#         }
                               
#         MPJPE = []
#         PAMPJPE = []
#         for seq_name, seq_range in sequence_range.items():
#             start_index, end_index = seq_range
            
#             current_seq_indices = frame_indices >= start_index             
#             current_seq_indices = np.logical_and(current_seq_indices, frame_indices < end_index)
            
#             gt_j3ds = all_gt_j3ds[current_seq_indices]
#             preds_j3d = all_preds_j3d[current_seq_indices]
#             vis_j3d = all_vis_j3d[current_seq_indices]
            
#             errors, errors_pa = compute_3d_errors_batch(gt_j3ds, preds_j3d, vis_j3d)

#             seq_mpjpe = np.mean(errors) 
#             seq_pampjpe = np.mean(errors_pa)
        
#             MPJPE.append(seq_mpjpe)
#             PAMPJPE.append(seq_pampjpe)
        
#         name_values = []           
#         for i, seq_name in enumerate(sequence_range.keys()):
#             name_values.append((f'{seq_name}_MPJPE', MPJPE[i]))
#         name_values.append(('MPJPE', np.mean(MPJPE)))

#         for i, seq_name in enumerate(sequence_range.keys()):
#             name_values.append((f'{seq_name}_PAMPJPE', PAMPJPE[i]))
#         name_values.append(('PAMPJPE', np.mean(PAMPJPE)))

#         name_values = OrderedDict(name_values)
                
#         return name_values, MPJPE
 
    @classmethod
    def evaluate_joints(cls, cfg, all_gt_j3ds, all_preds_j3d, all_vis_j3d):
        errors, errors_pa = compute_3d_errors_batch(all_gt_j3ds, all_preds_j3d, all_vis_j3d)
        
        MPJPE = np.mean(errors) 
        PAMPJPE = np.mean(errors_pa)

        name_values = []

        heatmap_sequence = {"1-1": 1,  # 0
                            "1-2": 1,  # 1
                            "1-3": 1,  # 2
                            "1-4": 1,  # 3
                            "2-1": 1,  # 4
                            "2-2": 1,  # 5
                            "2-3": 1,  # 6
                            "2-4": 1,  # 7
                            "3-1": 1,  # 8
                            "3-2": 1,  # 9
                            "3-3": 1,  # 10
                            "3-4": 1,  # 11
                            "4-1": 1,  # 12
                            "4-2": 1,  # 13
                            "4-3": 1,  # 14
                            "4-4": 1,  # 15
                            "5-1": 1,  # 16
                            "5-2": 1,  # 17
                            "4-5": 1,  # 18
                            "5-3": 1,  # 19
                            "5-4": 1}  # 20

        for i, joint_name in enumerate(heatmap_sequence):
            name_values.append((f'{joint_name}_MPJPE', errors[i]))
        name_values.append(('MPJPE', MPJPE))

        for i, joint_name in enumerate(heatmap_sequence):
            name_values.append((f'{joint_name}_PAMPJPE', errors_pa[i]))
        name_values.append(('PAMPJPE', PAMPJPE))

        name_values = OrderedDict(name_values)

        return name_values, MPJPE

