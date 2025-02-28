import platform
import copy
import numpy as np
from rich import print
from pathlib import Path
from torch.utils.data import Dataset
import os
from dataset.Joints3DDataset import Joints3DDataset
from dataset.representation import EROS, LNES, EventFrame
from dataset.egoevent_real import RealEventStream
from settings import config as cfg
import torch
import matplotlib.pyplot as plt

def get_representation(cfg, width, height):
    representation = cfg.DATASET.REPRESENTATION

    if representation == 'EROS':
        eros_config = cfg.DATASET.EROS
        repr = EROS(eros_config.kernel_size, height, width, eros_config.DECAY_BASE)
    elif representation == 'LNES':
        repr = LNES(cfg, height, width)
    elif representation == 'EventFrame':
        repr = EventFrame(cfg, height, width)
    else:
        raise NotImplementedError

    return repr

'''''
继承自Joints3DDataset。
处理单个数据序列，包括初始化、数据增强、数据获取等。
'''''

class SingleSequenceDataset(Joints3DDataset):
    def __init__(self, cfg, data_path, is_train):
        super().__init__(cfg, data_path, is_train)
        self.repr = get_representation(cfg, 1280, 720)

        self.dataset = RealEventStream(data_path, cfg, is_train)

        self.data_path = data_path

        self.num_joints = cfg.NUM_JOINTS

        self.is_train = is_train
    
    def isvalid(self):
        return len(self.dataset) > 0   
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            idx, kwargs = idx
        else:
            kwargs = {}
        data = {}
        data_batch, class1= self.dataset[idx]
        # lines = label.strip().split('\n')
        # coords = [list(map(int, line.split()[1:3])) for line in lines[:21]]
        # data_batch 是x y p t
        # label 是读取的整个txt
        # if self.is_train:
        #     dist = np.random.uniform(0.2, 1.0)
        #     n_events = data_batch.shape[0]
        #     retain_count = min(int(0.1 * n_events), 1000)
        #     retain_indices_front = np.arange(retain_count)  # 最前面的保留数据
        #     retain_indices_back = np.arange(n_events - retain_count, n_events)  # 最后面的保留数据
        #     retain_indices = np.concatenate((retain_indices_front, retain_indices_back))
        #     remaining_count = n_events - 2 * retain_count
        #     if remaining_count <= 0:
        #         choices = retain_indices
        #     else:
        #         remaining_indices = np.arange(retain_count, n_events - retain_count)
        #         choices = np.random.choice(remaining_indices, remaining_count, replace=False)
        #         choices = np.concatenate((retain_indices, choices))
        #     # choice_len = np.random.randint(int(n_events * dist), n_events)
        #     # choices = np.random.choice(np.arange(n_events), choice_len, replace=False)

        #     data_batch = data_batch[choices, :]
        
        processed_data = self.repr(data_batch)    #data_batch是原始数据

        # inp_processed = inp.copy()
        inp_processed = np.concatenate(processed_data, axis=2)
        new_array = np.transpose(inp_processed, (2, 1, 0))
        # coord_x coord_y 这俩个size相同
        transformed_data = super().transform(new_array, class1,self.is_train )
# 这里的transformed_data 是一个正一个负排列的

        
        return transformed_data

class EgoEvent(Dataset):
    def __init__(self, cfg, split='train', finetune=False):
        super().__init__()

        cfg = copy.deepcopy(cfg)  # 确保 cfg 被复制并赋值
        self.total_length = 0  # 初始化 total_length
        cfg.DATASET.TYPE = 'Real'
        train_dataset_root = Path(cfg.DATASET.TRAIN_ROOT)
        valid_dataset_root = Path(cfg.DATASET.VALID_ROOT)

        is_train = 1 
        self.is_train = is_train
        self.datasets = []
        self.data_paths = []
        self.lengths = []   # 初始化 lengths 列表
        self.total_length = 0  # 初始化 total_length

        assert split in ['train', 'valid', 'test']

        if split == 'train':
            for item in os.listdir(train_dataset_root):
                # 确保只处理以 'subj' 开头的文件夹
                if item.startswith('subj'): # or item.startswith('subj2'):#or item.startswith('subj3')
                    subject_path = train_dataset_root / item
                    for hand in ['left', 'right']:  # 处理 left 和 right 子文件夹
                    # for hand in ['left']:  # 处理 left 和 right 子文件夹
                        hand_path = subject_path / hand
                        if hand_path.is_dir():  # 确保是目录
                            for data_path in hand_path.glob('*.npy'):
                                # self.data_paths.append(data_path)
                                # ## 这里为了保证单手
                                filename = os.path.basename(data_path)
                                parts = filename.split('_')
                                x_str = parts[0].split('act')[1]
                                
                                if x_str.isdigit():  # 确保x_str是数字
                                    x = int(x_str)
                                    # 检查x的值是否在0-11或38-49范围内
                                    if (x >= 0 and x <= 36) or(x >= 38 and x <= 74) :
                                    # if (x >= 0 and x <= 6) :
                                        y_str = filename.split('_')[1].split('.')[0]  # 先按下划线分割，
                                        ##到这里都是为了保证单手
                                        y = int(y_str)
                                        if(y>=0):
                                            self.data_paths.append(data_path)
        elif split == 'valid':
            for item in os.listdir(valid_dataset_root):
                # 确保只处理以 'subj' 开头的文件夹
                if item.startswith('subj'):
                    subject_path = valid_dataset_root / item
                    for hand in ['left', 'right']:  # 处理 left 和 right 子文件夹
                        hand_path = subject_path / hand
                        if hand_path.is_dir():  # 确保是目录
                            for data_path in hand_path.glob('*.npy'):
                                # self.data_paths.append(data_path)
            # for item in os.listdir(train_dataset_root):
            #     # 确保只处理以 'subj' 开头的文件夹
            #     if item.startswith('subj'):
            #         subject_path = train_dataset_root / item
            #         for hand in ['left', 'right']:  # 处理 left 和 right 子文件夹
            #         # for hand in ['right']:  # 处理 left 和 right 子文件夹
            #             hand_path = subject_path / hand
            #             if hand_path.is_dir():  # 确保是目录
            #                 for data_path in hand_path.glob('*.npy'):
            #                     self.data_paths.append(data_path)
                                filename = os.path.basename(data_path)
                                parts = filename.split('_')
                                x_str = parts[0].split('act')[1]
                                
                                if x_str.isdigit():  # 确保x_str是数字
                                    x = int(x_str)
                                    # 检查x的值是否在0-11或38-49范围内
                                    if (x >= 0 and x <= 36)or (x>=38 and x<=74):
                                    # if (x >= 0 and x <= 10) or(x >= 38 and x <= 48):
                                        y_str = filename.split('_')[1].split('.')[0]  # 先按下划线分割，
                                        ##到这里都是为了保证单手
                                        y = int(y_str)
                                        if(y>=0):
                                            self.data_paths.append(data_path)

        self.total_length = len(self.data_paths)

        if split == 'train':
            is_train = True
        else:
            is_train = False

        self.is_train = is_train
   
        for data_path in self.data_paths:
            dataset = SingleSequenceDataset(cfg, data_path, is_train)
            self.datasets.append(dataset)

    def __len__(self):
        # 返回数据集的总长度
        return self.total_length
    
    def __getitem__(self, idx):
        # 直接从self.datasets中获取数据，而不是递归调用__getitem__
        dataset = self.datasets[idx]  # 假设self.datasets[idx]是一个SingleSequenceDataset实例
        
        data = dataset[idx]  # j2d :torch.Size([210, 3])
        # data['seg'] = data['seg'].squeeze(0)
        # hms：torch.Size([210, 180, 320])

        return data     
    
