import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt

EPS = 1.1920929e-07


class J2dMSELoss(nn.Module):
    def __init__(self, use_target_weight=0):
        super(J2dMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = 0#use_target_weight

    def forward(self, output, target, target_weight):
        num_joints = output.size(1)

        cnt = 0
        loss = 0
        for idx in range(num_joints):
            j2d_pred = output[:, idx]
            j2d_gt = target[:, idx]

            if self.use_target_weight:
                tw = target_weight[:, idx]

                loss += self.criterion(
                    j2d_pred.mul(tw),
                    j2d_gt.mul(tw)
                )
            else:
                loss += self.criterion(j2d_pred, j2d_gt)

        return loss.mean() / num_joints


class HeatMapJointsMSELoss(nn.Module):
    def __init__(self):
        super(HeatMapJointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        # self.use_target_weight = 0#use_target_weight

    def forward(self, output, target):
        # for i in  range(105):
        #     fig, ax = plt.subplots(figsize=(1280 / 80, 720 / 80))
                
        #     ax.imshow(target[i//60][i].detach().cpu().numpy(), cmap='jet') 
        #     ax.set_title(f'heatmap_gt_{i+1}')
        #     ax.axis('off')
        #     dir = '/mnt/sto/wlm/EventEgo3D-wlm/logs/'
        #     filename = f'{dir}heatmap_gt_{i+1}.jpg'
        #     plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        #     plt.close(fig)
        device = torch.device(os.environ["DEVICE"] if "DEVICE" in os.environ else "cpu")
        target = self.generate_target(target)
        target = torch.from_numpy(target)
        target = target.to(device)
        # target1 = self.generate_target(target[1])
        # target = torch.from_numpy(np.stack((target0, target1), axis=0))
        # target = torch.from_numpy(target)
        batch_size = target.size(0) 
        loss = 0
        for i in range(batch_size):
            for idx in range(output.shape[1]):
                heatmap_pred = output[i][idx].squeeze() #torch.Size([180, 320])
                heatmap_gt = target[i][idx].squeeze() #torch.Size([180, 320])

                loss_j= self.criterion(heatmap_pred, heatmap_gt)
                loss += loss_j
        return loss.mean() /batch_size# (2*output.shape[1])
    

    def generate_target(self, joints):
    
        heatmap_size = [320, 180] 
        sigma = 2
        image_size = [1280, 720] 
        num_joints = joints.shape[1] #105
        num_sets = joints.shape[1] // 21 #5 
        target = np.zeros((joints.shape[0],num_joints, heatmap_size[1], heatmap_size[0]), dtype=np.float32)
        tmp_size = sigma * 3
        feat_stride = [4,4]
# tmp_size 是高斯分布的半径，feat_stride 是图像尺寸与热图尺寸之间的比例。
        for i in range(joints.shape[0]):
            for joint_id in range(num_joints):
                num_joint = joint_id // 21
                
                # mu_x 和 mu_y 表示高斯分布的均值（中心）在热图坐标系中的位置
                mu_x = int(joints[i][joint_id][1].item() / feat_stride[0] + 0.5)
                mu_y = int(joints[i][joint_id][2].item() / feat_stride[1] + 0.5)
                if mu_x != 0 or mu_y !=0:
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
                    target[i][joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
                        target[i][joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]],
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
                )
        return target

class CustomClassificationLoss(nn.Module):
    def __init__(self):
        super(CustomClassificationLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, pred_classes, gt_classes):
        return self.criterion(pred_classes, gt_classes)
        # 遍历pred_cla_list中的每个预测
        # for pred_cla in pred_cla_list:
        #     # 取出预测张量
        #     pred_classes = pred_cla['class']
            
        #     # 计算损失并累加到总损失中
        #     loss_cla = criterion(pred_classes, gt_classes)
        #     total_loss += loss_cla
        
        # # 返回平均损失
        # return total_loss / len(pred_cla_list)

class SegmentationLoss(nn.Module):
    def __init__(self) -> None:
        super(SegmentationLoss, self).__init__()

        self.loss = nn.BCELoss()

    def forward(self, output, target, weight=None):
        if weight is None:
            return self.loss(output, target)
        else:
            weight = weight.view(-1, 1, 1, 1)
            return self.loss(output * weight, target * weight)

