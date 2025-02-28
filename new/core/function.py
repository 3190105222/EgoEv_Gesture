import time
import logging
import torch
from pathlib import Path
from core.inference import get_j2d_from_hms
from core.evaluate import accuracy, root_accuracy
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
logger = logging.getLogger(__name__)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def compute_fn(model, batch, prev_buffer=None, prev_key=None, batch_first=False):
    device = torch.device(os.environ["DEVICE"] if "DEVICE" in os.environ else "cpu")

    inps = batch['x'].to(device).float()
    # gt_seg = batch['seg'].to(device)
    classes = batch['class'].to(device)

    odd_channels = inps[:, 1::2, :, :]
    even_channels = inps[:, 0::2, :, :]
    odd_channels_reshaped = odd_channels.permute(1, 0, 2, 3).unsqueeze(2)
    even_channels_reshaped = even_channels.permute(1, 0, 2, 3).unsqueeze(2)

    inp_ok = torch.cat((even_channels_reshaped, odd_channels_reshaped), dim=2)

    # outputs = model(inp_ok, prev_buffer, prev_key, batch_first)
    outputs = model(inp_ok, prev_buffer, prev_key, batch_first)
    return inps, outputs, classes
def check_data(data):
    if torch.isnan(data).any() or torch.isinf(data).any():
        raise ValueError("Data contains NaN or Inf values")

def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                raise ValueError(f"Gradient for {name} contains NaN or Inf values")
def train(config, train_loader, model, criterion, optimizer, epoch, writer_dict):
    batch_time = AverageMeter()
    losses = AverageMeter()
    total_correct = 0
    total_samples = 0

    scaler = GradScaler()
    model.train()

    end = time.time()

    total_correct = 0
    total_samples = 0
    for i, batch in enumerate(train_loader):
        check_data(batch['x'])
        check_data(batch['class'])
        if i > config.TRAIN_ITERATIONS_PER_EPOCH:
            break

        # inputs, outputs, gt_classes = compute_fn(model, batch)

        # pred_cla = outputs
        # # print(type(pred_cla))
        # # pred_cla = torch.as_tensor(pred_cla)
        # loss_cla = criterion['cla'](pred_cla, gt_classes)

        # loss = loss_cla

        # loss.backward()
        # optimizer.step()
        optimizer.zero_grad()
# 使用 autocast 包装前向传播
        with autocast():
            inputs, outputs, gt_classes = compute_fn(model, batch)
            pred_cla = outputs
            loss_cla = criterion['cla'](pred_cla, gt_classes)
            loss = loss_cla
        
        # 检查 loss 是否为 NaN
        if torch.isnan(loss).any():
            logger.error(f"Loss is NaN at epoch {epoch}, iteration {i}. Terminating training.")
            raise ValueError("Loss is NaN. Training terminated.")
        
        scaler.scale(loss).backward()
        # 添加梯度裁剪
        scaler.unscale_(optimizer)  # 将梯度反缩放为原始值
        clip_grad_norm_(model.parameters(), max_norm=1)  # 设置最大梯度范数为1.0
        scaler.step(optimizer)
        scaler.update()
        
        losses.update(loss.item(), inputs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        # 计算预测正确的数量
        _, predicted = torch.max(pred_cla, 1)
        correct = (predicted == gt_classes).sum().item()
        total_correct += correct
        total_samples += gt_classes.size(0)

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      loss=losses)
            logger.info(msg)

        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        writer.add_scalar('train_loss', losses.val, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1
    epoch_accuracy = total_correct / total_samples if total_samples > 0 else 0
    msg = 'Epoch: [{0}] Complete\tAccuracy: {1:.4f} ({2} total, {3} correct)'.format(
    epoch, epoch_accuracy * 100, total_samples, total_correct)
    logger.info(msg)

# 将整个epoch的准确率记录到TensorBoard
    writer.add_scalar('train_accuracy', epoch_accuracy, epoch)

def validate(config, val_loader, valid_dataset, model, criterion, final_output_dir, tb_log_dir,writer_dict, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    total_correct = 0
    total_samples = 0

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, batch in enumerate(val_loader):
            # inputs, outputs, gt_classes = compute_fn(model, batch)

            # pred_cla = outputs
            # loss_cla = criterion['cla'](pred_cla, gt_classes)
            # # pred_seg = outputs['seg']
            # # loss_seg = criterion['seg'](pred_seg, gt_seg)

            # loss = loss_cla
            with autocast():
                inputs, outputs, gt_classes = compute_fn(model, batch)
                pred_cla = outputs
                loss_cla = criterion['cla'](pred_cla, gt_classes)
                loss = loss_cla
            losses.update(loss.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            _, predicted = torch.max(pred_cla, 1)
            correct = (predicted == gt_classes).sum().item()
            total_correct += correct
            total_samples += gt_classes.size(0)

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses)
                logger.info(msg)

        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', losses.avg, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1
    
    epoch_accuracy = total_correct / total_samples if total_samples > 0 else 0
    msg = 'Validation: [{0}] Complete\tAccuracy: {1:.4f} ({2} total, {3} correct)'.format(
        epoch, epoch_accuracy * 100, total_samples, total_correct)
    logger.info(msg)

    writer.add_scalar('valid_accuracy', epoch_accuracy, global_steps)
    
    return losses.avg