import utils
import argparse
import os
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.distributed as dist
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from settings import config as cfg
from core.loss import CustomClassificationLoss
from core.function import train, validate
from utils.utils import get_optimizer, save_checkpoint, create_logger

from dataset import EgoEvent, TemoralWrapper
from model import EEG

import numpy as np


def main():

    os.environ['DEVICE'] = 'cuda:2'
    device = torch.device(os.environ["DEVICE"] if "DEVICE" in os.environ else "cpu")

    base_exp_name = f'EXPERMENT'


    logger, final_output_dir, tb_log_dir = create_logger(cfg, base_exp_name)

    logger.info(cfg)

    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = EEG(cfg).to(device)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    criterions = {
        'cla': CustomClassificationLoss().to(device)
    }

    TrainDataset = EgoEvent

    train_dataset = TrainDataset(cfg, split='train')

    valid_dataset = EgoEvent(cfg, split='valid')
    train_dataset = TemoralWrapper(train_dataset, augment=True)
    valid_dataset = TemoralWrapper(valid_dataset, augment=False)

    batch_size = cfg.BATCH_SIZE
    n_workers = 4

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=3
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=3
    )

    best_perf = 1e6
    best_model = False
    last_epoch = -1
    optimizer = get_optimizer(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(final_output_dir, cfg.MODEL.CHECKPOINT_PATH)

    pretrain_model_path = ''
    if os.path.isfile(pretrain_model_path):
        logger.info(f"=> loading pre-trained model '{pretrain_model_path}'")
        checkpoint = torch.load(pretrain_model_path, map_location=device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        logger.info("=> loaded pre-trained model (strict=False)")
    else:
        logger.info(f"=> no pre-trained model found at '{pretrain_model_path}'")

    if os.path.isfile(checkpoint_file) and checkpoint_file.endswith('.pth'):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        last_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR, last_epoch=last_epoch
    )

    for epoch in trange(begin_epoch, cfg.TRAIN.END_EPOCH, desc='Epoch'):

            train(cfg, train_loader, model, criterions, optimizer, epoch, writer_dict)
            perf_indicator = validate(cfg, valid_loader, valid_dataset, model, criterions, final_output_dir, tb_log_dir, writer_dict, epoch)
            lr_scheduler.step()

        if perf_indicator <= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint(epoch + 1, {
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.state_dict(),
            'best_state_dict': model.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir, tb_log_dir)

    final_model_state_file = os.path.join(final_output_dir, 'final_state.pth')
    logger.info('=> saving final model state to {}'.format(final_model_state_file))
    torch.save(model.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()