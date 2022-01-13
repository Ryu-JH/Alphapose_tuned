"""Script for multi-gpu training."""
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torchsummary

# from alphapose.models.FastPose import createModel
from alphapose.models import builder
from alphapose.opt import cfg, logger, opt
from alphapose.utils.logger import board_writing, debug_writing
from alphapose.utils.metrics import DataLogger, calc_accuracy, calc_integral_accuracy, evaluate_mAP
from alphapose.utils.transforms import get_func_heatmap_to_coord

num_gpu = torch.cuda.device_count()
valid_batch = 1 * num_gpu
if opt.sync:
    norm_layer = nn.SyncBatchNorm
else:
    norm_layer = nn.BatchNorm2d

def main():
    model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
    model.load_state_dict(torch.load('/home/ryu/AlphaPose/pretrained_models/fast_res50_256x192.pth'))
    # print(model)
    # model = nn.Sequential(
    #     model,
    #     nn.Conv2d(
    #             17, 15, kernel_size=3, stride=1, padding=1)
    # )
    model = model.cuda()
    torchsummary.summary(model, (3, 224, 224))
    for name, layer in enumerate(model.children()):
        print(name, type(name))
        print('--------------------')
        print(layer, type(layer))
        print('--------------------')
    

    criterion = builder.build_loss(cfg.LOSS).cuda()

    if cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(m.parameters(), lr=cfg.TRAIN.LR)
    elif cfg.TRAIN.OPTIMIZER == 'rmsprop':
        optimizer = torch.optim.RMSprop(m.parameters(), lr=cfg.TRAIN.LR)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.TRAIN.LR_STEP, gamma=cfg.TRAIN.LR_FACTOR)

    writer = SummaryWriter('.tensorboard/{}-{}'.format(opt.exp_id, cfg.FILE_NAME))

    for i in range(10):
        opt.epoch = i
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']

        logger.info(f'############# Starting Epoch {opt.epoch} | LR: {current_lr} #############')

        # Training
        loss, miou = train(opt, train_loader, model, criterion, optimizer, writer)
        logger.epochInfo('Train', opt.epoch, loss, miou)

        lr_scheduler.step()

        if (i + 1) % opt.snapshot == 0:
            # Save checkpoint
            torch.save(m.module.state_dict(), './exp/{}-{}/model_{}.pth'.format(opt.exp_id, cfg.FILE_NAME, opt.epoch))
# m = createModel().cuda()
# print(m)

if __name__ == "__main__":
    main() 