# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*torchcodec.*",
    category=UserWarning,
    module="torchaudio"
)
import os
import sys
import argparse
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from speakerlab.utils.utils import set_seed, get_logger, AverageMeters, ProgressMeter, accuracy
from speakerlab.utils.config import build_config
from speakerlab.utils.builder import build
from speakerlab.utils.epoch import EpochCounter, EpochLogger


parser = argparse.ArgumentParser(description='Speaker Network Training')
parser.add_argument('--config', default='', type=str, help='Config file for training')
parser.add_argument('--resume', default=True, type=bool, help='Resume from recent checkpoint or not')
parser.add_argument('--seed', default=1234, type=int, help='Random seed for training.')
parser.add_argument('--gpu', nargs='+', help='GPU id to use.')

class ModelWrapper(nn.Module):
    def __init__(self, embedding_model, classifier):
        super().__init__()
        self.embedding_model = embedding_model
        self.classifier = classifier

    def forward(self, x):
        # embedding_model 必须返回 (embed, local_feat)
        embed, local_feat = self.embedding_model(x)

        # classifier 只能吃 embed
        logits = self.classifier(embed)

        return embed, local_feat, logits


def main():
    args, overrides = parser.parse_known_args(sys.argv[1:])
    config = build_config(args.config, overrides, True)

    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    gpu = int(args.gpu[rank])
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl')

    set_seed(args.seed)

    os.makedirs(config.exp_dir, exist_ok=True)
    logger = get_logger('%s/train.log' % config.exp_dir)
    logger.info(f"Use GPU: {gpu} for training.")

    # dataset
    train_dataset = build('dataset', config)
    # dataloader
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    config.dataloader['args']['sampler'] = train_sampler
    config.dataloader['args']['batch_size'] = int(config.batch_size / world_size)
    train_dataloader = build('dataloader', config)

    # model
    embedding_model = build('embedding_model', config)

    if hasattr(config, 'speed_pertub') and config.speed_pertub:
        config.num_classes = len(config.label_encoder) * 3
    else:
        config.num_classes = len(config.label_encoder)

    classifier = build('classifier', config)

    model = ModelWrapper(embedding_model, classifier)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model)


    # loss function，先初始化
    criterion = build('loss', config)

    # 收集所有需要优化的参数：模型主体参数 + Loss函数内部的鉴别器/投影网络参数
    # DDP模型参数可以直接访问.parameters()
    params_to_optimize = list(model.parameters())

    # 假设 FusionLoss 实例名为 criterion
    # 1. 确保 Loss 函数内的 DIM 投影网络参数被添加
    if hasattr(criterion, 'dim_module'):
        print('Adding DIM module parameters to optimizer.')
        params_to_optimize.extend(list(criterion.dim_module.parameters()))

    # 2. 确保 Loss 函数内的分类权重 (如果它在 Loss 内部) 被添加
    if hasattr(criterion, 'arc_margin'):
        print('Adding ArcMarginLoss parameters to optimizer.')
        # 假设 ArcMarginLoss 包含可训练参数，例如 W 矩阵
        params_to_optimize.extend(list(criterion.arc_margin.parameters()))

    # optimizer
    config.optimizer['args']['params'] = params_to_optimize # 使用收集到的参数列表
    optimizer = build('optimizer', config)

    # # optimizer
    # config.optimizer['args']['params'] = model.parameters()
    # optimizer = build('optimizer', config)

    
    # scheduler
    config.lr_scheduler['args']['step_per_epoch'] = len(train_dataloader)
    lr_scheduler = build('lr_scheduler', config)
    config.margin_scheduler['args']['step_per_epoch'] = len(train_dataloader)
    margin_scheduler = build('margin_scheduler', config)

    # others
    epoch_counter = build('epoch_counter', config)
    checkpointer = build('checkpointer', config)

    epoch_logger = EpochLogger(save_file=os.path.join(config.exp_dir, 'train_epoch.log'))

    # resume from a checkpoint
    if args.resume:
        checkpointer.recover_if_possible(device='cuda')

    # ===== 打印模型参数名 =====
    if rank == 0:
        print("=== Model Parameters After Loading Checkpoint ===")
        for name, param in model.named_parameters():
            print(name, param.shape)

    cudnn.benchmark = True

    for epoch in epoch_counter:
        train_sampler.set_epoch(epoch)

        # train one epoch
        train_stats = train(
            train_dataloader,
            model,
            criterion,
            optimizer,
            epoch,
            lr_scheduler,
            margin_scheduler,
            logger,
            config,
            rank,
        )

        if rank == 0:
            # log
            epoch_logger.log_stats(
                stats_meta={"epoch": epoch},
                stats=train_stats,
            )
            # save checkpoint
            if epoch % config.save_epoch_freq == 0:
                checkpointer.save_checkpoint(epoch=epoch)

        dist.barrier()

def train(train_loader, model, criterion, optimizer, epoch, lr_scheduler, margin_scheduler, logger, config, rank):
    train_stats = AverageMeters()
    train_stats.add('Time', ':6.3f')
    train_stats.add('Data', ':6.3f')
    # 新增 DIM Loss 和 AAM Loss 的记录项
    train_stats.add('Total_Loss', ':.4e') # 原来的 'Loss' 改名为 'Total_Loss'
    train_stats.add('DIM_Loss', ':.4e')
    train_stats.add('AAM_Loss', ':.4e') 
    train_stats.add('Acc@1', ':6.2f')
    train_stats.add('Lr', ':.3e')
    train_stats.add('Margin', ':.3f')
    progress = ProgressMeter(
        len(train_loader),
        train_stats,
        # 这里也要对应更新，使用新的总损失名称
        prefix="Epoch: [{}]".format(epoch)
    )

    #train mode
    model.train()

    end = time.time()
    for i, (x, y) in enumerate(train_loader):
        # data loading time
        train_stats.update('Data', time.time() - end)

        # update
        iter_num = (epoch-1)*len(train_loader) + i
        lr_scheduler.step(iter_num)
        margin_scheduler.step(iter_num)

        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        embed, local_feat, logits = model(x)

        # =================================================================
        # 步骤 1: 接收两个返回值：总损失（用于反向传播）和损失指标字典
        loss_total, loss_metrics = criterion(
            (embed, local_feat, logits),
            y
        )
        # =================================================================


        acc1 = accuracy(logits, y)


        # compute gradient and do optimizer step
        optimizer.zero_grad()
        # 步骤 2a: 使用总损失（loss_total）进行反向传播
        loss_total.backward() 
        optimizer.step()

        # recording
        # =================================================================
        # 步骤 2b: 使用字典中的各项损失来更新 train_stats
        train_stats.update('Total_Loss', loss_metrics['Total_loss'], x.size(0))
        train_stats.update('DIM_Loss', loss_metrics['DIM_loss'], x.size(0))
        train_stats.update('AAM_Loss', loss_metrics['AAM_loss'], x.size(0))
        # =================================================================
        train_stats.update('Acc@1', acc1.item(), x.size(0))
        train_stats.update('Lr', optimizer.param_groups[0]["lr"])
        train_stats.update('Margin', margin_scheduler.get_margin())
        train_stats.update('Time', time.time() - end)

        if rank == 0 and i % config.log_batch_freq == 0:
            logger.info(progress.display(i))

        end = time.time()

    key_stats={
        # 确保这里也使用新的 Total_Loss 名称
        'Avg_loss': train_stats.avg('Total_Loss'), 
        'Avg_acc': train_stats.avg('Acc@1'),
        'Lr_value': train_stats.val('Lr')
    }
    return key_stats

if __name__ == '__main__':
    main()
