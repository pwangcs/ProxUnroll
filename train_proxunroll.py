import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from model.proxunroll import ProxUnroll
from test_proxunroll import test
import torch.optim as optim
import os
import cv2
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from opts import parse_args
import time
import einops
import random
import datetime
from utils import load_checkpoint, checkpoint, TrainData, Logger, time2file_name


STAGE_WEIGHTS = [0.01, 0.01, 0.01, 0.01, 0.01, 0.95]


def compute_pt_loss(criterion, outputs, prox_outputs):
    stage_outputs = outputs[-6:]
    loss = sum(
        w * torch.sqrt(criterion(stage_outputs[i], prox_outputs[i]))
        for i, w in enumerate(STAGE_WEIGHTS)
    )
    return loss


def train(args, network, optimizer, logger, weight_path, result_path1, result_path2=None):
    criterion = nn.MSELoss().to(args.device)
    rank = dist.get_rank() if args.distributed else 0
    dataset = TrainData(args.train_data_path, train_sizes=args.train_sizes)
    num_crops = args.num_train_crops

    if args.distributed:
        dist_sampler = DistributedSampler(dataset, shuffle=True, drop_last=True, seed=args.seed)
        train_data_loader = DataLoader(
            dataset=dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, drop_last=True, pin_memory=True,
            sampler=dist_sampler,
        )
    else:
        dist_sampler = None
        train_data_loader = DataLoader(
            dataset=dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers,
        )

    cr = [0.01, 0.04, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    for epoch in range(args.pretrain_epoch + 1, args.pretrain_epoch + args.epochs + 1):
        if dist_sampler is not None:
            dist_sampler.set_epoch(epoch)
        epoch_loss = 0
        network = network.train()
        start_time = time.time()
        for iteration, data in enumerate(train_data_loader):
            idx = iteration % len(cr)
            for gt in data:
                b, h, w = gt.shape
                gt = gt.float().to(args.device)
                optimizer.zero_grad()
                outputs, prox_outputs, images = network(gt, cr[idx])
                loss = compute_pt_loss(criterion, outputs, prox_outputs)

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                if rank == 0 and (iteration % args.iter_step) == 0:
                    lr = optimizer.param_groups[0]['lr']
                    logger.info(
                        'epoch: {:<3d}, iter: {:<4d}, size: [{}, {}, {}], cr: {:.2f}, loss: {:.4f}, lr: {:.6f}.'.format(
                            epoch, iteration, b, h, w, cr[idx], loss.item(), lr,
                        )
                    )

                if rank == 0 and (iteration % args.iter_step) == 0:
                    image_path = os.path.join(
                        result_path1,
                        'epoch_{}_iter_{}_cr_{}_reso_{}_{}.png'.format(epoch, iteration, cr[idx], h, w),
                    )
                    result_img = einops.rearrange(images[0].detach(), 'c s h w -> (c h) (s w)')
                    result_img = (result_img.cpu().numpy() * 255).astype(np.float32)
                    cv2.imwrite(image_path, result_img)

        end_time = time.time()
        if rank == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                'epoch: {}, avg. loss: {:.5f}, lr: {:.6f}, time: {:.2f}s.\n'.format(
                    epoch, epoch_loss / (num_crops * (iteration + 1)), lr, end_time - start_time,
                )
            )

        if rank == 0 and (epoch % args.save_model_step) == 0:
            model_out_path = os.path.join(weight_path, 'epoch_{}.pth'.format(epoch))
            model = network.module if args.distributed else network
            checkpoint(epoch, model, optimizer, model_out_path)

        if rank == 0 and args.test_flag:
            logger.info('epoch: {}, psnr and ssim test results:'.format(epoch))
            model = network.module if args.distributed else network
            for color in [False, True]:
                for test_cr in [0.01, 0.04, 0.10, 0.25, 0.50]:
                    test_path = os.path.join(result_path2, 'cr_' + str(test_cr))
                    os.makedirs(test_path, exist_ok=True)
                    logger.info('CR: {}.'.format(test_cr))
                    psnr_dict, ssim_dict = test(args, test_cr, color, model, logger, test_path, epoch=epoch)
                    logger.info('psnr: {}.'.format(psnr_dict))
                    logger.info('ssim: {}.'.format(ssim_dict))


if __name__ == '__main__':
    torch.set_float32_matmul_precision('highest')
    args = parse_args()
    args.pretrain_epoch = 0

    local_rank = 0
    rank = 0
    if args.distributed:
        local_rank = int(os.environ['LOCAL_RANK'])
        args.device = torch.device('cuda', local_rank)
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
    elif not torch.cuda.is_available():
        args.device = torch.device('cpu')
    else:
        args.device = torch.device(args.device)

    date_time = time2file_name(str(datetime.datetime.now()))
    result_path1 = weight_path = log_path = result_path2 = None
    if rank == 0:
        result_path1 = os.path.join('results', args.decoder_type, date_time, 'train')
        weight_path = os.path.join('weights', args.decoder_type, date_time)
        log_path = os.path.join('log', args.decoder_type)
        os.makedirs(result_path1, exist_ok=True)
        os.makedirs(weight_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)
        if args.test_flag:
            result_path2 = os.path.join('results', args.decoder_type, date_time, 'test')
            os.makedirs(result_path2, exist_ok=True)

    logger = Logger(log_path) if rank == 0 else None

    if rank == 0:
        logger.info(
            '\n' + 'Date:' + date_time + '\n'
            + 'Solver: {}'.format(args.solver) + '\n'
            + 'Network Architecture: {}: {}, {}-{}-{}'.format(
                args.decoder_type, args.dim, args.enc_blocks, args.mid_blocks, args.dec_blocks,
            ) + '\n'
            + 'Batch Size: {}'.format(args.batch_size) + '\n'
            + 'Learning Rate: {:.6f}'.format(args.lr) + '\n'
            + 'Train Epochs: {}'.format(args.epochs) + '\n'
            + 'Train Sizes: {} ({} crops/iter)'.format(args.train_sizes, args.num_train_crops) + '\n'
            + 'Test or Not: {}'.format(args.test_flag) + '\n'
            + 'Pretrain Model: {}'.format(args.pretrained_model_path)
        )

    seed = random.randint(1, 10000) if rank == 0 else 0
    if args.distributed:
        seed_tensor = torch.tensor(seed, device=args.device)
        dist.broadcast(seed_tensor, src=0)
        seed = int(seed_tensor.item())
    if rank == 0:
        logger.info('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    args.seed = seed

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank if args.distributed else 0)
        torch.cuda.empty_cache()

    network = ProxUnroll(
        solver=args.solver,
        color_channel=args.color_channel,
        dim=args.dim,
        mid_blocks=args.mid_blocks,
        enc_blocks=args.enc_blocks,
        dec_blocks=args.dec_blocks,
    ).to(args.device)

    if args.torchcompile:
        assert hasattr(torch, 'compile'), 'torch.compile() is required for --torchcompile.'
        network = torch.compile(network, backend=args.torchcompile)

    optimizer = optim.Adam(network.parameters(), lr=args.lr)

    if args.pretrained_model_path is not None:
        pretrained_dict = torch.load(args.pretrained_model_path, map_location=args.device)
        args.pretrain_epoch = pretrained_dict.get('pretrain_epoch', 0)
        load_checkpoint(network, pretrained_dict, logger)
    elif rank == 0:
        logger.info('No pretrained model.')

    if args.distributed:
        pretrain_epoch = torch.tensor(args.pretrain_epoch, device=args.device)
        dist.broadcast(pretrain_epoch, src=0)
        args.pretrain_epoch = int(pretrain_epoch.item())

    if args.distributed:
        network = DDP(network, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    train(args, network, optimizer, logger, weight_path, result_path1, result_path2)
