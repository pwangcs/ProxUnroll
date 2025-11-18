import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
##  HQS-ProUnroll Training
from model.hqs_proxunroll import ProxUnroll
##  ADMM-ProUnroll Training
# from model.admm_proxunroll import ProxUnroll
import torch.optim as optim
import os
import cv2
import scipy.io as scio
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from opts import parse_args
from test_pnp_hqs1 import test
import time
import einops
import random
import datetime
from utils import load_checkpoint, checkpoint, TrainData, Logger, time2file_name, compare_psnr


def train(args, network, optimizer, scheduler, logger, weight_path, result_path1, result_path2=None):
    criterion  = nn.MSELoss()
    criterion = criterion.to(args.device)
    rank = 0
    if args.distributed:
        rank = dist.get_rank()
    dataset = TrainData(args.train_data_path)
    dist_sampler = None

    if args.distributed:
        dist_sampler = DistributedSampler(dataset, shuffle=True, drop_last=True, seed=args.seed)
        train_data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size,shuffle=False, num_workers=args.num_workers,
                            drop_last=True, pin_memory=True, sampler=dist_sampler)
    else:
        train_data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    cr = [0.01, 0.04, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    for epoch in range(args.pretrain_epoch + 1, args.pretrain_epoch + args.epochs + 1):
        epoch_loss = 0
        network = network.train()
        start_time = time.time()
        for iteration, data in enumerate(train_data_loader):
            idx = iteration % len(cr)
            for gt in data:
                b, h, w = gt.shape
                gt = gt.float().to(args.device)
                optimizer.zero_grad()
                outputs, prox_outputs, all_out = network(gt,cr[idx])  
                loss = 0.01*torch.sqrt(criterion(outputs[0], prox_outputs[0])) \
                            + 0.01*torch.sqrt(criterion(outputs[1], prox_outputs[1])) \
                            + 0.01*torch.sqrt(criterion(outputs[2], prox_outputs[2])) \
                            + 0.01*torch.sqrt(criterion(outputs[3], prox_outputs[3])) \
                            + 0.01*torch.sqrt(criterion(outputs[4], prox_outputs[4])) \
                            + 0.95*torch.sqrt(criterion(outputs[5], prox_outputs[5])) 

                epoch_loss += loss.item()

                loss.backward()
                optimizer.step()

                if rank==0 and (iteration % args.iter_step) == 0:
                    lr = optimizer.state_dict()['param_groups'][0]['lr']
                    logger.info('epoch: {:<3d}, iter: {:<4d}, size: [{}, {}, {}], cr: {:.2f}, loss: {:.4f}, lr: {:.6f}.'.format(epoch, iteration, b, h, w, cr[idx], loss.item(), lr))

                if rank==0 and (iteration % args.iter_step) == 0:
                    image_path = './'+ result_path1+ '/'+'epoch_{}_iter_{}_cr_{}_reso_{}_{}.png'.format(epoch, iteration, cr[idx], h, w)
                    result_img = einops.rearrange(all_out[0].detach(), 'm n h w -> (m h) (n w)')
                    result_img = result_img.cpu().numpy()*255
                    result_img = result_img.astype(np.float32)
                    cv2.imwrite(image_path,result_img)

        end_time = time.time()
        if rank==0:
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            logger.info('epoch: {}, avg. loss: {:.5f}, lr: {:.6f}, time: {:.2f}s.\n'.format(epoch, epoch_loss/(2*(iteration+1)), lr, end_time-start_time))

        if rank==0 and (epoch % args.save_model_step) == 0:
            model_out_path = './' + weight_path + '/' + 'epoch_{}.pth'.format(epoch)
            if args.distributed:
                checkpoint(epoch, network.module, optimizer, model_out_path)
            else:
                checkpoint(epoch, network, optimizer, model_out_path)

        if rank==0 and args.test_flag:
            logger.info('epoch: {}, psnr and ssim test results:'.format(epoch))
            for color in [False, True]:
                for cr in [0.01,0.04,0.10,0.25,0.50]:  
                    test_path = result_path2 + "/" + "cr_" + str(cr)
                    if not os.path.exists(test_path):
                        os.makedirs(test_path,exist_ok=True)
                    logger.info("CR: {}.".format(cr))
                    if args.distributed:
                        psnr_dict, ssim_dict = test(args, cr, color, network.module, logger, test_path, epoch=epoch)
                    else:
                        psnr_dict, ssim_dict = test(args, cr, color, network, logger, test_path, epoch=epoch)
                    logger.info("psnr: {}.".format(psnr_dict))
                    logger.info("ssim: {}.".format(ssim_dict))


if __name__ == '__main__':
    torch.set_float32_matmul_precision('highest')
    args = parse_args()
    rank = 0
    args.pretrain_epoch = 0
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)
    if rank ==0:
        result_path1 = 'results' + '/' + '{}'.format(args.decoder_type) + '/' + date_time + '/train'
        weight_path = 'weights' + '/' + '{}'.format(args.decoder_type) + '/' + date_time
        log_path = 'log' + '/' + '{}'.format(args.decoder_type)
        if not os.path.exists(result_path1):
            os.makedirs(result_path1,exist_ok=True)
        if not os.path.exists(weight_path):
            os.makedirs(weight_path,exist_ok=True)
        if not os.path.exists(log_path):
            os.makedirs(log_path,exist_ok=True)
        if args.test_flag:
            result_path2 = 'results' + '/' + '{}'.format(args.decoder_type) + '/' + date_time + '/test'
            if not os.path.exists(result_path2):
                os.makedirs(result_path2,exist_ok=True)
        else:
            result_path2 = None
    
    logger = Logger(log_path)
    
    if args.distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        args.device = torch.device('cuda',local_rank)
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()

    if rank==0:
        logger.info('\n'+'Date:' + date_time + '\n' +
                'Network Architecture: {}: {}, {}-{}-{}'.format(args.decoder_type, args.dim, args.enc_blocks,args.mid_blocks,args.dec_blocks) + '\n' +
                'Batch Size: {}'.format(args.batch_size) + '\n' +
                'Learning Rate: {:.6f}'.format(args.lr) + '\n' +
                'Train Epochs: {}'.format(args.epochs) + '\n' +
                'Test or Not: {}'.format(args.test_flag) + '\n' +
                'Pretrain Model: {}'.format(args.pretrained_model_path)
                ) 

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = random.randint(1, 10000)
    # seed = 0
    logger.info('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    args.seed = seed

    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    network = ProxUnroll(color_channel=args.color_channel,
                dim=args.dim,
                mid_blocks=args.mid_blocks,
                enc_blocks=args.enc_blocks,
                dec_blocks=args.dec_blocks).to(args.device)
    has_compile = hasattr(torch, 'compile')
    if args.torchcompile:
        assert has_compile, 'A version of torch w/ torch.compile() is required for --compile, possibly a nightly.'
        network = torch.compile(network, backend=args.torchcompile)
    optimizer = optim.Adam(network.parameters(), lr=args.lr)
    scheduler = None
    
    if rank==0:
        if args.pretrained_model_path is not None:
            pretrained_dict = torch.load(args.pretrained_model_path)
            args.pretrain_epoch = pretrained_dict['pretrain_epoch'] if 'pretrain_epoch' in pretrained_dict.keys() else 0
            load_checkpoint(network, pretrained_dict, logger)
        else:
            logger.info('No pretrained model.')

    if args.distributed:
        network = DDP(network, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)

    train(args, network, optimizer, scheduler, logger, weight_path, result_path1, result_path2)
