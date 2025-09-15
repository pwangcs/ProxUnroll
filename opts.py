import argparse 
import torch 
import numpy as np
import os
from time import time
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--pretrained_model_path', default=None, type=str)
    parser.add_argument("--test_model_path", default=None, type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--decoder_type', default='hqs_proxunroll', type=str) 
    parser.add_argument('--color', default=False, type=bool)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--color_channel', default=1,type=int)
    parser.add_argument('--dim', default=48,type=int)
    parser.add_argument('--mid_blocks', default=2)
    parser.add_argument('--enc_blocks', default=[2,2,2])
    parser.add_argument('--dec_blocks', default=[2,2,2])
    parser.add_argument('--save_model_step', default=1, type=int)
    parser.add_argument('--save_train_image_step', default=100, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--iter_step', default=100, type=int)
    parser.add_argument('--test_flag', default=False, type=bool)
    parser.add_argument('--train_data_path',type=str,default='/home/wangping/datasets/BSDS400')
    parser.add_argument('--test_data_path',type=str,default='/home/wangping/datasets/Set11')
    parser.add_argument('--test_color_data_path',type=str,default='/home/wangping/datasets/CBSD68')
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--torchcompile',nargs='?', type=str, default=None, const='inductor')

    args = parser.parse_args()

    return args
