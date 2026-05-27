from torch.utils.data import DataLoader
import torch
import os
import numpy as np
import einops
from opts import parse_args
from model.proxunroll import ProxUnroll
from utils import Logger, load_checkpoint, TestData, compare_psnr
import cv2
from skimage.metrics import structural_similarity as ski_ssim


def test(args, cr, color, network, logger, test_dir, epoch=1):
    network = network.eval()
    test_data = TestData(args.test_color_data_path) if color else TestData(args.test_data_path)
    test_data_loader = DataLoader(test_data, shuffle=False, batch_size=1)

    psnr_dict, ssim_dict = {}, {}
    psnr_list, ssim_list = [], []
    rec_list = []

    for data in test_data_loader:
        data = data[0]
        gt = data.float().numpy()

        if gt.shape[0] > gt.shape[1]:
            inp = cv2.rotate(gt[:, :, 0], cv2.ROTATE_90_CLOCKWISE)
            fliped = True
        else:
            inp = gt[:, :, 0]
            fliped = False

        with torch.no_grad():
            outs, _, _ = network(torch.from_numpy(inp).unsqueeze(0).to(args.device), cr)

        out = outs[-1].squeeze(0).clamp(0, 1).cpu().numpy()
        psnr = compare_psnr(inp * 255, out * 255)
        ssim = ski_ssim(inp * 255, out * 255, data_range=255)
        psnr_list.append(np.round(psnr, 4))
        ssim_list.append(np.round(ssim, 4))

        if fliped:
            out = einops.rearrange(out, 'a b -> b a')
        if color:
            out = np.concatenate((np.expand_dims(out, 2), gt[:, :, 1:]), axis=-1)

        rec_list.append(out)

    for i, name in enumerate(test_data.data_list):
        _name, _ = name.split('.')
        psnr_dict[_name] = psnr_list[i]
        ssim_dict[_name] = ssim_list[i]
        image_name = os.path.join(test_dir, _name + '_' + str(psnr_list[i]) + '_' + str(ssim_list[i]) + '.png')
        result_img = cv2.cvtColor(rec_list[i], cv2.COLOR_YCrCb2BGR) if color else rec_list[i]
        cv2.imwrite(image_name, (result_img * 255).astype(np.float32))

    if logger is not None:
        logger.info('psnr_mean: {:.4f}.'.format(np.mean(psnr_list)))
        logger.info('ssim_mean: {:.4f}.'.format(np.mean(ssim_list)))
    return psnr_dict, ssim_dict


if __name__ == '__main__':
    args = parse_args()
    if args.test_model_path is None:
        args.test_model_path = os.path.join('weight', args.decoder_type + '.pth')

    if not torch.cuda.is_available():
        args.device = torch.device('cpu')
    else:
        args.device = torch.device(args.device)

    network = ProxUnroll(
        solver=args.solver,
        color_channel=args.color_channel,
        dim=args.dim,
        mid_blocks=args.mid_blocks,
        enc_blocks=args.enc_blocks,
        dec_blocks=args.dec_blocks,
    ).to(args.device)

    log_dir = os.path.join('test_results', args.solver, 'log')
    os.makedirs(log_dir, exist_ok=True)
    logger = Logger(log_dir)

    pretrained_dict = torch.load(args.test_model_path, map_location=args.device)
    load_checkpoint(network, pretrained_dict, logger)

    for color in [False, True]:
        logger.info('Dataset: {} (solver={}).'.format('CBSD68' if color else 'Set11', args.solver))
        for cr in [0.01, 0.04, 0.10, 0.25, 0.50]:
            mode = 'color' if color else 'gray'
            test_path = os.path.join('test_results', args.solver, mode, 'cr_' + str(cr))
            os.makedirs(test_path, exist_ok=True)
            logger.info('CR: {}.'.format(cr))
            psnr_dict, ssim_dict = test(args, cr, color, network, logger, test_path)
            logger.info('psnr: {}.'.format(psnr_dict))
            logger.info('ssim: {}.'.format(ssim_dict))
