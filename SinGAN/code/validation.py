from tqdm import trange
from torch.nn import functional as F
import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.utils as vutils
import pickle
import os

# from utils import *


def validateSinGAN(data_loader, networks, stage, args, additional=None):
    # set nets
    D = networks[0]
    G = networks[1]
    # switch to train mode
    D.eval()
    G.eval()
    # summary writer
    # writer = additional[0]
    val_it = iter(data_loader)

    z_rec = additional['z_rec']

    x_in = next(val_it)
    x_in = x_in.to(args.device, non_blocking=True)
    x_org = x_in

    x_in = F.interpolate(x_in, (args.size_List[stage], args.size_List[stage]), mode='bilinear', align_corners=True)
    vutils.save_image(x_in.detach().cpu(), os.path.join(args.res_dir, 'ORG_{}.png'.format(stage)),
                      nrow=1, normalize=True)
    x_in_list = [x_in]
    for xidx in range(1, stage + 1):
        x_tmp = F.interpolate(x_org, (args.size_List[xidx], args.size_List[xidx]), mode='bilinear', align_corners=True)
        x_in_list.append(x_tmp)

    for z_idx in range(len(z_rec)):
        z_rec[z_idx] = z_rec[z_idx].to(args.device, non_blocking=True)

    with torch.no_grad():
        x_rec_list = G(z_rec)

        # calculate rmse for each scale
        rmse_list = [1.0]
        for rmseidx in range(1, stage + 1):
            rmse = torch.sqrt(F.mse_loss(x_rec_List[rmseidx], x_in_List[rmseidx]))
            if args.validation:
                rmse /= 100.0
            rmse_list.append(rmse)
        if len(rmse_list) > 1:
            rmse_List[-1] = 0.0

        vutils.save_image(x_rec_List[-1].detach().cpu(), os.path.join(args.res_dir, 'REC_{}.png'.format(stage)),
                          nrow=1, normalize=True)

        for k in range(0):
            z_list = [rmse_List[z_idx] * torch.randn(args.batch_size, args.n_channels, args.size_List[z_idx],
                                                     args.size_List[z_idx]).to(args.device, non_blocking=True) for z_idx in range(stage + 1)]
            x_fake_list = G(z_list)

            vutils.save_image(x_fake_List[-1].detach().cpu(), os.path.join(args.res_dir, 'GEN_{}_{}.png'.format(stage, k)),
                              nrow=1, normalize=True)



