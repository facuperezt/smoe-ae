from SinGAN.code.singan_models.discriminator import Discriminator
import os
import torch
import matplotlib.pyplot as plt


class DiscriminatorAsLoss:
    def __init__(self, disc: Discriminator):
        self.D = disc
        self.D.eval()

    def forward(self, x, y):
        loss = 0
        for stage, sub_dis in enumerate(self.D.sub_discriminators):
            self.D.current_scale = stage
            _x = x
            _y = y
            for i, layer in enumerate(sub_dis):
                _x = layer(_x)
                _y = layer(_y)
                _loss = ((y.detach() - x.detach())**2).squeeze()
                loss += _loss.sum().cpu()
        return loss

def get_trained_discriminators(log_dir: str, device: torch.device = torch.device("cpu")):
    discriminator = Discriminator()
    check_load = open(os.path.join(log_dir, "checkpoint.txt"), 'r')
    to_restore = check_load.readlines()[-1].strip()
    load_file = os.path.join(log_dir, to_restore)
    if os.path.isfile(load_file):
        print("=> loading checkpoint '{}'".format(load_file))
        checkpoint = torch.load(load_file, map_location='cpu')
        for _ in range(int(checkpoint['stage'])):
            discriminator.progress()

        discriminator.load_state_dict(checkpoint['D_state_dict'])
        print("=> loaded checkpoint '{}' (stage {})"
                .format(load_file, checkpoint['stage']))
        return DiscriminatorAsLoss(discriminator.to(device))
    else:
        print("=> no checkpoint found at '{}'".format(log_dir))