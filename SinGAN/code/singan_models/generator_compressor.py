from torch import nn
from torch.nn import functional as F
import torch

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from models import Elvira2

class GeneratorWithCompression(nn.Module):
    def __init__(self, img_size_min, num_scale, scale_factor=4/3, n_channels: int = 3, device=torch.device('cpu')):
        super(GeneratorWithCompression, self).__init__()
        self.img_size_min = img_size_min
        self.scale_factor = scale_factor
        self.num_scale = num_scale
        self.n_channels = n_channels
        self.nf = 32
        self.current_scale = 0
        self.compressor = Elvira2(config_file_path="models/cfg_files/elvira_model.json", device=device)
        self.compressor.eval()
        self.size_list = [int(self.img_size_min * scale_factor**i) for i in range(num_scale + 1)]
        print(self.size_list)

        self.sub_generators = nn.ModuleList()

        first_generator = nn.ModuleList()

        first_generator.append(nn.Sequential(nn.Conv2d(n_channels, self.nf, 3, 1, 1),
                                             nn.BatchNorm2d(self.nf),
                                             nn.LeakyReLU(2e-1)))
        for _ in range(3):
            first_generator.append(nn.Sequential(nn.Conv2d(self.nf, self.nf, 3, 1, 1),
                                                 nn.BatchNorm2d(self.nf),
                                                 nn.LeakyReLU(2e-1)))

        first_generator.append(nn.Sequential(nn.Conv2d(self.nf, n_channels, 3, 1, 1),
                                             nn.Tanh()))

        first_generator = nn.Sequential(*first_generator)

        self.sub_generators.append(first_generator)

    def forward(self, z, img=None):
        x_list = []
        x_first = self.sub_generators[0](self.compressor.embed_artifacts_without_resize(z[0]).detach())
        x_list.append(x_first)

        if img is not None:
            x_inter = img
        else:
            x_inter = x_first

        for i in range(1, self.current_scale + 1):
            x_inter = F.interpolate(x_inter, (self.size_List[i], self.size_List[i]), mode='bilinear', align_corners=True)
            x_prev = x_inter
            x_inter = self.compressor.embed_artifacts_without_resize(z[i].detach())
            x_inter = self.sub_generators[i](x_inter) + x_prev
            x_list.append(x_inter)

        return x_list

    def progress(self):
        self.current_scale += 1

        # if self.current_scale % 8 == 0:
        #     self.nf *= 2

        tmp_generator = nn.ModuleList()
        tmp_generator.append(nn.Sequential(nn.Conv2d(self.n_channels, self.nf, 3, 1, 1),
                                           nn.BatchNorm2d(self.nf),
                                           nn.LeakyReLU(2e-1)))

        for _ in range(3):
            tmp_generator.append(nn.Sequential(nn.Conv2d(self.nf, self.nf, 3, 1, 1),
                                               nn.BatchNorm2d(self.nf),
                                               nn.LeakyReLU(2e-1)))

        tmp_generator.append(nn.Sequential(nn.Conv2d(self.nf, self.n_channels, 3, 1, 1),
                                           nn.Tanh()))

        tmp_generator = nn.Sequential(*tmp_generator)

        if self.current_scale % 4 != 0:
            prev_generator = self.sub_generators[-1]

            # Initialize layers via copy
            if self.current_scale >= 1:
                tmp_generator.load_state_dict(prev_generator.state_dict())

        self.sub_generators.append(tmp_generator)
        print("GENERATOR PROGRESSION DONE")
