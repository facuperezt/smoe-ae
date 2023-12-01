import json
import torch
from components import TorchSMoE_SMoE as SMoE, OffsetBlock, GlobalMeanOptimizer, TorchSMoE_AE as AE, Img2Block, Block2Img, PositionalEncodingPermute1D as PositionalEncoding, MixedLossFunction



class AserejePipeline(torch.nn.Module):
    def __init__(self, config_file_path: str, device: torch.device = torch.device('cpu')):
        super().__init__()
        self.cfg = json.load(config_file_path)
        self.img2block = Img2Block(**self.cfg['img2block']).to(device)
        self.ae = AE(**self.cfg['ae']).to(device)
        self.positional_encoding = PositionalEncoding(**self.cfg['positional_encoding']).to(device)
        self.global_mean_optimizer = GlobalMeanOptimizer(**self.cfg['global_mean_optimizer']).to(device)
        self.smoe = SMoE(**self.cfg['smoe']).to(device)
        self.block2img = Block2Img(**self.cfg['block2img']).to(device)
        self.loss_function = MixedLossFunction(**self.cfg['loss_function']).to(device)

    def forward(self, x):
        x_blocked = self.img2block(x)
        x_smoe, x_comb = self.ae(x_blocked)
        x_comb = self.positional_encoding(x_comb)
        x = self.global_mean_optimizer(x_smoe, x_comb) + x_smoe
        x = self.smoe(x)
        x = self.block2img(x)

        self.x_blocked, self.x_smoe = x_blocked, x_smoe

        return x, x_smoe
    
    def loss(self, x, y):
        return self.loss_function(x, y) + self.loss_function(self.x_smoe, self.x_blocked)