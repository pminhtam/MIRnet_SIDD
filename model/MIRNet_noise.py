from .MIRNet import MIRNet_DGF
import torch.nn as nn
from guided_filter_pytorch.subnet import UNet
from guided_filter_pytorch.guided_filter import ConvGuidedFilter2
import torch
class MIRNet_noise(nn.Module):
    def __init__(self,n_colors=3,out_channels=0):
        super(MIRNet_noise, self).__init__()
        self.unet = UNet(in_channels=n_colors+out_channels, out_channels = out_channels)
        self.mir_dgf = MIRNet_DGF(n_colors=n_colors,out_channels=out_channels)

    def forward(self, data,x_hr):

        pred_i, pred = self.mir_dgf(data,x_hr)
        noise = x_hr - pred
        inp = torch.cat((x_hr,noise),dim=1)
        pred = self.unet(inp)
        return pred_i, pred