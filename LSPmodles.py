import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module


class LSPMSLayer(Module):
    def __init__(self, in_planes, S):
        super(LSPMSLayer, self).__init__()

        self.in_planes = in_planes
        self.S = S

        self.GP      = nn.AdaptiveAvgPool2d(self.S)
        self.GP_1X1  = nn.Conv2d(self.in_planes, self.in_planes, 1, 1, bias=False)
        self.GP_Rule = nn.ReLU(inplace=True)
        self.conv     = nn.Conv2d(in_channels=in_planes, out_channels=S*S, kernel_size=1, bias=False)

    def forward(self, x):

        B, C, H, W = x.size()

        x_reshape_T = x.view(B, -1, H*W).permute(0, 2, 1)
        x_reshape   = x.view(B, -1, H*W)
        MM1         = torch.bmm(x_reshape_T, x_reshape)
        MM1         = F.softmax(MM1, dim=-1)

        x_conv = self.conv(x).view(B, -1, H*W)
        MM2    = torch.bmm(x_conv, MM1)

        GP = self.GP(x)
        GP = self.GP_1X1(GP)
        GP = self.GP_Rule(GP)
        GP = GP.view(B, -1, self.S*self.S)

        MM3     = torch.bmm(GP, MM2).view(B, C, H, W)
        results = torch.add(MM3, x)

        return results

class LSPMS(Module):
    def __init__(self, in_planes):
        super(LSPMS, self).__init__()
        self.in_planes = in_planes
        self.LSPMS_1 = LSPMSLayer(self.in_planes, 1)
        self.LSPMS_2 = LSPMSLayer(self.in_planes, 2)
        self.LSPMS_3 = LSPMSLayer(self.in_planes, 3)
        self.LSPMS_6 = LSPMSLayer(self.in_planes, 6)

        self.conv  = nn.Conv2d(5*self.in_planes, self.in_planes, 1, 1, bias=False)

    def forward(self, x):


        LSPMS_1 = self.LSPMS_1(x)
        LSPMS_2 = self.LSPMS_2(x)
        LSPMS_3 = self.LSPMS_3(x)
        LSPMS_6 = self.LSPMS_6(x)
        out = torch.cat([x, LSPMS_1, LSPMS_2, LSPMS_3, LSPMS_6], dim=1)
        out = self.conv(out)
        return out


