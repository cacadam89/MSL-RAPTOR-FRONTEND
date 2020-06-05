# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F
from os import path
try:
    from torch2trt import torch2trt, TRTModule
except ImportError:
    print("Torch2trt not found")
    pass
import torch
class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError

    def template(self, template):
        raise NotImplementedError

    def track(self, search):
        raise NotImplementedError

    def param_groups(self, start_lr, feature_mult=1, key=None):
        if key is None:
            params = filter(lambda x:x.requires_grad, self.parameters())
        else:
            params = [v for k, v in self.named_parameters() if (key in k) and v.requires_grad]
        params = [{'params': params, 'lr': start_lr * feature_mult}]
        return params


def conv2d_dw_group(x, kernel):
    batch, channel = kernel.shape[:2]
    x = x.view(1, batch*channel, x.size(2), x.size(3))  # 1 * (b*c) * k * k
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))  # (b*c) * 1 * H * W
    out = F.conv2d(x, kernel, groups=batch*channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out


class DepthCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3):
        super(DepthCorr, self).__init__()
        # adjust layer for asymmetrical features
        self.conv_kernel = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        self.conv_search = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )

        self.head = nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, out_channels, kernel_size=1)
                )

    def forward_corr(self, kernel, input):
        # print(kernel.shape)
        # print(input.shape)
        # print('\n')
        kernel = self.conv_kernel(kernel)
        input = self.conv_search(input)
        feature = conv2d_dw_group(input, kernel)
        return feature

    def forward(self, kernel, search):
        feature = self.forward_corr(kernel, search)
        out = self.head(feature)
        return out

    def init_trt(self,fp16_mode,trt_weights_path):
        if not path.exists(trt_weights_path+'/conv_kernel_trt.pth'):
            x_kernel = torch.ones((1,256,7,7)).cuda()
            x_input = torch.ones((1,256,31,31)).cuda()
            x_feature = torch.ones((1,256,25,25)).cuda()
            self.conv_kernel = torch2trt(self.conv_kernel,[x_kernel],fp16_mode=fp16_mode)
            self.conv_search = torch2trt(self.conv_search,[x_input],fp16_mode=fp16_mode)
            self.head = torch2trt(self.head,[x_feature],fp16_mode=fp16_mode)
            torch.save(self.conv_kernel.state_dict(), trt_weights_path+'/conv_kernel_trt.pth')
            torch.save(self.conv_search.state_dict(), trt_weights_path+'/conv_search_trt.pth')
            torch.save(self.head.state_dict(), trt_weights_path+'/head_trt.pth')
        else:
            self.conv_kernel = TRTModule()
            self.conv_search = TRTModule()
            self.head = TRTModule()

            self.conv_kernel.load_state_dict(torch.load(trt_weights_path+'/conv_kernel_trt.pth'))
            self.conv_search.load_state_dict(torch.load(trt_weights_path+'/conv_search_trt.pth'))
            self.head.load_state_dict(torch.load(trt_weights_path+'/head_trt.pth'))
