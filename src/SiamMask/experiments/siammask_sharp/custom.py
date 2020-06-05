from models.siammask_sharp import SiamMask
from models.features import MultiStageFeature
from models.rpn import RPN, DepthCorr
from models.mask import Mask
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils_siammask.load_helper import load_pretrain
from resnet import resnet50
import sys
from os import path
try:
    from torch2trt import torch2trt, TRTModule
except ImportError:
    print("Torch2trt not found")
    pass

class ResDownS(nn.Module):
    def __init__(self, inplane, outplane):
        super(ResDownS, self).__init__()
        self.downsample = nn.Sequential(
                nn.Conv2d(inplane, outplane, kernel_size=1, bias=False),
                nn.BatchNorm2d(outplane))
        self.downsample_15 = self.downsample_31 = self.downsample
    def init_trt(self,fp16_mode,trt_weights_path):
        if not path.exists(trt_weights_path+'/downsample_15_trt.pth'):
            x_ds_15 = torch.ones((1,1024,15,15)).cuda()
            x_ds_31 = torch.ones((1,1024,31,31)).cuda()
            self.downsample_15 = torch2trt(self.downsample,[x_ds_15],fp16_mode=fp16_mode)
            self.downsample_31 = torch2trt(self.downsample,[x_ds_31],fp16_mode=fp16_mode)
            torch.save(self.downsample_15.state_dict(), trt_weights_path+'/downsample_15_trt.pth')
            torch.save(self.downsample_31.state_dict(), trt_weights_path+'/downsample_31_trt.pth')
        else:
            self.downsample_15 = TRTModule()
            self.downsample_15.load_state_dict(torch.load(trt_weights_path+'/downsample_15_trt.pth'))
            self.downsample_31 = TRTModule()
            self.downsample_31.load_state_dict(torch.load(trt_weights_path+'/downsample_31_trt.pth'))

    def forward(self, x):
        if x.shape[-1] == 15:
            x = self.downsample_15(x)
        elif x.shape[-1] == 31:
            x = self.downsample_31(x)
        else:
            x = self.downsample(x)
        if x.size(3) < 20:
            l = 4
            r = -4
            x = x[:, :, l:r, l:r]
        return x


class ResDown(MultiStageFeature):
    def __init__(self, pretrain=False):
        super(ResDown, self).__init__()
        self.features = resnet50(layer3=True, layer4=False)
        self.features_127 = self.features_255 = self.features
        if pretrain:
            load_pretrain(self.features, 'resnet.model')

        self.downsample = ResDownS(1024, 256)

        self.layers = [self.downsample, self.features.layer2, self.features.layer3]
        self.train_nums = [1, 3]
        self.change_point = [0, 0.5]

        self.unfix(0.0)
    
    def init_trt(self,fp16_mode,trt_weights_path):
        if not path.exists(trt_weights_path+'/features_127_trt.pth'):
            x_resnet_127 = torch.ones((1,3,127,127)).cuda()
            x_resnet_255 = torch.ones((1,3,255,255)).cuda()
            self.features_127 = torch2trt(self.features,[x_resnet_127],fp16_mode=fp16_mode)
            self.features_255 = torch2trt(self.features,[x_resnet_255],fp16_mode=fp16_mode)
            torch.save(self.features_127.state_dict(), trt_weights_path+'/features_127_trt.pth')
            torch.save(self.features_255.state_dict(), trt_weights_path+'/features_255_trt.pth')
        else:
            self.features_127 = TRTModule()
            self.features_255 = TRTModule()
            self.features_127.load_state_dict(torch.load(trt_weights_path+'/features_127_trt.pth'))
            self.features_255.load_state_dict(torch.load(trt_weights_path+'/features_255_trt.pth'))

        self.downsample.init_trt(fp16_mode,trt_weights_path)

    def param_groups(self, start_lr, feature_mult=1):
        lr = start_lr * feature_mult

        def _params(module, mult=1):
            params = list(filter(lambda x:x.requires_grad, module.parameters()))
            if len(params):
                return [{'params': params, 'lr': lr * mult}]
            else:
                return []

        groups = []
        groups += _params(self.downsample)
        groups += _params(self.features, 0.1)
        return groups

    def forward(self, x):
        output = self.features_127(x)
        p3 = self.downsample(output[-1])
        return p3

    def forward_all(self, x):
        output = self.features_255(x)
        p3 = self.downsample(output[-1])
        return output, p3


class UP(RPN):
    def __init__(self, anchor_num=5, feature_in=256, feature_out=256):
        super(UP, self).__init__()

        self.anchor_num = anchor_num
        self.feature_in = feature_in
        self.feature_out = feature_out

        self.cls_output = 2 * self.anchor_num
        self.loc_output = 4 * self.anchor_num

        self.cls = DepthCorr(feature_in, feature_out, self.cls_output)
        self.loc = DepthCorr(feature_in, feature_out, self.loc_output)

    def forward(self, z_f, x_f):
        cls = self.cls(z_f, x_f)
        loc = self.loc(z_f, x_f)
        return cls, loc

    def init_trt(self,fp16_mode,trt_weights_path):
        self.cls.init_trt(fp16_mode,trt_weights_path)
        self.loc.init_trt(fp16_mode,trt_weights_path)

class MaskCorr(Mask):
    def __init__(self, oSz=63):
        super(MaskCorr, self).__init__()
        self.oSz = oSz
        self.mask = DepthCorr(256, 256, self.oSz**2)

    def forward(self, z, x):
        return self.mask(z, x)

    def init_trt(self,fp16_mode,trt_weights_path):
        self.mask.init_trt(fp16_mode,trt_weights_path)

class Refine(nn.Module):
    def __init__(self):
        super(Refine, self).__init__()
        self.v0 = nn.Sequential(nn.Conv2d(64, 16, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(16, 4, 3, padding=1),nn.ReLU())

        self.v1 = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(64, 16, 3, padding=1), nn.ReLU())

        self.v2 = nn.Sequential(nn.Conv2d(512, 128, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(128, 32, 3, padding=1), nn.ReLU())

        self.h2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())

        self.h1 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())

        self.h0 = nn.Sequential(nn.Conv2d(4, 4, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(4, 4, 3, padding=1), nn.ReLU())

        self.deconv = nn.ConvTranspose2d(256, 32, 15, 15)

        self.post0 = nn.Conv2d(32, 16, 3, padding=1)
        self.post1 = nn.Conv2d(16, 4, 3, padding=1)
        self.post2 = nn.Conv2d(4, 1, 3, padding=1)
        
        for modules in [self.v0, self.v1, self.v2, self.h2, self.h1, self.h0, self.deconv, self.post0, self.post1, self.post2,]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, f, corr_feature, pos=None, test=False):
        if test:
            p0 = torch.nn.functional.pad(f[0], [16, 16, 16, 16])[:, :, 4*pos[0]:4*pos[0]+61, 4*pos[1]:4*pos[1]+61]
            p1 = torch.nn.functional.pad(f[1], [8, 8, 8, 8])[:, :, 2 * pos[0]:2 * pos[0] + 31, 2 * pos[1]:2 * pos[1] + 31]
            p2 = torch.nn.functional.pad(f[2], [4, 4, 4, 4])[:, :, pos[0]:pos[0] + 15, pos[1]:pos[1] + 15]
        else:
            p0 = F.unfold(f[0], (61, 61), padding=0, stride=4).permute(0, 2, 1).contiguous().view(-1, 64, 61, 61)
            if not (pos is None): p0 = torch.index_select(p0, 0, pos)
            p1 = F.unfold(f[1], (31, 31), padding=0, stride=2).permute(0, 2, 1).contiguous().view(-1, 256, 31, 31)
            if not (pos is None): p1 = torch.index_select(p1, 0, pos)
            p2 = F.unfold(f[2], (15, 15), padding=0, stride=1).permute(0, 2, 1).contiguous().view(-1, 512, 15, 15)
            if not (pos is None): p2 = torch.index_select(p2, 0, pos)

        if not(pos is None):
            p3 = corr_feature[:, :, pos[0], pos[1]].view(-1, 256, 1, 1)
        else:
            p3 = corr_feature.permute(0, 2, 3, 1).contiguous().view(-1, 256, 1, 1)

        out = self.deconv(p3)
        out = self.post0(F.interpolate(self.h2(out) + self.v2(p2), size=(31, 31)))
        out = self.post1(F.interpolate(self.h1(out) + self.v1(p1), size=(61, 61)))
        out = self.post2(F.interpolate(self.h0(out) + self.v0(p0), size=(127, 127)))
        out = out.view(-1, 127*127)
        return out

    def init_trt(self,fp16_mode,trt_weights_path):
        if not path.exists(trt_weights_path+'/v2_trt.pth'):
            x_deconv = torch.ones((1,256,1,1)).cuda()
            x_v2 = torch.ones((1,512,15,15)).cuda()
            x_h2 = torch.ones((1,32,15,15)).cuda()
            x_post0 = torch.ones((1,32,31,31)).cuda()
            x_v1 = torch.ones((1,256,31,31)).cuda()
            x_h1 = torch.ones((1,16,31,31)).cuda()
            x_post1 = torch.ones((1,16,61,61)).cuda()
            x_v0 = torch.ones((1,64,61,61)).cuda()
            x_h0 = torch.ones((1,4,61,61)).cuda()
            x_post2 = torch.ones((1,4,127,127)).cuda()

            # self.deconv = torch2trt(self.deconv,[x_deconv])
            self.v2 = torch2trt(self.v2,[x_v2],fp16_mode=fp16_mode)
            self.h2 = torch2trt(self.h2,[x_h2],fp16_mode=fp16_mode)
            self.post0 = torch2trt(self.post0,[x_post0],fp16_mode=fp16_mode)
            self.v1 = torch2trt(self.v1,[x_v1],fp16_mode=fp16_mode)
            self.h1 = torch2trt(self.h1,[x_h1],fp16_mode=fp16_mode)
            self.post1 = torch2trt(self.post1,[x_post1],fp16_mode=fp16_mode)
            self.v0 = torch2trt(self.v0,[x_v0],fp16_mode=fp16_mode)
            self.h0 = torch2trt(self.h0,[x_h0],fp16_mode=fp16_mode)
            self.post2 = torch2trt(self.post2,[x_post2],fp16_mode=fp16_mode)

            torch.save(self.v2.state_dict(), trt_weights_path+'/v2_trt.pth')
            torch.save(self.h2.state_dict(), trt_weights_path+'/h2_trt.pth')
            torch.save(self.post0.state_dict(), trt_weights_path+'/post0_trt.pth')
            torch.save(self.v1.state_dict(), trt_weights_path+'/v1_trt.pth')
            torch.save(self.h1.state_dict(), trt_weights_path+'/h1_trt.pth')
            torch.save(self.post1.state_dict(), trt_weights_path+'/post1_trt.pth')
            torch.save(self.v0.state_dict(), trt_weights_path+'/v0_trt.pth')
            torch.save(self.h0.state_dict(), trt_weights_path+'/h0_trt.pth')
            torch.save(self.post2.state_dict(), trt_weights_path+'/post2_trt.pth')

        else:
            self.v2 = TRTModule()
            self.h2 = TRTModule()
            self.post0 = TRTModule()
            self.v1 = TRTModule()
            self.h1 = TRTModule()
            self.post1 = TRTModule()
            self.v0 = TRTModule()
            self.h0 = TRTModule()
            self.post2 = TRTModule()

            self.v2.load_state_dict(torch.load(trt_weights_path+'/v2_trt.pth'))
            self.h2.load_state_dict(torch.load(trt_weights_path+'/h2_trt.pth'))
            self.post0.load_state_dict(torch.load(trt_weights_path+'/post0_trt.pth'))
            self.v1.load_state_dict(torch.load(trt_weights_path+'/v1_trt.pth'))
            self.h1.load_state_dict(torch.load(trt_weights_path+'/h1_trt.pth'))
            self.post1.load_state_dict(torch.load(trt_weights_path+'/post1_trt.pth'))
            self.v0.load_state_dict(torch.load(trt_weights_path+'/v0_trt.pth'))
            self.h0.load_state_dict(torch.load(trt_weights_path+'/h0_trt.pth'))
            self.post2.load_state_dict(torch.load(trt_weights_path+'/post2_trt.pth'))

    def param_groups(self, start_lr, feature_mult=1):
        params = filter(lambda x:x.requires_grad, self.parameters())
        params = [{'params': params, 'lr': start_lr * feature_mult}]
        return params


class Custom(SiamMask):
    def __init__(self, pretrain=False, **kwargs):
        super(Custom, self).__init__(**kwargs)
        self.features = ResDown(pretrain=pretrain)
        self.rpn_model = UP(anchor_num=self.anchor_num, feature_in=256, feature_out=256)
        self.mask_model = MaskCorr()
        self.refine_model = Refine()

    def init_trt(self,fp16_mode=False,features=True,rpn=False,mask=False,refine=False,  trt_weights_path='/root/msl_raptor_ws/src/msl_raptor/src/front_end/SiamMask/weights_trt'):
        
        modulename = 'torch2trt'
        if modulename not in sys.modules:
            print('torch2trt not found, trt not used')
            return
        if features:
            self.features.init_trt(fp16_mode,trt_weights_path)
        if rpn:
            self.rpn_model.init_trt(fp16_mode,trt_weights_path)
        if mask:
            self.mask_model.init_trt(fp16_mode,trt_weights_path)
        if refine:
            self.refine_model.init_trt(fp16_mode,trt_weights_path)

    def refine(self, f, pos=None):
        return self.refine_model(f, pos)

    def template(self, template):        
        self.zf = self.features(template)

    def track(self, search):
        search = self.features(search)
        rpn_pred_cls, rpn_pred_loc = self.rpn(self.zf, search)
        return rpn_pred_cls, rpn_pred_loc

    def track_mask(self, search):
        self.feature, self.search = self.features.forward_all(search)
        rpn_pred_cls, rpn_pred_loc = self.rpn(self.zf, self.search)
        self.corr_feature = self.mask_model.mask.forward_corr(self.zf, self.search)
        pred_mask = self.mask_model.mask.head(self.corr_feature)
        return rpn_pred_cls, rpn_pred_loc, pred_mask

    def track_refine(self, pos):
        pred_mask = self.refine_model(self.feature, self.corr_feature, pos=pos, test=True)
        return pred_mask

