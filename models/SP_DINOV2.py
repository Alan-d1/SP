"""latest version of SuperpointNet. Use it!

"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_
from models.unet_parts import *
import numpy as np



class SP_DINOV2(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network with Transformer. """

    def __init__(self, img_size=(240, 320), patch_size=8, d_model=128,dinov2_weights = None):
        super(SP_DINOV2, self).__init__()
        if dinov2_weights is None:
            dinov2_weights = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth", map_location="cpu")
        from .transformer import vit_small
        vit_kwargs = dict(img_size= 518,
            patch_size= 14,
            init_values = 1.0,
            ffn_layer = "mlp",
            block_chunks = 0,
        )
        dinov2_vitl14 = vit_small(**vit_kwargs).eval()
        dinov2_vitl14.load_state_dict(dinov2_weights)
        # Transformer Encoder
        # self.dinov2_vitl14 = nn.ModuleList([dinov2_vitl14])
        self.dinov2_vitl14 = [dinov2_vitl14]

        self.convAdjust = nn.Conv2d(384, 128, kernel_size=1)
        c1, c2, c3, c4, c5, d1, det_h = 64, 64, 128, 128, 256, 256, 65
        self.relu = torch.nn.ReLU(inplace=True)

        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnPa = nn.BatchNorm2d(c5)
        self.convPb = torch.nn.Conv2d(c5, det_h, kernel_size=1, stride=1, padding=0)
        self.bnPb = nn.BatchNorm2d(det_h)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnDa = nn.BatchNorm2d(c5)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)
        self.bnDb = nn.BatchNorm2d(d1)


    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x H x W.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # Transformer Encoder
        B,C,H,W = x.shape
        # x = x.repeat(1, 3, 1, 1)  # 形状将变为 (B, 3, H, W)
        x = F.interpolate(x, size=(672, 672), mode='bilinear', align_corners=False)#BC 672 672
        with torch.no_grad():
            if self.dinov2_vitl14[0].device != x.device:
                self.dinov2_vitl14[0] = self.dinov2_vitl14[0].to(x.device)
            dinov2_features_16 = self.dinov2_vitl14[0].forward_features(x)
            features_16 = dinov2_features_16['x_norm_patchtokens'].permute(0, 2, 1).reshape(B, 384, 48, 48)#BC 48 48
            del dinov2_features_16

        x = F.interpolate(features_16, size=(H // 8, W // 8), mode='bilinear', align_corners=False)#BC，H/8, W/8
        x = self.convAdjust(x)

        # Detector Head.
        cPa = self.relu(self.bnPa(self.convPa(x)))
        semi = self.bnPb(self.convPb(cPa))
        # Descriptor Head.
        cDa = self.relu(self.bnDa(self.convDa(x)))
        desc = self.bnDb(self.convDb(cDa))

        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.

        output = {'semi': semi, 'desc': desc}
        return output

    def process_output(self, sp_processer):
        """
        input:
          N: number of points
        return: -- type: tensorFloat
          pts: tensor [batch, N, 2] (no grad)  (x, y)
          pts_offset: tensor [batch, N, 2] (grad) (x, y)
          pts_desc: tensor [batch, N, 256] (grad)
        """
        from utils.utils import flattenDetection
        # from models.model_utils import pred_soft_argmax, sample_desc_from_points
        output = self.output
        semi = output['semi']
        desc = output['desc']
        # flatten
        heatmap = flattenDetection(semi)  # [batch_size, 1, H, W]
        # nms
        heatmap_nms_batch = sp_processer.heatmap_to_nms(heatmap, tensor=True)
        # extract offsets
        outs = sp_processer.pred_soft_argmax(heatmap_nms_batch, heatmap)
        residual = outs['pred']
        # extract points
        outs = sp_processer.batch_extract_features(desc, heatmap_nms_batch, residual)

        # output.update({'heatmap': heatmap, 'heatmap_nms': heatmap_nms, 'descriptors': descriptors})
        output.update(outs)
        self.output = output
        return output


def get_matches(deses_SP):
    from models.model_wrap import PointTracker
    tracker = PointTracker(max_length=2, nn_thresh=1.2)
    f = lambda x: x.cpu().detach().numpy()
    # tracker = PointTracker(max_length=2, nn_thresh=1.2)
    # print("deses_SP[1]: ", deses_SP[1].shape)
    matching_mask = tracker.nn_match_two_way(f(deses_SP[0]).T, f(deses_SP[1]).T, nn_thresh=1.2)
    return matching_mask

    # print("matching_mask: ", matching_mask.shape)
    # f_mask = lambda pts, maks: pts[]
    # pts_m = []
    # pts_m_res = []
    # for i in range(2):
    #     idx = xs_SP[i][matching_mask[i, :].astype(int), :]
    #     res = reses_SP[i][matching_mask[i, :].astype(int), :]
    #     print("idx: ", idx.shape)
    #     print("res: ", idx.shape)
    #     pts_m.append(idx)
    #     pts_m_res.append(res)
    #     pass

    # pts_m = torch.cat((pts_m[0], pts_m[1]), dim=1)
    # matches_test = toNumpy(pts_m)
    # print("pts_m: ", pts_m.shape)

    # pts_m_res = torch.cat((pts_m_res[0], pts_m_res[1]), dim=1)
    # # pts_m_res = toNumpy(pts_m_res)
    # print("pts_m_res: ", pts_m_res.shape)
    # # print("pts_m_res: ", pts_m_res)

    # pts_idx_res = torch.cat((pts_m, pts_m_res), dim=1)
    # print("pts_idx_res: ", pts_idx_res.shape)


