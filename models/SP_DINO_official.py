"""latest version of SuperpointNet. Use it!

"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_
from models.unet_parts import *
import numpy as np



class SP_DINO_official(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network with Transformer. """

    def __init__(self, img_size=(240, 320), patch_size=8, d_model=128,dinov2_weights = None):
        super(SP_DINO_official, self).__init__()
        self.dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

        # self.dinov2_vitl14 = [dinov2_vitl14]
        self.fc = nn.Linear(384, 384 * 30 * 40)
        self.convAdjust = nn.Conv2d(384, 128, kernel_size=1)

        # Detector Head.
        self.convPb = torch.nn.Conv2d(d_model, 65, kernel_size=1, stride=1, padding=0)

        # Descriptor Head.
        self.convDb = torch.nn.Conv2d(d_model, 256, kernel_size=1, stride=1, padding=0)

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
        x = x.repeat(1, 3, 1, 1)  # 形状将变为 (B, 3, H, W)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)#BC224 224
        x = self.dinov2_vits14(x)#B 384
        x = self.fc(x)
        x = self.convAdjust(x)

        # Detector Head
        semi = self.convPb(x)  # [N, 65, H/8, W/8]

        # Descriptor Head
        desc = self.convDb(x)  # [N, 256, H/8, W/8]
        dn = torch.norm(desc, p=2, dim=1, keepdim=True)  # 计算范数
        desc = desc.div(dn)  # L2 归一化

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


