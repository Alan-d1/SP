"""latest version of SuperpointNet. Use it!

"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_
from models.unet_parts import *
import numpy as np

def conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=2),
        nn.ReLU(inplace=True)
    )


def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True)
    )


class TransformerEncoder(nn.Module):
    def __init__(self, d_model=128, num_heads=8, num_layers=6, patch_size=8, img_size=(240, 320)):
        super(TransformerEncoder, self).__init__()
        self.patch_size = patch_size
        self.H, self.W = img_size
        assert self.H % patch_size == 0 and self.W % patch_size == 0, "Image dimensions must be divisible by patch size."
        self.num_patches_h = self.H // patch_size
        self.num_patches_w = self.W // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        self.d_model = d_model

        # 线性投影层，将每个patch映射到d_model维度
        self.projection = nn.Linear(patch_size * patch_size, d_model)

        # 位置编码
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model))

        # Transformer Encoder 堆叠层
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x shape: [N, 1, H, W]
        N, C, H, W = x.shape
        assert H == self.H and W == self.W, f"Input image size must be ({self.H}, {self.W})"

        # 切分图片成patch，并flatten每个patch
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # x shape: [N, C, num_patches_h, num_patches_w, patch_size, patch_size]
        x = x.contiguous().view(N, C, self.num_patches, -1)  # [N, C, num_patches, patch_size * patch_size]
        x = x.permute(0, 2, 1, 3).contiguous().view(N, self.num_patches,
                                                    -1)  # [N, num_patches, C * patch_size * patch_size]
        # 由于 C=1，简化为 [N, num_patches, patch_size * patch_size]

        # 线性映射每个patch
        x = self.projection(x)  # [N, num_patches, d_model]

        # 加入位置编码
        x = x + self.position_embedding  # [N, num_patches, d_model]

        # 通过Transformer Encoder
        x = self.transformer(x)  # [N, num_patches, d_model]
        return x


class SP_T(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network with Transformer. """

    def __init__(self, img_size=(240, 320), patch_size=8, d_model=128):
        super(SP_T, self).__init__()

        # Transformer Encoder
        self.encoder = TransformerEncoder(d_model=d_model, patch_size=patch_size, img_size=img_size)

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
        x = self.encoder(x)  # [N, num_patches, d_model]

        # 重塑回特征图的形式，适应后续的卷积
        x = x.view(x.shape[0], self.encoder.d_model, self.encoder.num_patches_h,
                   self.encoder.num_patches_w)  # [N, d_model, H/8, W/8]

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



