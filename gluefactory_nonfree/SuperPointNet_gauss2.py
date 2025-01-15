"""latest version of SuperpointNet. Use it!

"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_
from torch.utils.checkpoint import checkpoint

from models.unet_parts import *
import numpy as np
from gluefactory.models.utils.misc import pad_and_stack


def simple_nms(scores, radius):
    """Perform non maximum suppression on the heatmap using max-pooling.
    This method does not suppress contiguous points that have the same score.
    Args:
        scores: the score heatmap of size `(B, H, W)`.
        radius: an integer scalar, the radius of the NMS window.
    """

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=radius * 2 + 1, stride=1, padding=radius
        )
    #看是否是区域内的最大值5*5
    zeros = torch.zeros_like(scores)#1 HW
    max_mask = scores == max_pool(scores)#区域最大值的mask
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0#筛选最大值的r领域
        supp_scores = torch.where(supp_mask, zeros, scores)#最大值r领域赋值0，其他保留
        new_max_mask = supp_scores == max_pool(supp_scores)#找到其余领域的最大值
        max_mask = max_mask | (new_max_mask & (~supp_mask))#合并原最大值和其余领域最大值
    return torch.where(max_mask, scores, zeros)#返回最大值下的score


def top_k_keypoints(keypoints, scores, k):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0, sorted=True)
    return keypoints[indices], scores


def sample_k_keypoints(keypoints, scores, k):
    if k >= len(keypoints):
        return keypoints, scores
    indices = torch.multinomial(scores, k, replacement=False)
    return keypoints[indices], scores[indices]


def soft_argmax_refinement(keypoints, scores, radius: int):
    width = 2 * radius + 1
    sum_ = torch.nn.functional.avg_pool2d(
        scores[:, None], width, 1, radius, divisor_override=1
    )
    ar = torch.arange(-radius, radius + 1).to(scores)
    kernel_x = ar[None].expand(width, -1)[None, None]
    dx = torch.nn.functional.conv2d(scores[:, None], kernel_x, padding=radius)
    dy = torch.nn.functional.conv2d(
        scores[:, None], kernel_x.transpose(2, 3), padding=radius
    )
    dydx = torch.stack([dy[:, 0], dx[:, 0]], -1) / sum_[:, 0, :, :, None]
    refined_keypoints = []
    for i, kpts in enumerate(keypoints):
        delta = dydx[i][tuple(kpts.t())]
        refined_keypoints.append(kpts.float() + delta)
    return refined_keypoints


# Legacy (broken) sampling of the descriptors
def sample_descriptors(keypoints, descriptors, s):
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5#分子是+0.5得到中心像素，-4得到第一个投影点。分母是-1得到像素+0.5得到中心
    keypoints /= torch.tensor(
        [(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)],
    ).to(
        keypoints
    )[None]
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    args = {"align_corners": True} if torch.__version__ >= "1.3" else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", **args
    )#维度为(N,C,Hin,Win) 的input，维度为(N,Hout,Wout,2) 的grid，则该函数output的维度为(N,C,Hout,Wout)。
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1
    )
    return descriptors

# The original keypoint sampling is incorrect. We patch it here but
# keep the original one above for legacy.
def sample_descriptors_fix_sampling(keypoints, descriptors, s: int = 8):
    """Interpolate descriptors at keypoint locations"""
    b, c, h, w = descriptors.shape
    keypoints = keypoints / (keypoints.new_tensor([w, h]) * s)
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", align_corners=False
    )
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1
    )
    return descriptors


def normalize_sift_descriptors(descriptors):
    # 对 SIFT 描述符进行 L2 归一化
    norms = torch.norm(descriptors, p=2, dim=1, keepdim=True)
    normalized_descriptors = descriptors / norms  # L2 归一化
    return normalized_descriptors
import cv2
import torch

def add_additional_keypoints(keypoints, scores, image_size, r, remove_borders):
    batch_size = keypoints.shape[0]
    device = keypoints.device
    keypoints_add = []
    scores_add = []

    for i in range(batch_size):
        keypoints_batch = keypoints[i]  # B*N*2, shape for one batch
        w, h = image_size[i]  # image dimensions (height, width)

        new_keypoints = []
        new_scores = []

        # Apply border limits to avoid generating keypoints near the edges
        border_limit = remove_borders if remove_borders else 0

        # Iterate through the image grid with step size r, excluding borders
        for x in range(border_limit, w - border_limit, r):
            for y in range(border_limit, h - border_limit, r):
                if new_keypoints:
                    combined_keypoints = torch.cat(
                        [keypoints_batch, torch.tensor(new_keypoints).float().to(device)], dim=0
                    )
                else:
                    combined_keypoints = keypoints_batch

                # Check if there's a keypoint within r distance in x, y direction
                dist = torch.sqrt((combined_keypoints[:, 0] - x) ** 2 + (combined_keypoints[:, 1] - y) ** 2)
                if torch.all(dist >= r):  # No keypoints within the radius
                    new_keypoints.append([x, y])
                    new_scores.append(1.0)  # Score is 1 for all new points

        # Add new keypoints and scores for this batch
        if new_keypoints:
            keypoints_add.append(torch.tensor(new_keypoints).float().to(device))  # Move to the correct device
            scores_add.append(torch.tensor(new_scores).float().to(device))  # Move to the correct device
        else:
            keypoints_add.append(torch.empty((0, 2), device=device))  # Ensure empty tensor with shape [0, 2]
            scores_add.append(torch.empty((0,), device=device))  # Ensure empty tensor with shape [0]

    # Convert to tensor
    keypoints_add = torch.stack(keypoints_add, 0) if keypoints_add else torch.empty((batch_size, 0, 2), device=device)
    scores_add = torch.stack(scores_add, 0) if scores_add else torch.empty((batch_size, 0), device=device)

    print(scores_add.shape)
    return keypoints_add, scores_add
from gluefactory.models.base_model import BaseModel

# from models.SubpixelNet import SubpixelNet
import omegaconf
from omegaconf import OmegaConf
class SuperPointNet_gauss2(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """
    default_conf = {
        "name": None,
        "trainable": True,  # if false: do not optimize this model parameters
        "freeze_batch_normalization": False,  # use test-time statistics
        "timeit": False,  # time forward pass
        "has_detector": True,
        "has_descriptor": True,
        "descriptor_dim": 256,
        # Inference
        "sparse_outputs": True,
        "dense_outputs": False,
        "nms_radius": 4,
        "refinement_radius": 0,
        "detection_threshold": 0.005,
        "max_num_keypoints": -1,
        "max_num_keypoints_val": None,
        "force_num_keypoints": False,
        "randomize_keypoints_training": False,
        "remove_borders": 4,
        "legacy_sampling": True,  # True to use the old broken sampling
        "max_num_keypoints": 2048,
        "nms_radius": 3,
        "dense_check": False,  # 检查dense程度
        "dense_sift": False,  # dense且sift
        "see": False,  # dense且sift
    }
    required_data_keys = ["image"]
    def __init__(self, conf):
        super(SuperPointNet_gauss2, self).__init__()
        import os
        # print("Current working directory:", os.getcwd())

        # fixme: backward compatibility
        if "pad" in conf and "pad" not in self.default_conf:  # backward compat.
            with omegaconf.read_write(conf):
                with omegaconf.open_dict(conf):
                    conf["interpolation"] = {"pad": conf.pop("pad")}

        if isinstance(conf, dict):
            conf = OmegaConf.create(conf)
        self.conf = conf = OmegaConf.merge(self.default_conf, conf)
        OmegaConf.set_readonly(conf, True)
        OmegaConf.set_struct(conf, True)

        checkpoint = 'pretrained/desc+dis/superPointNet_20000_checkpoint.pth.tar'

        # self.load_state_dict(torch.load(checkpoint)["model_state_dict"])

        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        det_h = 65
        self.trans = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)
        self.inc = inconv(1, c1)
        self.down1 = down(c1, c2)
        self.down2 = down(c2, c3)
        self.down3 = down(c3, c4)

        self.relu = torch.nn.ReLU(inplace=True)
        # self.outc = outconv(64, n_classes)
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
        self.output = None
        self.load_state_dict(torch.load(checkpoint)["model_state_dict"])
        # self.load_state_dict(torch.load(checkpoint, weights_only=True)["model_state_dict"])
        self.are_weights_initialized=True

    def forward(self, data):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x patch_size x patch_size.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # Let's stick to this version: first BN, then relu
        image=data["image"]
        x = self.trans(image)
        if image.shape[1] == 3:  # RGB
            scale = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            x = (image * scale).sum(1, keepdim=True)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        pred = {}
        # Detector Head.
        cPa = self.relu(self.bnPa(self.convPa(x4)))
        semi = self.bnPb(self.convPb(cPa))
        scores = torch.nn.functional.softmax(semi, 1)[:, :-1]
        b, c, h, w = scores.shape
        # (b, c, h, w) → (b, h, w, c)
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        # (b, h, w, 8, 8) → (b, h, 8, w, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
        pred["keypoint_scores"] = dense_scores = scores

        # Descriptor Head.
        cDa = self.relu(self.bnDa(self.convDa(x4)))
        desc = self.bnDb(self.convDb(cDa))

        dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
        dense_desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
        pred["descriptors"] = dense_desc


        if self.conf.sparse_outputs:
            assert self.conf.has_detector and self.conf.has_descriptor
            #1HW
            scores = simple_nms(scores, self.conf.nms_radius)

            # Discard keypoints near the image borders
            if self.conf.remove_borders:
                scores[:, : self.conf.remove_borders] = -1
                scores[:, :, : self.conf.remove_borders] = -1
                if "image_size" in data:
                    for i in range(scores.shape[0]):
                        w, h = data["image_size"][i]
                        scores[i, int(h.item()) - self.conf.remove_borders :] = -1
                        scores[i, :, int(w.item()) - self.conf.remove_borders :] = -1
                else:
                    scores[:, -self.conf.remove_borders :] = -1
                    scores[:, :, -self.conf.remove_borders :] = -1

            # Extract keypoints 三元组和score一元
            best_kp = torch.where(scores > self.conf.detection_threshold)
            scores = scores[best_kp]

            # Separate into batches B*N*2
            keypoints = [
                torch.stack(best_kp[1:3], dim=-1)[best_kp[0] == i] for i in range(b)
            ]
            scores = [scores[best_kp[0] == i] for i in range(b)]#B*N

            # Keep the k keypoints with highest score
            max_kps = self.conf.max_num_keypoints

            # for val we allow different
            if not self.training and self.conf.max_num_keypoints_val is not None:
                max_kps = self.conf.max_num_keypoints_val

            # Keep the k keypoints with highest score B*N*2/B*N
            if max_kps > 0:
                if self.conf.randomize_keypoints_training and self.training:
                    # instead of selecting top-k, sample k by score weights
                    keypoints, scores = list(
                        zip(
                            *[
                                sample_k_keypoints(k, s, max_kps)
                                for k, s in zip(keypoints, scores)
                            ]
                        )
                    )
                else:
                    keypoints, scores = list(
                        zip(
                            *[
                                top_k_keypoints(k, s, max_kps)
                                for k, s in zip(keypoints, scores)
                            ]
                        )
                    )
                keypoints, scores = list(keypoints), list(scores)

            if self.conf["refinement_radius"] > 0:
                keypoints = soft_argmax_refinement(
                    keypoints, dense_scores, self.conf["refinement_radius"]
                )

            # Convert (h, w) to (x, y)翻转 for batch。BN2
            keypoints = [torch.flip(k, [1]).float() for k in keypoints]

            if self.conf.force_num_keypoints:
                keypoints = pad_and_stack(
                    keypoints,
                    max_kps,
                    -2,
                    mode="random_c",
                    bounds=(
                        0,
                        data.get("image_size", torch.tensor(image.shape[-2:]))
                        .min()
                        .item(),
                    ),
                )
                scores = pad_and_stack(scores, max_kps, -1, mode="zeros")
            else:
                keypoints = torch.stack(keypoints, 0)#BN2
                scores = torch.stack(scores, 0)#BN

            #点搞密集点
            if self.conf.dense_check:
                keypoints_add, scores_add=add_additional_keypoints(keypoints,scores, data["image_size"],32,self.conf.remove_borders)
                if ~self.conf.dense_sift:
                    keypoints = torch.cat([keypoints, keypoints_add], dim=1) if keypoints_add.size(0) > 0 else keypoints
                    scores = torch.cat([scores, scores_add], dim=1) if scores_add.size(0) > 0 else scores

            # Extract descriptors
            if (len(keypoints) == 1) or self.conf.force_num_keypoints:
                # Batch sampling of the descriptors
                if self.conf.legacy_sampling:
                    desc = sample_descriptors(keypoints, dense_desc, 8)
                else:
                    desc = sample_descriptors_fix_sampling(keypoints, dense_desc, 8)
            else:
                if self.conf.legacy_sampling:
                    desc = [
                        sample_descriptors(k[None], d[None], 8)[0]#K【None】加维度1*N
                        for k, d in zip(keypoints, dense_desc)
                    ]
                else:
                    desc = [
                        sample_descriptors_fix_sampling(k[None], d[None], 8)[0]
                        for k, d in zip(keypoints, dense_desc)
                    ]

            # 如果打算sift做新特征
            if self.conf.dense_sift:
                desc = desc.transpose(-1, -2)
                # descriptions = extract_sift_features(keypoints_add, image, data["image_size"], feature_dim=256)
                # # 检查两个张量在 batch 维度和 feature 维度是否匹配
                # assert desc.shape[0] == descriptions.shape[0], "Batch size must be the same for both tensors"
                # assert desc.shape[2] == descriptions.shape[2], "Feature dimensions must match"
                # # 在第二个维度 (N1 和 N2) 上合并
                # descriptions = torch.cat([desc, descriptions], dim=1)  # B * (N1 + N2) * 256
                # scores = torch.cat([scores, scores_add], dim=1) if scores_add.size(0) > 0 else scores
                # keypoints = torch.cat([keypoints, keypoints_add], dim=1) if keypoints_add.size(0) > 0 else keypoints
            else:
                descriptions = desc.transpose(-1, -2)

            #表示像素中心
            pred = {
                "keypoints": keypoints + 0.5,
                "keypoint_scores": scores,
                "descriptors": descriptions,
            }

            if self.conf.dense_outputs:
                pred["dense_descriptors"] = dense_desc

        return pred
