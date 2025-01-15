"""
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

Described in:
    SuperPoint: Self-Supervised Interest Point Detection and Description,
    Daniel DeTone, Tomasz Malisiewicz, Andrew Rabinovich, CVPRW 2018.

Original code: github.com/MagicLeapResearch/SuperPointPretrainedNetwork

Adapted by Philipp Lindenberger (Phil26AT)
"""

import torch
from torch import nn
import cv2
import numpy as np

from gluefactory.models.base_model import BaseModel
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

def extract_sift_features(keypoints_add, image, image_size, feature_dim=256):
    batch_size = image.shape[0]  # B
    device = image.device
    descriptions = []

    sift = cv2.SIFT_create()  # 初始化 SIFT 特征提取器

    for i in range(batch_size):
        keypoints_batch = keypoints_add[i]  # 该 batch 的关键点
        h, w = image_size[i]  # 图像的尺寸 (H, W)
        image_batch = image[i].squeeze(0).cpu().numpy()  # 该 batch 的图像 (H * W), 转换为 numpy 格式

        # 转换为 OpenCV 支持的格式（H * W）
        image_batch = (image_batch * 255).astype('uint8')  # 图像可能在 0-1 范围内，需要放大到 0-255

        # 将 keypoints 转换为 OpenCV 的 KeyPoint 格式
        opencv_keypoints = [cv2.KeyPoint(x=kp[0].item(), y=kp[1].item(), size=5) for kp in keypoints_batch]  # `size` 参数必须提供

        # 提取 SIFT 特征
        _, sift_descriptors = sift.compute(image_batch, opencv_keypoints)
        # print(sift_descriptors)
        # 如果没有描述符，填充零向量
        if sift_descriptors is None:
            sift_descriptors = torch.zeros((len(keypoints_batch), 128)).to(device)
        else:
            sift_descriptors = torch.tensor(sift_descriptors, dtype=torch.float32).to(device)

        # L2 归一化 SIFT 描述符
        sift_descriptors = normalize_sift_descriptors(sift_descriptors)
        # 将 128 维扩展到 256 维
        if sift_descriptors.shape[1] == 128:
            padding = torch.zeros((sift_descriptors.shape[0], feature_dim - 128)).to(device)  # 补充 128 维的零
            sift_descriptors = torch.cat([sift_descriptors, padding], dim=1)  # (N, 256)

        descriptions.append(sift_descriptors)

        # 将描述符转换为 B * N * 256 的张量
    descriptions = torch.stack(descriptions, 0)  # (B, N, 256)

    return descriptions


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


def add_additional_keypoints2(keypoints, scores, image_size, r,remove_borders):
    batch_size = keypoints.shape[0]
    device = keypoints.device
    keypoints_add = []
    scores_add = []

    for i in range(batch_size):
        keypoints_batch = keypoints[i]  # B*N*2, shape for one batch
        w,h = image_size[i]  # image dimensions (height, width)

        # Define border limits
        border_limit = r if remove_borders else 0

        # Create a grid of new keypoints at intervals of r, while respecting borders
        x_coords = torch.arange(border_limit, w - border_limit, r, device=device)
        y_coords = torch.arange(border_limit, h - border_limit, r, device=device)
        new_keypoints = torch.cartesian_prod(x_coords, y_coords)  # Create a grid of (x, y) points

        # Convert existing keypoints to a set of tuples for easy comparison
        existing_keypoints_set = set(map(tuple, keypoints_batch.cpu().numpy()))

        # Filter out new keypoints that already exist and create scores for them
        unique_new_keypoints = []
        for kp in new_keypoints:
            if tuple(kp.cpu().numpy()) not in existing_keypoints_set:
                unique_new_keypoints.append(kp)

        # Convert the list back to a tensor
        unique_new_keypoints = torch.stack(unique_new_keypoints).to(device) if unique_new_keypoints else torch.empty((0, 2), device=device)

        # Create scores for the unique new keypoints
        unique_new_scores = torch.ones(unique_new_keypoints.shape[0], device=device)  # All new scores are 1

        # Add new keypoints and scores for this batch
        keypoints_add.append(unique_new_keypoints)
        scores_add.append(unique_new_scores)

    # Stack all batches into tensors
    keypoints_add = torch.cat(keypoints_add, dim=0).unsqueeze(0) if keypoints_add else torch.empty((0, 2), device=device)  # Shape (1, total_new_points, 2)
    scores_add = torch.cat(scores_add, dim=0).unsqueeze(0) if scores_add else torch.empty((0,), device=device)  # Shape (1, total_new_points)

    # Combine with existing keypoints and scores
    combined_keypoints = torch.cat([keypoints, keypoints_add], dim=1)  # Combine along the keypoints axis
    combined_scores = torch.cat([scores, scores_add], dim=1)  # Combine scores, ensuring dimensions match

    print(combined_scores.shape)  # Check shape of combined scores
    return combined_keypoints, combined_scores


class SuperPoint(BaseModel):
    default_conf = {
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
        "max_num_keypoints":2048,
        "detection_threshold": 0.0,
        "nms_radius": 3,
        "dense_check": False, #检查dense程度
        "dense_sift": False,  # dense且sift
        "see": False,  # dense且sift
    }
    required_data_keys = ["image"]

    checkpoint_url = "https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/weights/superpoint_v1.pth"  # noqa: E501

    def _init(self, conf):
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        if conf.has_detector:
            self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
            self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        if conf.has_descriptor:
            self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
            self.convDb = nn.Conv2d(
                c5, conf.descriptor_dim, kernel_size=1, stride=1, padding=0
            )

        self.load_state_dict(
            torch.hub.load_state_dict_from_url(str(self.checkpoint_url)), strict=False
        )

    def _forward(self, data):
        image = data["image"]
        if image.shape[1] == 3:  # RGB
            scale = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            image = (image * scale).sum(1, keepdim=True)

        # Shared Encoder
        x = self.relu(self.conv1a(image))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        pred = {}
        if self.conf.has_detector:
            # Compute the dense keypoint scores
            cPa = self.relu(self.convPa(x))
            #1 65 HW
            scores = self.convPb(cPa)
            #1 64 HW这里切片减少一维度，若这维度高则代表大概率无特征点，只是消减概率，但每个点仍然有概率
            scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
            b, c, h, w = scores.shape
            #(b, c, h, w) → (b, h, w, c)
            scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
            #(b, h, w, 8, 8) → (b, h, 8, w, 8)
            scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
            pred["keypoint_scores"] = dense_scores = scores
        if self.conf.has_descriptor:
            # Compute the dense descriptors
            cDa = self.relu(self.convDa(x))
            dense_desc = self.convDb(cDa)
            dense_desc = torch.nn.functional.normalize(dense_desc, p=2, dim=1)
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
                descriptions = extract_sift_features(keypoints_add, image, data["image_size"], feature_dim=256)
                # 检查两个张量在 batch 维度和 feature 维度是否匹配
                assert desc.shape[0] == descriptions.shape[0], "Batch size must be the same for both tensors"
                assert desc.shape[2] == descriptions.shape[2], "Feature dimensions must match"
                # 在第二个维度 (N1 和 N2) 上合并
                descriptions = torch.cat([desc, descriptions], dim=1)  # B * (N1 + N2) * 256
                scores = torch.cat([scores, scores_add], dim=1) if scores_add.size(0) > 0 else scores
                keypoints = torch.cat([keypoints, keypoints_add], dim=1) if keypoints_add.size(0) > 0 else keypoints
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

    def loss(self, pred, data):
        raise NotImplementedError

    def metrics(self, pred, data):
        raise NotImplementedError
