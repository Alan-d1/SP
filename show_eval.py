# If we are on colab: this clones the repo and installs the dependencies
from pathlib import Path

import matplotlib.pyplot as plt

from gluefactory_nonfree.SuperPointNet_gauss2 import SuperPointNet_gauss2 as SuperPoint
# from gluefactory.models.matchers.lightglue_pretrained import LightGlue
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch
import cv2
import time
from lightglue import LightGlue
# 设置不启用梯度计算
torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

# 初始化特征提取器和匹配器
extractor1 = SuperPoint(SuperPoint.default_conf).eval().to(device)
extractor2 = SuperPoint(SuperPoint.default_conf).eval().to(device)
matcher = LightGlue(features="superpoint").eval().to(device)
import torchvision.transforms as transforms

# 定义图像预处理流水线，包括转换为 tensor 和归一化操作
image_transforms = transforms.Compose([
    transforms.ToTensor(),  # 将图像从 [H, W, C] 转为 [C, H, W] 并且像素值归一化到 [0, 1]
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 对于 RGB 图像常用的均值
    #                      std=[0.229, 0.224, 0.225])  # 对于 RGB 图像常用的标准差
])


def process_image(image_path,device):
    """
    读取图像并进行预处理，包括转换为 tensor 和归一化
    :param image_path: 图像文件的路径
    :return: 归一化后的图像 tensor 和图像尺寸 (宽, 高)
    """
    # 使用 OpenCV 读取图像
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"无法找到图像文件: {image_path}")

    # 将图像从 BGR 转换为 RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 应用预处理操作，先将图像转换为 tensor 并进行归一化
    image_tensor = image_transforms(image)  # 归一化并转换为 tensor

    # 获取图像尺寸 (高度, 宽度)
    h, w = image.shape[:2]
    image_size_tensor = torch.tensor([w, h], device=device).unsqueeze(0)  # 变成 [1, 2] 的张量

    # 返回处理后的图像 tensor 和图像尺寸
    return {
        "image": image_tensor.unsqueeze(0).to(device),  # 添加 batch 维度，变成 [1, 3, H, W]
        "image_size": image_size_tensor  # 返回图像的宽和高
    }


def nn_matcher(feats0, feats1):
    """
    使用最近邻匹配两组特征。
    :param feats0: 第一张图像的特征字典，包含 'descriptors' 和 'keypoints'
    :param feats1: 第二张图像的特征字典，包含 'descriptors' 和 'keypoints'
    :return: 匹配结果的张量，形状为 [N, 2]，每行存储一个匹配对的索引 (index_in_feats0, index_in_feats1)
    """
    desc0 = feats0['descriptors']  # 第一张图像的描述符, [N0, D]
    desc1 = feats1['descriptors']  # 第二张图像的描述符, [N1, D]

    # 计算欧氏距离矩阵，形状为 [N0, N1]
    dists = torch.cdist(desc0, desc1, p=2)  # p=2 表示欧氏距离

    # 对每个描述符，在第二张图像中找到最近邻索引
    matches01 = torch.argmin(dists, dim=1)  # [N0]，每个元素是 feats1 中匹配的索引

    # 生成匹配对，形状为 [N0, 2]
    matches = torch.stack([torch.arange(matches01.size(0)), matches01], dim=-1)
    return matches



# 0022/2835868540_572241d9f7_o.jpg 0022/1610927842_7027d5148d_o.jpg
# 0022/3686652357_54392802f0_o.jpg 0022/3716501170_348e26ecdf_o.jpg
# 0015/3885626944_ef7b25d477_o.jpg 0015/2268632857_50bc94ff02_o.jpg

# 替换为你的图像路径
image1_path = Path("data/megadepth1500/images/0015/3885626944_ef7b25d477_o.jpg")
image2_path = Path("data/megadepth1500/images/0015/2268632857_50bc94ff02_o.jpg")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')# 处理图像并存储在 data1 和 data2 中
data1 = process_image(image1_path,device)
data2 = process_image(image2_path,device)
# 提取第一张图像的特征
feats0_raw = extractor1(data1)

# 提取第二张图像的特征
feats1_raw = extractor2(data2)

scores=feats1_raw['keypoint_scores']
max_value = scores.max()  # 获取最大值
min_value = scores.min()  # 获取最小值
print(max_value)
print(min_value)
mean_value = scores.mean()  # 均值
median_value = scores.median()  # 中位数

print(f"均值: {mean_value.item()}, 中位数: {median_value.item()}")
# 进行特征匹配
matches01 = matcher({"image0": feats0_raw, "image1": feats1_raw})

# 移除批次维度
feats0, feats1, matches01 = [rbd(x) for x in [feats0_raw, feats1_raw, matches01]]

# 提取匹配的关键点
kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

# 打印推理时间
from lightglue.utils import load_image, rbd
image0 = load_image(image1_path)
image1 = load_image(image2_path)

print("-------------------------------------------------------------")
print("Inference completed.")

# 可视化匹配结果
axes = viz2d.plot_images([image0, image1])
viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
# 保存结果图像为 1.png
plt.savefig('13.png', bbox_inches='tight', pad_inches=0)

# 显示结果图像
plt.show()
kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
viz2d.plot_images([image0, image1])
viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
# # 保存结果图像为 1.png
plt.savefig('14.png', bbox_inches='tight', pad_inches=0)
#
# # 显示结果图像
plt.show()