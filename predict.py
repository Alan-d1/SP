import torch
from gluefactory.utils.tensor import batch_to_device
import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms


def pca_to_rgb(tensor_image, r2, n_components=3):
    """
    使用 PCA 降维并将两个特征 Tensor 映射为 RGB 图像。
    支持不同大小的特征图。

    :param tensor_image: 形状为 (B, C, H1, W1) 的 Tensor（第一个特征图）
    :param r2: 形状为 (B, C, H2, W2) 的 Tensor（第二个特征图）
    :param n_components: PCA 降到的维度（通常为 3，用于 RGB）
    :return: 两个 RGB 图像
    """
    B, C1, H1, W1 = tensor_image.shape
    B, C2, H2, W2 = r2.shape

    # 假设我们处理的是批次中的第一个图像
    tensor_image = tensor_image[0]  # 选取一个样本
    r2 = r2[0]

    # 展平为 (H * W, C)
    flattened_image = tensor_image.view(C1, -1).T  # 第一张图像
    flattened_r2 = r2.view(C2, -1).T  # 第二张图像

    # 在第一个 Tensor 上拟合 PCA
    pca = PCA(n_components=n_components)
    reduced_image = pca.fit_transform(flattened_image.cpu().numpy())  # 第一张降维结果
    reduced_r2 = pca.transform(flattened_r2.cpu().numpy())  # 第二张使用相同投影矩阵降维

    # 还原为原始图像大小
    reduced_image = torch.tensor(reduced_image).view(H1, W1, n_components)
    reduced_r2 = torch.tensor(reduced_r2).view(H2, W2, n_components)

    # 将数据映射到 [0, 255]
    def normalize_to_rgb(tensor):
        tensor = tensor - tensor.min()
        tensor = tensor / tensor.max()  # 归一化到 [0, 1]
        tensor = (tensor * 255).byte()  # 映射到 [0, 255]
        return tensor

    reduced_image = normalize_to_rgb(reduced_image)
    reduced_r2 = normalize_to_rgb(reduced_r2)

    # 转换为 RGB 图像
    rgb_image_1 = reduced_image.cpu().numpy().astype(np.uint8)
    rgb_image_2 = reduced_r2.cpu().numpy().astype(np.uint8)

    return rgb_image_1, rgb_image_2

import os
import cv2

def tensor_to_cv2_image(tensor_image):
    # 检查形状
    if tensor_image.ndim != 4 or tensor_image.shape[1] != 3:
        raise ValueError("Input tensor must have shape [1, 3, H, W]")

    # 移除批次维度，将形状从 [1, 3, H, W] 转为 [3, H, W]
    tensor_image = tensor_image.squeeze(0)

    # 将像素值范围从 [0, 1] 转换为 [0, 255]
    tensor_image = tensor_image * 255.0
    tensor_image = tensor_image.byte()  # 转为 uint8

    # 转换为 NumPy 数组，形状从 [3, H, W] 转为 [H, W, 3]
    np_image = tensor_image.permute(1, 2, 0).cpu().numpy()

    # OpenCV 使用的是 BGR 格式，转换为 BGR
    bgr_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    return bgr_image

def img_pair_visual(
    image1,
    image2,
    matched_keypoints,
    matched_warped_keypoints,
    good_matches_mask=None,
):
    # 转换为 OpenCV 格式
    image1 = tensor_to_cv2_image(image1)
    image2 = tensor_to_cv2_image(image2)
    img_width = image1.shape[1]

    # 确保两张图片高度一致
    height = max(image1.shape[0], image2.shape[0])
    if image1.shape[0] < height:
        image1 = np.pad(image1, ((0, height - image1.shape[0]), (0, 0), (0, 0)))
    if image2.shape[0] < height:
        image2 = np.pad(image2, ((0, height - image2.shape[0]), (0, 0), (0, 0)))

    # 水平拼接图片
    image_pair = np.hstack((image1, image2))

    # # 确保 keypoint 坐标为 (x, y)
    # matched_keypoints = matched_keypoints[:, [1, 0]]  # 转换为 (col, row)
    # matched_warped_keypoints = matched_warped_keypoints[:, [1, 0]]  # 转换为 (col, row)

    # 将坐标转换为整数
    matched_keypoints = matched_keypoints.astype(int)
    matched_warped_keypoints = matched_warped_keypoints.astype(int)

    # 绘制匹配点和连线
    for i in range(len(matched_keypoints)):
        # 获取点坐标
        img1_coords = tuple(matched_keypoints[i])  # 第一张图的点
        img2_coords = tuple(matched_warped_keypoints[i])  # 第二张图的点
        img2_coords = (img2_coords[0] + img_width, img2_coords[1])  # 偏移宽度

        # 绘制点
        radius = 3
        point_thickness = -1  # 填充点
        image_pair = cv2.circle(image_pair, img1_coords, radius, (0, 255, 255), point_thickness)
        image_pair = cv2.circle(image_pair, img2_coords, radius, (0, 255, 255), point_thickness)

        # 绘制连线
        line_thickness = 1
        if good_matches_mask is None:
            color = (0, 255, 0)  # 全部使用绿色
        else:
            color = (0, 255, 0) if good_matches_mask[i] else (0, 0, 255)
        image_pair = cv2.line(image_pair, img1_coords, img2_coords, color, line_thickness)

    return image_pair

from PIL import Image
def visualize_keypoints(image1, image2, keypoints1, keypoints2):
    """
    Visualize keypoints distribution on two images without showing matching relationships.

    Args:
        image1 (numpy.ndarray): First image in CV2 format (H x W x C).
        image2 (numpy.ndarray): Second image in CV2 format (H x W x C).
        keypoints1 (numpy.ndarray): Keypoints for the first image, shape (N, 2).
        keypoints2 (numpy.ndarray): Keypoints for the second image, shape (M, 2).

    Returns:
        numpy.ndarray: Visualized image pair with keypoints.
    """
    # Convert tensors to OpenCV images if necessary
    image1 = tensor_to_cv2_image(image1)
    image2 = tensor_to_cv2_image(image2)
    img_width = image1.shape[1]

    # Align heights of the images
    height = max(image1.shape[0], image2.shape[0])
    if image1.shape[0] < height:
        image1 = np.pad(image1, ((0, height - image1.shape[0]), (0, 0), (0, 0)))
    if image2.shape[0] < height:
        image2 = np.pad(image2, ((0, height - image2.shape[0]), (0, 0), (0, 0)))

    # Combine the two images side by side
    image_pair = np.hstack((image1, image2))

    for kp in keypoints1:
        kp = kp.astype(int)
        image_pair = cv2.circle(image_pair, (kp[0], kp[1]), 3, (0, 255, 0), -1)

    for kp in keypoints2:
        kp = kp.astype(int)
        image_pair = cv2.circle(
            image_pair, (kp[0] + img_width, kp[1]), 3, (255, 0, 0), -1
        )

    return image_pair

def visualize(images_0_raw,images_1_raw,sparse_positions_0,sparse_positions_1,
              matches):
    OUTPUT_IMAGE_PATH = "./img.png"
    kp = visualize_keypoints(
        images_0_raw,
        images_1_raw,
        sparse_positions_0[0].detach().cpu().numpy(),
        sparse_positions_1[0].detach().cpu().numpy(),)
    cv2.imwrite('kp.png',kp)
    image_pair = img_pair_visual(
        images_0_raw,
        images_1_raw,
        sparse_positions_0[0][matches[:, 0]].detach().cpu().numpy(),
        sparse_positions_1[0][matches[:, 1]].detach().cpu().numpy(),
    )

    cv2.imwrite('out.png',image_pair)

    # heat_rgb1,heat_rgb2 = pca_to_rgb(heat1,heat2)
    #
    # pil_image = Image.fromarray(heat_rgb1)
    #
    # # 保存图像
    # pil_image.save('1.png')
    # pil_image = Image.fromarray(heat_rgb2)
    #
    # # 保存图像
    # pil_image.save('2.png')


def process_image(image_path,device):
    """
    读取图像并进行预处理，包括转换为 tensor 和归一化
    :param image_path: 图像文件的路径
    :return: 归一化后的图像 tensor 和图像尺寸 (宽, 高)
    """
    # 定义图像预处理流水线，包括转换为 tensor 和归一化操作
    image_transforms = transforms.Compose([
        transforms.ToTensor(),  # 将图像从 [H, W, C] 转为 [C, H, W] 并且像素值归一化到 [0, 1]
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 对于 RGB 图像常用的均值
        #                      std=[0.229, 0.224, 0.225])  # 对于 RGB 图像常用的标准差
    ])
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
    :param feats0: 第一张图像的特征字典，包含 'descriptors' 和 'keypoints'。
    :param feats1: 第二张图像的特征字典，包含 'descriptors' 和 'keypoints'。
    :return: 匹配结果的张量，形状为 [N, 2]。
    """
    desc0 = feats0['descriptors']
    desc1 = feats1['descriptors']

    # 如果维度是 [1, N, D]，移除第一个维度
    if desc0.dim() == 3 and desc0.size(0) == 1:
        desc0 = desc0.squeeze(0)
    if desc1.dim() == 3 and desc1.size(0) == 1:
        desc1 = desc1.squeeze(0)

    # 检查描述符形状
    assert desc0.ndim == 2 and desc1.ndim == 2, f"Descriptors must be 2D tensors, got {desc0.shape}, {desc1.shape}"

    # 计算欧氏距离矩阵
    dists = torch.cdist(desc0, desc1, p=2)  # [N0, N1]

    # 找到最近邻匹配
    matches01 = torch.argmin(dists, dim=1)  # [N0]

    # 生成匹配对
    indices0 = torch.arange(matches01.size(0), device=matches01.device)  # [N0]
    matches = torch.stack([indices0, matches01], dim=-1)  # [N0, 2]

    return matches.unsqueeze(0)


from gluefactory_nonfree.SuperPointNet_gauss2 import SuperPointNet_gauss2
from gluefactory.models.matchers.nearest_neighbor_matcher import NearestNeighborMatcher
from gluefactory_nonfree.superpoint import SuperPoint

if __name__ == '__main__':
    # weights_path = 'pretrained/superPointNet_20000_checkpoint.pth.tar'
    # weights_path = 'pretrained/noadd/superPointNet_30000_checkpoint.pth.tar'
    weights_path = 'pretrained/superpoint_v1.pth'

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    image1_path = "/usr1/home/s124mdg51_07/silk/data/megadepth1500/images/0015/492130269_796b5bf602_o.jpg"
    image2_path = "/usr1/home/s124mdg51_07/silk/data/megadepth1500/images/0015/3611827485_281ac6d564_o.jpg"
    # SuperPoint 模型的配置
    config = {
        'max_keypoints': 2048,  # 最大特征点数量
        'keypoint_threshold': 0.0  # 特征点阈值
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 处理图像并存储在 data1 和 data2 中
    data1 = process_image(image1_path, device)
    data2 = process_image(image2_path, device)

    # 初始化 SuperPoint 模型
    # sp_model = SuperPointNet_gauss2(config)

    sp_model = SuperPoint(config)

    # # 加载模型的state_dict时
    # state_dict = torch.load(weights_path, map_location=torch.device('cpu'))["model_state_dict"]
    # # 移除 'module.' 前缀
    # new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    #
    # sp_model.load_state_dict(new_state_dict)

    # sp_model.load_state_dict(torch.load(weights_path, map_location=device)["model_state_dict"])
    sp_model.load_state_dict(torch.load(weights_path, map_location=device))

    sp_model = sp_model.eval()
    sp_model = sp_model.to(device)
    r1 = sp_model(data1)
    r2 = sp_model(data2)



    # matches = nn_matcher(r1,r2)
    keypoints1 = r1['keypoints']
    keypoints2 = r2['keypoints']
    keypoint_scores1 = r1['keypoint_scores']
    keypoint_scores2 = r2['keypoint_scores']
    desc1 = r1['descriptors']
    desc2 = r2['descriptors']

    data = {
        "descriptors0": desc1,
        "descriptors1": desc2,
    }

    # 初始化匹配器
    matcher = NearestNeighborMatcher(conf={
        "ratio_thresh": 1,  # 设置 ratio 检查的阈值
        "distance_thresh": None,  # 可选：距离阈值
        "mutual_check": True,  # 是否进行互检查
    })

    # # 执行前向匹配
    matches = matcher(data)

    # 提取匹配结果
    matches0 = matches["matches0"]  # 对应 `descriptors0` 的匹配索引
    matches1 = matches["matches1"]  # 对应 `descriptors1` 的匹配索引
    matching_scores0 = matches["matching_scores0"]  # 置信度
    matching_scores1 = matches["matching_scores1"]

    # 将有效的索引和匹配值组成对
    print(matches0.shape)  # 打印 matches0 的形状
    m = matches0[0]
    matches = [[i, m[i].item()] for i in range(len(m)) if m[i].item() != -1]
    matches = np.array(matches)  # 将 matches 转换为 NumPy 数组
    print(matches)


    visualize(data1['image'],data2['image'],keypoints1,keypoints2,matches)