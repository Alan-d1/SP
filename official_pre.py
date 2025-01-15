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
    image1 = tensor_to_cv2_image(image1)
    image2 = tensor_to_cv2_image(image2)
    img_width = image1.shape[1]

    height = max(image1.shape[0], image2.shape[0])

    if image1.shape[0] < height:
        image1 = np.pad(image1, ((0, height - image1.shape[0]), (0, 0), (0, 0)))

    if image2.shape[0] < height:
        image2 = np.pad(image2, ((0, height - image2.shape[0]), (0, 0), (0, 0)))

    image_pair = np.hstack((image1, image2))

    # convert keypoints to col, row (x, y) order
    matched_keypoints = matched_keypoints[:, [1, 0]]
    matched_warped_keypoints = matched_warped_keypoints[:, [1, 0]]

    matched_keypoints = matched_keypoints.astype(int)
    matched_warped_keypoints = matched_warped_keypoints.astype(int)

    # draw matched keypoint points and lines associating matched keypoints (point correspondences)
    for i in range(len(matched_keypoints)):
        img1_coords = matched_keypoints[i]
        img2_coords = matched_warped_keypoints[i]
        # add the width so the coordinates show up correctly on the second image
        img2_coords = (img2_coords[0] + img_width, img2_coords[1])

        radius = 1
        thickness = 2
        # points will be red (BGR color)
        image_pair = cv2.circle(image_pair, img1_coords, radius, (0, 0, 255), thickness)
        image_pair = cv2.circle(image_pair, img2_coords, radius, (0, 0, 255), thickness)

        thickness = 1

        if good_matches_mask is None:
            color = (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255),
            )
        else:
            if good_matches_mask[i]:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
        image_pair = cv2.line(image_pair, img1_coords, img2_coords, color, thickness)
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

    # Draw keypoints on the first image
    for kp in keypoints1:
        kp = kp.astype(int)
        image_pair = cv2.circle(image_pair, (kp[1], kp[0]), 3, (0, 255, 0), -1)

    # Draw keypoints on the second image
    for kp in keypoints2:
        kp = kp.astype(int)
        image_pair = cv2.circle(
            image_pair, (kp[1] + img_width, kp[0]), 3, (255, 0, 0), -1
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

    cv2.imwrite('out.png',kp)

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


from gluefactory_nonfree.SuperPointNet_gauss2 import SuperPointNet_gauss2

if __name__ == '__main__':
    weights_path = 'pretrained/superPointNet_20000_checkpoint.pth.tar'
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
    sp_model = SuperPointNet_gauss2(config)
    sp_model.load_state_dict(torch.load(weights_path, map_location=device)["model_state_dict"])
    sp_model = sp_model.eval()
    sp_model = sp_model.to(device)
    r1 = sp_model(data1)
    r2 = sp_model(data2)
    matches = nn_matcher(r1,r2)
    keypoints1 = r1['keypoints']
    keypoints2 = r2['keypoints']
    keypoint_scores1 = r1['keypoint_scores']
    keypoint_scores2 = r2['keypoint_scores']
    desc1 = r1['descriptors']
    desc2 = r2['descriptors']
    visualize(data1['image'],data2['image'],keypoints1,keypoints2,matches)