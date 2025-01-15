import numpy as np
import torch
# from torch.autograd import Variable
# import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
# from tqdm import tqdm
# from utils.loader import dataLoader, modelLoader, pretrainedLoader
import logging

from utils.tools import dict_update

# from utils.utils import labels2Dto3D, flattenDetection, labels2Dto3D_flattened
# from utils.utils import pltImshow, saveImg
from utils.utils import precisionRecall_torch
# from utils.utils import save_checkpoint

from pathlib import Path
from Train_model_frontend import Train_model_frontend

# load confi
filename = "configs/superpoint_coco_train_heatmap.yaml"
import yaml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_default_tensor_type(torch.FloatTensor)
with open(filename, "r") as f:
    config = yaml.safe_load(f)

from utils.loader import dataLoader as dataLoader

# data = dataLoader(config, dataset='hpatches')
task = config["data"]["dataset"]

data = dataLoader(config, dataset=task, warp_input=True)
# test_set, test_loader = data['test_set'], data['test_loader']
train_loader, val_loader = data["train_loader"], data["val_loader"]

# model_fe = Train_model_frontend(config)
# print('==> Successfully loaded pre-trained network.')
# import tqdm
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

# 假设已经导入 train_loader
# 创建一个 2x4 的子图
num_images = 8
rows, cols = 2, 4

# 迭代训练数据
for i, sample_train in tqdm(enumerate(train_loader)):
    img, labels_2D, mask_2D = (
        sample_train["image"],
        sample_train["labels_2D"],
        sample_train["valid_mask"],
    )
    
    # 创建图形和子图
    fig, axes = plt.subplots(rows, cols, figsize=(16, 8))
    axes = axes.flatten()  # 将子图展平，方便索引

    for j in range(num_images):
        # 找到 labels_2D 中非零元素的索引
        non_zero_indices = torch.nonzero(labels_2D[j])  # j 表示当前批次中的第 j 张图

        # 显示灰度图
        axes[j].imshow(img[j].squeeze().cpu().numpy(), cmap='gray')

        # 如果 labels_2D 有非零元素，标记这些点
        if non_zero_indices.numel() > 0:
            y_coords, x_coords = non_zero_indices[:, 1].cpu().numpy(), non_zero_indices[:, 2].cpu().numpy()  # 只保留2和3的索引
            axes[j].scatter(x_coords, y_coords, c='red', s=10, label='Non-zero points')  # 标记非零点

        axes[j].set_title(f'Image {j+1} - Non-zero Points')
        axes[j].axis('off')  # 不显示坐标轴

    plt.tight_layout()  # 自动调整子图间距
    plt.show()

    # 如果只想查看非零元素的数量
    for j in range(num_images):
        num_nonzero = non_zero_indices.size(0)
        print(f'Total non-zero elements in labels_2D for image {j+1}: {num_nonzero}')
