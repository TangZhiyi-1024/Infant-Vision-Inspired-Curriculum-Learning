import os
import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from loader import ImageDatasetWithoutLabels  # 加载自定义数据集类


def calculate_rdm(features, metric="euclidean"):
    """
    计算 Representational Dissimilarity Matrix (RDM)。
    Args:
        features (numpy.ndarray): 特征矩阵 (n_samples, n_features)。
        metric (str): 距离度量方式，例如 "euclidean" 或 "cosine"。
    Returns:
        numpy.ndarray: RDM 矩阵。
    """
    distances = pdist(features, metric=metric)
    return squareform(distances)


def plot_rdm(rdm, title, save_path):
    """
    绘制并保存 RDM 热力图。
    Args:
        rdm (numpy.ndarray): RDM 矩阵。
        title (str): 图像标题。
        save_path (str): 保存路径。
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 确保目录存在
    plt.figure(figsize=(8, 6))
    plt.imshow(rdm, cmap="viridis", interpolation="nearest")
    plt.colorbar()
    plt.title(title)
    plt.savefig(save_path)
    plt.close()


def generate_rdms(curriculum_types, root_dir, num_images, layers, output_dir):
    """
    生成 RDM 并保存。
    Args:
        curriculum_types (list): 课程学习类型（如 "blur_only", "contrast_only" 等）。
        root_dir (str): 数据集根目录。
        num_images (int): 加载图像的数量。
        layers (list): 要提取的模型层名称。
        output_dir (str): 保存 RDM 的输出目录。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(pretrained=False).to(device)
    model.eval()

    for curriculum_type in curriculum_types:
        # 使用 loader 加载数据
        dataset = ImageDatasetWithoutLabels(
            root_dir=root_dir,
            split="train",
            num_images=num_images,
            blur=curriculum_type in ["blur_only", "both"],
            contrast=curriculum_type in ["contrast_only", "both"]
        )
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

        for layer_name in layers:
            activation = {layer_name: []}

            # 注册钩子以捕获中间层输出
            def hook_fn(module, input, output):
                activation[layer_name].append(output.detach().cpu().numpy())

            hook = dict(model.named_modules())[layer_name].register_forward_hook(hook_fn)

            # 前向传播
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                model(inputs)

            hook.remove()

            # 计算 RDM
            features = np.concatenate(activation[layer_name], axis=0).reshape(len(dataset), -1)
            rdm = calculate_rdm(features)

            # 保存 RDM
            save_path = os.path.join(output_dir, f"rdm_{curriculum_type}_{layer_name}.png")
            plot_rdm(rdm, f"{curriculum_type} - {layer_name}", save_path)
            print(f"RDM saved: {save_path}")


if __name__ == "__main__":
    # 配置
    curriculum_types = ["no_curriculum", "blur_only", "contrast_only", "both"]  # 课程学习类型
    root_dir = "./tiny-imagenet-200"  # 数据集路径
    num_images = 100  # 加载前 100 张图像
    layers_to_extract = ["layer1", "layer3", "fc"]  # 提取的模型层
    output_dir = "./rdm_results"  # RDM 保存路径

    # 生成 RDM
    generate_rdms(curriculum_types, root_dir, num_images, layers_to_extract, output_dir)
