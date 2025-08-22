import os
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from loader import ImageDatasetWithoutLabels  # 加载自定义数据集类


def calculate_rdm(features, metric="euclidean"):
    """ 计算 Representational Dissimilarity Matrix (RDM) """
    distances = pdist(features, metric=metric)
    return squareform(distances)


def plot_rdm(rdm, title, save_path):
    """ 绘制并保存 RDM 热力图 """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(rdm, cmap="viridis", interpolation="nearest")
    plt.colorbar()
    plt.title(title)
    plt.savefig(save_path)
    plt.close()


def evaluate_downstream_performance(features, labels):
    """
    评估模型在下游任务上的性能，包括 kNN 和 Logistic 回归。
    """
    # 归一化 & PCA 降维
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    pca = PCA(n_components=128)  # 降到 128 维
    features = pca.fit_transform(features)

    # 划分数据集 (80% 训练, 20% 测试)
    num_train = int(0.8 * len(features))
    train_features, test_features = features[:num_train], features[num_train:]
    train_labels, test_labels = labels[:num_train], labels[num_train:]

    print(f"Train features: {train_features.shape}, Train labels: {train_labels.shape}")
    print(f"Test features: {test_features.shape}, Test labels: {test_labels.shape}")

    # kNN 分类（尝试不同的 k）
    best_knn_acc = 0
    best_k = None
    for k in [1, 3, 5, 10, 20]:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_features, train_labels)
        knn_predictions = knn.predict(test_features)
        knn_acc = accuracy_score(test_labels, knn_predictions)
        print(f"kNN k={k}, Accuracy: {knn_acc:.4f}")

        if knn_acc > best_knn_acc:
            best_knn_acc = knn_acc
            best_k = k

    # Logistic 回归
    logistic = LogisticRegression(max_iter=5000)
    logistic.fit(train_features, train_labels)
    logistic_predictions = logistic.predict(test_features)
    logistic_acc = accuracy_score(test_labels, logistic_predictions)

    print(f"Best kNN k={best_k}, Accuracy: {best_knn_acc:.4f}")
    print(f"Logistic Regression Accuracy: {logistic_acc:.4f}")

    return {
        "kNN_Accuracy": best_knn_acc,
        "Logistic_Accuracy": logistic_acc
    }


def extract_features(model, dataloader, layers, device):
    """
    提取模型的中间层特征
    """
    model.eval()
    layer_features = {layer: [] for layer in layers}
    labels = []

    def hook_fn(layer_name):
        def hook(module, input, output):
            layer_features[layer_name].append(output.detach().cpu().numpy().reshape(output.size(0), -1))

        return hook

    hooks = []
    for layer in layers:
        hooks.append(dict(model.named_modules())[layer].register_forward_hook(hook_fn(layer)))

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            model(inputs)
            labels.extend(targets.numpy())

    for hook in hooks:
        hook.remove()

    for layer in layers:
        layer_features[layer] = np.concatenate(layer_features[layer], axis=0)

    return layer_features, np.array(labels)


def compute_gradient_norm(model, loss):
    """
    计算模型梯度范数
    """
    model.zero_grad()
    loss.backward()
    grad_norm = 0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm += torch.norm(param.grad, p=2).item()
    return grad_norm


if __name__ == "__main__":
    # 配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(pretrained=True).to(device)

    root_dir = "./tiny-imagenet-200"  # 数据集路径
    num_images = 5000  # 训练更多数据
    dataset = ImageDatasetWithoutLabels(root_dir=root_dir, split="train", num_images=num_images)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    layers_to_evaluate = ["layer1", "layer3", "fc"]  # 提取的模型层

    # 提取特征
    layer_features, labels = extract_features(model, dataloader, layers_to_evaluate, device)

    # 计算下游任务性能
    for layer, features in layer_features.items():
        print(f"Evaluating downstream performance for layer: {layer}")
        performance = evaluate_downstream_performance(features, labels)
        print(
            f"Layer {layer} - kNN Accuracy: {performance['kNN_Accuracy']:.4f}, Logistic Regression Accuracy: {performance['Logistic_Accuracy']:.4f}")

    # 计算梯度范数
    loss_fn = torch.nn.CrossEntropyLoss()
    inputs, targets = next(iter(dataloader))
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)

    grad_norm = compute_gradient_norm(model, loss)
    print(f"Gradient Norm: {grad_norm:.4f}")
