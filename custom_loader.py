from loader import ImageDatasetWithoutLabels  # 使用现有的功能
from PIL import Image, ImageFilter
import random
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset


class CustomDatasetLoader:
    def __init__(self, root_dir, split="val", num_images=100, seed=None, age_months=12):
        """
        Initialize the custom dataset loader with image selection and transformations.

        Args:
            root_dir (str): Root directory of the dataset.
            split (str): Dataset split ("train" or "val").
            num_images (int): Number of images to select (default: 100).
            seed (int, optional): Random seed for reproducibility.
            age_months (int): Age in months for curriculum learning (default: 12).
        """
        self.root_dir = root_dir
        self.split = split
        self.num_images = num_images
        self.seed = seed
        self.age_months = age_months
        self.image_paths = self.select_random_images()  # 自动随机选择图片
        self.transformed_images = self.apply_transformations()  # 自动应用变换

    def select_random_images(self):
        """
        Randomly select images from the dataset.

        Returns:
            list: Selected image paths.
        """
        dataset = ImageDatasetWithoutLabels(self.root_dir, split=self.split)
        if self.seed is not None:
            random.seed(self.seed)
        indices = random.sample(range(len(dataset)), self.num_images)
        return [dataset.image_paths[idx] for idx in indices]

    def apply_transformations(self):
        """
        Apply transformations to the selected images.

        Returns:
            dict: Dictionary of transformed images.
        """
        # 定义四种变换
        transform_no_change = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        transform_low_acuity = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Lambda(lambda img: img.filter(ImageFilter.GaussianBlur(radius=2))),
            transforms.ToTensor()
        ])
        transform_contrast_adjust = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Lambda(lambda img: (img - 0.5) * 1.5 + 0.5),
            transforms.Lambda(lambda img: torch.clamp(img, 0, 1))
        ])
        transform_both = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Lambda(lambda img: img.filter(ImageFilter.GaussianBlur(radius=2))),
            transforms.ToTensor(),
            transforms.Lambda(lambda img: (img - 0.5) * 1.5 + 0.5),
            transforms.Lambda(lambda img: torch.clamp(img, 0, 1))
        ])

        transformed_images = {"no_change": [], "low_acuity": [], "contrast_adjust": [], "both": []}
        num_images_per_group = self.num_images // 4  # 每组 25%

        for i, img_path in enumerate(self.image_paths):
            img = Image.open(img_path).convert("RGB")
            if i < num_images_per_group:
                transformed_images["no_change"].append(transform_no_change(img))
            elif i < 2 * num_images_per_group:
                transformed_images["low_acuity"].append(transform_low_acuity(img))
            elif i < 3 * num_images_per_group:
                transformed_images["contrast_adjust"].append(transform_contrast_adjust(img))
            else:
                transformed_images["both"].append(transform_both(img))

        return transformed_images

    def create_dataloaders(self, batch_size=8):
        """
        Create DataLoaders for each transformation group.

        Args:
            batch_size (int): Batch size for DataLoader.

        Returns:
            dict: DataLoaders for each transformation group.
        """
        dataloaders = {}
        for key, images in self.transformed_images.items():
            images_tensor = torch.stack(images)
            dummy_labels = torch.zeros(len(images_tensor))  # 占位标签
            dataset = TensorDataset(images_tensor, dummy_labels)
            dataloaders[key] = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return dataloaders



