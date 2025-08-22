import os
import random
import torch
from PIL import Image, ImageFilter      # Image processing libraries
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms, models


# Custom dataset for loading images without labels
class ImageDatasetWithoutLabels(Dataset):
    def __init__(self, root_dir, split="train", age=None, transform=None, blur=False, contrast=False, num_images=None, seed=None):
        """
        Args:
            root_dir (str): Root directory of the dataset.
            split (str): Dataset split ("train" or "val").
            age (int, optional): Age in months for curriculum-based transformations.
            transform (callable, optional): Transformations to apply to images.
            blur (bool): Whether to apply blur transformation based on age.
            contrast (bool): Whether to apply contrast transformation based on age.
            num_images (int, optional): Limit the number of images for debugging/testing.
            seed (int, optional): Seed for randomization (for reproducibility).
        """
        self.root_dir = os.path.join(root_dir, split)  # Define dataset path
        self.age = age  # Age parameter for curriculum learning
        self.blur = blur  # Blur flag
        self.contrast = contrast  # Contrast flag
        self.transform = transform  # Image transformations
        self.image_paths = []  # Store paths to images
        self.labels = []  # Store corresponding labels

        if seed is not None:
            random.seed(seed)  # Set seed for reproducibility

            # Load images and labels
        if split == "train":
            for class_dir in os.listdir(self.root_dir):
                class_path = os.path.join(self.root_dir, class_dir, "images")
                if os.path.isdir(class_path):  # Check if it's a directory
                    for img_name in os.listdir(class_path):  # Iterate over images in the directory
                        self.image_paths.append(os.path.join(class_path, img_name))  # Append image paths
                        self.labels.append(class_dir)  # Class folder name as label
        elif split == "val":
            # Validation data uses an annotation file
            val_annotations_path = os.path.join(self.root_dir, "val_annotations.txt")
            with open(val_annotations_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split("\t")  # Parse annotation file
                    img_name, label = parts[0], parts[1]
                    self.image_paths.append(os.path.join(self.root_dir, "images", img_name))
                    self.labels.append(label)

        # Create a mapping from labels to numeric indices
        unique_labels = sorted(set(self.labels))  # Unique sorted labels
        self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}  # Map labels to indices
        self.labels = [self.label_to_index[label] for label in self.labels]  # Convert labels to indices

        # Optionally limit the number of images
        if num_images is not None:
            combined = list(zip(self.image_paths, self.labels))  # Pair paths and labels
            random.shuffle(combined)  # Shuffle for randomness
            combined = combined[:num_images]  # Limit number of samples
            self.image_paths, self.labels = zip(*combined)  # Unpack paths and labels

    def __len__(self):
        """Return the total number of samples."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Load an image and its corresponding label."""
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")  # Open image with PIL and ensure RGB format
        label = self.labels[idx]  # Retrieve label

        # Apply curriculum transformations (blur and/or contrast)
        if self.age is not None:
            if self.blur:
                image = self.adjust_blur(image, self.age)  # Apply blur
            if self.contrast:
                image = self.adjust_contrast(image, self.age)  # Apply contrast

        # Apply additional transformations if provided
        if self.transform:
            image = self.transform(image)  # Ensure image is converted to a Tensor
        else:
            # Default transformation if none is provided
            default_transform = transforms.ToTensor()
            image = default_transform(image)

        return image, torch.tensor(label, dtype=torch.long)  # Return image and label as tensors

    @staticmethod
    def adjust_blur(image, age_months):
        """
        Adjust image blur based on the provided age in months.
        Args:
            image (PIL.Image): Image to apply blur on.
            age_months (int): Age in months.
        Returns:
            PIL.Image: Blurred image.
        """
        # Data for age-based visual acuity (simulating how sharp vision is at different ages)
        age_data_acuity = np.array([0, 1, 3, 6, 12, 24, 36, 48])  # Age intervals
        acuity_data = np.array([600, 581, 231, 109, 66, 36, 25, 20])  # Corresponding visual acuity values

        # Interpolate visual acuity based on age
        acuity = np.interp(age_months, age_data_acuity, acuity_data)

        max_blur = 4.0  # Maximum blur radius
        blur_radius = max_blur * (acuity - acuity_data[-1]) / (acuity_data[0] - acuity_data[-1])  # Map acuity to blur

        return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))  # Apply Gaussian blur

    @staticmethod
    def adjust_contrast(image, age_months):
        """
        Adjust image contrast based on the provided age in months.
        Args:
            image (PIL.Image): Image to adjust contrast for.
            age_months (int): Age in months.
        Returns:
            PIL.Image: Contrast-adjusted image.
        """
        # Data for age-based contrast sensitivity
        age_data_contrast = np.array([1, 3, 6, 8, 10, 216])  # Age intervals
        contrast_data = np.array([2, 8, 20, 30, 45, 50], dtype=float) / 50  # Corresponding contrast levels

        # Interpolate contrast sensitivity based on age
        peak_sensitivity = np.interp(age_months, age_data_contrast, contrast_data)

        # Adjust contrast directly
        image = np.array(image, dtype=np.float32) / 255.0  # Normalize image to [0, 1]
        adjusted_image = (image - 0.5) * peak_sensitivity + 0.5  # Scale contrast
        adjusted_image = np.clip(adjusted_image, 0, 1)  # Clip values to valid range
        return Image.fromarray((adjusted_image * 255).astype(np.uint8))  # Convert back to image