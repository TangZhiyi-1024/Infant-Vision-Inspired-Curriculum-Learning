# Infant-Vision-Inspired Curriculum Learning (ResNet18 on Tiny ImageNet-200)

This project trains **ResNet18** on **Tiny-ImageNet-200** with a curriculum that simulates infant visual development: we progressively **reduce blur and contrast degradation** early in training and phase in clean images near the end for steadier optimization and better generalization.&#x20;

## Features

* Gradually decreasing **blur/contrast** strength over training (curriculum scheduling)
* **RDM (Representational Dissimilarity Matrix)** analysis across layers
* Downstream evaluation with **kNN** and **Logistic Regression** on intermediate features
* Transparent training knobs: optimizer, hyperparameters, and data augmentation.&#x20;

## Dataset

* **Tiny-ImageNet-200** (64×64, 200 classes; typical splits 100k train / 10k val).&#x20;

## Method Overview

* **Backbone**: ResNet18
* **Optimization**: SGD (`lr=0.1`, `momentum=0.9`) with standard augmentations (random crop/flip).
* **Curriculum schedule**:

  * Map training progress to a notional “age” (0–216 months) using an **exponential** progression (`curr_factor = 5.0`).
  * For the first **\~85%** of training, gradually **dial down** blur/contrast strength, then finish on **clean** images.
* **Variants**: baseline / blur-only / contrast-only / **both (blur+contrast)**.
* **Observation**: the **combined** curriculum yields the **lowest validation loss**, faster convergence, and more stable gradients.&#x20;

## Data Selection & Preprocessing (Example)

A custom loader selects a fixed number of unique images (with optional seed for reproducibility) and splits them evenly into four transformation groups: (1) no transform, (2) low-acuity blur, (3) contrast adjustment, (4) blur + contrast. The transformed samples are wrapped into `TensorDataset`s and `DataLoader`s for efficient batching.&#x20;

## RDM Analysis

We register forward hooks on multiple layers (e.g., `layer1`, `layer3`, `fc`) to extract activations, flatten them, compute pairwise distances, and visualize the **RDM** as heatmaps. Lower layers are more sensitive to local transforms; deeper layers integrate effects into more **abstract** representations. Contrast-only emphasizes structured dissimilarities; blur-only and both add diversity; baseline appears more uniform.&#x20;

## Downstream Evaluation (Illustrative Results)

* **layer1**: kNN **4.9%** (k=20), Logistic Regression **6.2%**
* **layer3**: kNN **13.6%** (k=20), Logistic Regression **18.1%**
* **fc**: kNN **11.7%** (k=20), Logistic Regression **16.2%**
* Training gradient norm ≈ **2040.08**

These results suggest progressive feature refinement across depth, with **layer3** features being most discriminative for simple downstream probes in this setup.&#x20;



