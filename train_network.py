import os
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
import matplotlib.pyplot as plt
import numpy as np
from loader import ImageDatasetWithoutLabels  # Custom dataset loader

# Parse command-line arguments for configuration
def parse_args():
    parser = argparse.ArgumentParser(description='Train a ResNet18 model with and without curriculum learning.')
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of total epochs for training (default: 50)')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training (default: 128)')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--log-interval', type=int, default=10, help='Log training status every n batches')
    # Curriculum learning parameters
    parser.add_argument('--curr-factor', type=float, default=5.0, help='Factor for curriculum scheduling (default: 5.0)')
    parser.add_argument('--curriculum-type', type=str, default='contrast_only',
                        choices=['no_curriculum', 'blur_only', 'contrast_only', 'both'],
                        help='Type of curriculum learning to apply (default: no_curriculum)')
    parser.add_argument('--prog-type', type=str, default='exponential',
                        choices=['linear', 'logarithmic', 'exponential'], help='Progression Curve (default: linear)')
    # Paths for data and output
    parser.add_argument('--output-dir', type=str, default='./output', help='Directory to save outputs (default: ./output)')
    parser.add_argument('--data-dir', type=str, default='./tiny-imagenet-200', help='Path to Tiny ImageNet dataset (default: ./tiny-imagenet-200')
    return parser.parse_args()

# Map epochs to ages using curriculum learning
def apply_curriculum(epoch, total_epochs, age_range=(0, 216), curr_factor=None, prog_type='linear'):
    """
    Calculate a curriculum-based "age" for the current epoch.
    Args:
        epoch (int): Current training epoch.
        total_epochs (int): Total number of training epochs.
        age_range (tuple): Age range for mapping epochs.
        curr_factor (float): Scaling factor for curriculum learning.
        prog_type (str): Type of progression curve ('linear', 'logarithmic', 'exponential').
    Returns:
        int: Computed age for the current epoch.
    """
    scale = age_range[1] - age_range[0]
    normalized_epoch = epoch / total_epochs
    clip_point = 17 / 20  # Adjust scaling
    adjusted_epoch = min(normalized_epoch / clip_point, 1)

    if prog_type == 'linear':
        age = age_range[0] + adjusted_epoch * scale * curr_factor
    elif prog_type == 'logarithmic':
        log_scaled = np.log1p(adjusted_epoch * (np.expm1(curr_factor) - 1)) / curr_factor
        age = age_range[0] + log_scaled * scale
    elif prog_type == 'exponential':
        exp_scaled = (np.exp(adjusted_epoch * curr_factor) - 1) / (np.exp(curr_factor) - 1)
        age = age_range[0] + exp_scaled * scale
    else:
        raise ValueError(f"Unsupported curriculum type: {prog_type}")

    return int(min(max(age, age_range[0]), age_range[1]))

# Visualize the curriculum mapping
def plot_curriculum(epochs, curr_factor, prog_type, output_dir, curriculum_function, age_range=(0, 216)):
    """
    Create a plot showing the curriculum's age mapping over epochs.
    Args:
        epochs (int): Total number of epochs.
        curr_factor (float): Scaling factor for curriculum mapping.
        prog_type (str): Progression Curve ('linear', 'logarithmic', 'exponential').
        output_dir (str): Directory to save the plot.
        curriculum_function (function): Mapping function to compute ages.
        age_range (tuple): Age range for mapping.
    """
    os.makedirs(output_dir, exist_ok=True)
    ages = [curriculum_function(e, epochs, age_range, curr_factor, prog_type) for e in range(epochs)]

    plt.figure(figsize=(8, 6))
    plt.plot(range(epochs), ages, label=f'{prog_type.capitalize()} (Factor: {curr_factor})')
    plt.xlabel('Epochs')
    plt.ylabel('Assigned Age (Months)')
    plt.title('Curriculum Age Assignment over Epochs')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'curriculum_{prog_type}.png'))
    plt.show()
    plt.close()

# Train the model for one epoch
def train_model(model, train_loader, optimizer, criterion, device, epoch, num_epochs, curr_factor, prog_type, is_curriculum, curriculum_function=None):
    """
    Train the model for a single epoch.
    Args:
        model: The PyTorch model to train.
        train_loader: Dataloader for training data.
        optimizer: Optimizer for the model.
        criterion: Loss function.
        device: Training device ('cpu' or 'cuda').
        epoch: Current epoch.
        num_epochs: Total epochs.
        curr_factor: Curriculum scaling factor.
        prog_type: Progression Curve ('linear', 'logarithmic', 'exponential').
        is_curriculum: Apply curriculum learning if True.
        curriculum_function: Mapping function for curriculum learning.
    Returns:
        float: Average training loss.
    """
    curriculum_function = curriculum_function or apply_curriculum
    model.train()

    # Apply curriculum learning if enabled
    age = curriculum_function(epoch, num_epochs, curr_factor=curr_factor, prog_type=prog_type) if is_curriculum else None
    if is_curriculum:
        train_loader.dataset.age = age  # Update dataset with assigned age
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels) / len(outputs)  # Normalize by batch size
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, batch_idx * len(images), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs} | Training Loss: {avg_loss:.4f} | Age: {age}")
    return avg_loss

# Evaluate the model
def evaluate_model(model, device, validation_loader, criterion, set_name="Validation"):
    """
    Evaluate the model's performance on the validation dataset.
    Args:
        model: The PyTorch model to evaluate.
        device: Device to perform evaluation.
        validation_loader: DataLoader for validation data.
        criterion: Loss function.
        set_name: Name of the dataset (e.g., 'Validation').
    Returns:
        tuple: Accuracy and validation loss.
    """
    model.eval()
    val_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            val_loss += criterion(outputs, target).item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    val_loss /= len(validation_loader.dataset)
    accuracy = 100. * correct / total
    print(f"{set_name} | Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return accuracy, val_loss

# Main function to run the training and evaluation
def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # for Mac: M3 Chip

    # Define data transformations to avoid overfitting
    transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.RandomCrop(size=(56, 56)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    plot_curriculum(args.epochs, args.curr_factor, args.prog_type, './curriculum_plot', apply_curriculum)


    # Prepare datasets and dataloaders based on curriculum type
    root_dir = args.data_dir
    print(f"Loading datasets from: {root_dir}")
    if args.curriculum_type == "no_curriculum":
        train_loader = DataLoader(ImageDatasetWithoutLabels(root_dir, age=None, transform=transform),
                                batch_size=args.batch_size, shuffle=True)
    elif args.curriculum_type == "blur_only":
        train_loader = DataLoader(ImageDatasetWithoutLabels(root_dir, age=None, blur=True, transform=transform),
                                batch_size=args.batch_size, shuffle=True)
    elif args.curriculum_type == "contrast_only":
        train_loader = DataLoader(ImageDatasetWithoutLabels(root_dir, age=None, contrast=True, transform=transform),
                                batch_size=args.batch_size, shuffle=True)
    elif args.curriculum_type == "both":
        train_loader = DataLoader(ImageDatasetWithoutLabels(root_dir, age=None, blur=True, contrast=True, transform=transform),
                                batch_size=args.batch_size, shuffle=True)
    else:
        raise ValueError(f"Unsupported curriculum type: {args.curriculum_type}")
    print(f"Using Curriculum Type: {args.curriculum_type}")

    validation_loader = DataLoader(ImageDatasetWithoutLabels(root_dir, split="val", transform=transform),
                                   batch_size=args.batch_size, shuffle=False)

    # Model initialization
    print("Initializing the model...")
    print(f"Using model type: ResNet18")
    # Model initialization ResNet18
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 200)

    model.to(device)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    train_losses, validation_losses = [], []
    best_val_loss = float('inf')
    best_model_weights = None

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss = train_model(model, train_loader, optimizer, criterion, device, epoch, args.epochs,
                                 args.curr_factor, args.prog_type, args.curriculum_type != "no_curriculum")
        train_losses.append(train_loss)
        val_acc, val_loss = evaluate_model(model, device, validation_loader, criterion)
        validation_losses.append(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict().copy()

    # Load best model
    if best_model_weights:
        model.load_state_dict(best_model_weights)

    # Save results
    model_save_path = os.path.join(args.output_dir, f"r18_{args.curriculum_type}_best.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Best model saved to {model_save_path}")

    losses_save_path = os.path.join(args.output_dir, f"r18_{args.curriculum_type}_losses.npz")
    np.savez(losses_save_path, train_losses=train_losses, validation_losses=validation_losses)
    print(f"Training and validation losses saved to {losses_save_path}")

    # Plot and save loss curves
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
    plt.plot(range(1, len(validation_losses) + 1), validation_losses, label="Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f"Loss Curves for r18 ({args.curriculum_type})")
    plt.legend()
    loss_plot_path = os.path.join(args.output_dir, f"r18_{args.curriculum_type}_loss_plot.png")
    plt.savefig(loss_plot_path)
    print(f"Loss plot saved to {loss_plot_path}")
    plt.show()
    plt.close()

if __name__ == "__main__":
    args = parse_args()
    run(args)