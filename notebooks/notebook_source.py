# %% [markdown]
# # Fashion MNIST Classification Project
#
# **Objective:** Build a Convolutional Neural Network (CNN) using PyTorch to classify grayscale images of clothing items from the Fashion-MNIST dataset into 10 categories.
#
# **Dataset:** Fashion-MNIST — 70,000 28x28 grayscale images (60,000 train / 10,000 test) across 10 clothing classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.
#
# **Workflow:**
# 1. Import libraries and set up the environment
# 2. Load and explore the Fashion-MNIST dataset
# 3. Preprocess the data (normalization, DataLoaders)
# 4. Build a CNN architecture
# 5. Train the model
# 6. Evaluate performance on the test set
# 7. Visualize predictions and analyze results

# %% [markdown]
# ## 1. Import Libraries

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report

import time
import random

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"Using device: {device}")

# %% [markdown]
# ## 2. Load the Fashion-MNIST Dataset
#
# We use `torchvision.datasets.FashionMNIST` to download and load the dataset.
# (A GitHub-hosted mirror of the official dataset is used as the download source in
# case the default host is unreachable from this environment.)

# %%
# Use a reliable mirror for the raw dataset files
datasets.FashionMNIST.mirrors = [
    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
]

# Class names for Fashion-MNIST
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Define transforms: convert to tensor and normalize using the dataset's mean/std
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))  # Fashion-MNIST mean/std
])

# Download / load train and test sets
train_dataset = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=transform
)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Image shape: {train_dataset[0][0].shape}")
print(f"Number of classes: {len(CLASS_NAMES)}")

# %% [markdown]
# ### 2.1 Explore the Data
#
# Let's visualize a few sample images with their labels to understand what we're working with.

# %%
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    img, label = train_dataset[i]
    # Un-normalize for display
    img_display = img.squeeze().numpy() * 0.3530 + 0.2860
    ax.imshow(img_display, cmap="gray")
    ax.set_title(CLASS_NAMES[label], fontsize=10)
    ax.axis("off")
plt.suptitle("Sample Fashion-MNIST Images", fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2.2 Class Distribution
#
# Check that the classes are balanced in the training set.

# %%
train_labels = np.array(train_dataset.targets)
unique, counts = np.unique(train_labels, return_counts=True)

plt.figure(figsize=(10, 4))
plt.bar([CLASS_NAMES[i] for i in unique], counts, color="steelblue")
plt.title("Training Set Class Distribution")
plt.ylabel("Number of samples")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

for i, c in zip(unique, counts):
    print(f"{CLASS_NAMES[i]:15s}: {c} samples")

# %% [markdown]
# ## 3. Prepare DataLoaders
#
# We split a validation set out of the training data and build DataLoaders for
# training, validation, and testing.

# %%
BATCH_SIZE = 256
VAL_SPLIT = 0.1

n_train = len(train_dataset)
n_val = int(n_train * VAL_SPLIT)
n_train_final = n_train - n_val

train_subset, val_subset = torch.utils.data.random_split(
    train_dataset, [n_train_final, n_val],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train batches: {len(train_loader)} ({len(train_subset)} samples)")
print(f"Val batches:   {len(val_loader)} ({len(val_subset)} samples)")
print(f"Test batches:  {len(test_loader)} ({len(test_dataset)} samples)")

# %% [markdown]
# ## 4. Build the CNN Model
#
# The architecture consists of two convolutional blocks (each with Conv2d, BatchNorm,
# ReLU, and MaxPool) followed by fully connected layers with dropout for regularization.

# %%
class FashionCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(FashionCNN, self).__init__()

        # Convolutional Block 1: 1 -> 32 channels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Convolutional Block 2: 32 -> 64 channels
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Convolutional Block 3: 64 -> 128 channels
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # After 3 pooling layers: 28 -> 14 -> 7 -> 3
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # 28x28 -> 14x14
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # 14x14 -> 7x7
        x = self.pool(F.relu(self.bn3(self.conv3(x))))   # 7x7 -> 3x3

        x = x.view(x.size(0), -1)  # Flatten

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

model = FashionCNN(num_classes=10).to(device)
print(model)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# %% [markdown]
# ### 4.1 Verify Output Shape
#
# Sanity check: pass a dummy batch through the model to confirm the output shape is correct.

# %%
dummy_input = torch.randn(4, 1, 28, 28).to(device)
dummy_output = model(dummy_input)
print(f"Input shape:  {dummy_input.shape}")
print(f"Output shape: {dummy_output.shape}")
assert dummy_output.shape == (4, 10), "Output shape mismatch!"
print("Model architecture verified successfully.")

# %% [markdown]
# ## 5. Define Loss Function, Optimizer, and Training Loop

# %%
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# %% [markdown]
# ## 6. Train the Model

# %%
NUM_EPOCHS = 10

history = {
    "train_loss": [], "train_acc": [],
    "val_loss": [], "val_acc": []
}

best_val_acc = 0.0
start_time = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_fashion_cnn.pth")

    print(f"Epoch [{epoch:2d}/{NUM_EPOCHS}] "
          f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

total_time = time.time() - start_time
print(f"\nTraining completed in {total_time/60:.2f} minutes")
print(f"Best validation accuracy: {best_val_acc:.4f}")

# %% [markdown]
# ### 6.1 Plot Training History

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

epochs_range = range(1, NUM_EPOCHS + 1)

axes[0].plot(epochs_range, history["train_loss"], label="Train Loss", marker="o")
axes[0].plot(epochs_range, history["val_loss"], label="Val Loss", marker="o")
axes[0].set_title("Loss over Epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(epochs_range, history["train_acc"], label="Train Accuracy", marker="o")
axes[1].plot(epochs_range, history["val_acc"], label="Val Accuracy", marker="o")
axes[1].set_title("Accuracy over Epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Evaluate on the Test Set
#
# Load the best model checkpoint (highest validation accuracy) and evaluate it on the
# held-out test set.

# %%
model.load_state_dict(torch.load("best_fashion_cnn.pth"))
test_loss, test_acc = evaluate(model, test_loader, criterion, device)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

# %% [markdown]
# ### 7.1 Confusion Matrix and Classification Report

# %%
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(9, 8))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix - Test Set")
plt.colorbar()
tick_marks = np.arange(len(CLASS_NAMES))
plt.xticks(tick_marks, CLASS_NAMES, rotation=45, ha="right")
plt.yticks(tick_marks, CLASS_NAMES)

thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], "d"),
                  ha="center", va="center",
                  color="white" if cm[i, j] > thresh else "black", fontsize=8)

plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.show()

print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

# %% [markdown]
# ### 7.2 Per-Class Accuracy

# %%
class_correct = np.zeros(10)
class_total = np.zeros(10)

for label, pred in zip(all_labels, all_preds):
    class_total[label] += 1
    if label == pred:
        class_correct[label] += 1

class_acc = class_correct / class_total

plt.figure(figsize=(10, 4))
plt.bar(CLASS_NAMES, class_acc, color="seagreen")
plt.title("Per-Class Test Accuracy")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=45, ha="right")
plt.axhline(y=test_acc, color="red", linestyle="--", label=f"Overall Acc: {test_acc:.3f}")
plt.legend()
plt.tight_layout()
plt.show()

for i, name in enumerate(CLASS_NAMES):
    print(f"{name:15s}: {class_acc[i]:.4f} ({int(class_correct[i])}/{int(class_total[i])})")

# %% [markdown]
# ## 8. Visualize Predictions
#
# Let's look at some correctly and incorrectly classified examples.

# %%
def unnormalize(img_tensor):
    return img_tensor.squeeze().cpu().numpy() * 0.3530 + 0.2860

# Get one batch of test images
images, labels = next(iter(test_loader))
images_dev = images.to(device)
with torch.no_grad():
    outputs = model(images_dev)
    _, preds = torch.max(outputs, 1)
preds = preds.cpu()

fig, axes = plt.subplots(3, 6, figsize=(15, 8))
for i, ax in enumerate(axes.flat):
    img = unnormalize(images[i])
    true_label = CLASS_NAMES[labels[i].item()]
    pred_label = CLASS_NAMES[preds[i].item()]
    correct = labels[i].item() == preds[i].item()

    ax.imshow(img, cmap="gray")
    color = "green" if correct else "red"
    ax.set_title(f"T: {true_label}\nP: {pred_label}", fontsize=8, color=color)
    ax.axis("off")

plt.suptitle("Model Predictions on Test Samples (Green=Correct, Red=Incorrect)", fontsize=13)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 8.1 Misclassified Examples
#
# Specifically inspect a sample of misclassified images across the full test set.

# %%
misclassified_idx = np.where(all_preds != all_labels)[0]
print(f"Total misclassified: {len(misclassified_idx)} / {len(all_labels)} "
      f"({len(misclassified_idx)/len(all_labels)*100:.2f}%)")

sample_idx = np.random.choice(misclassified_idx, size=min(12, len(misclassified_idx)), replace=False)

fig, axes = plt.subplots(2, 6, figsize=(15, 6))
for ax, idx in zip(axes.flat, sample_idx):
    img, label = test_dataset[idx]
    img_display = unnormalize(img)
    true_label = CLASS_NAMES[label]
    pred_label = CLASS_NAMES[all_preds[idx]]

    ax.imshow(img_display, cmap="gray")
    ax.set_title(f"T: {true_label}\nP: {pred_label}", fontsize=8, color="red")
    ax.axis("off")

plt.suptitle("Sample Misclassified Images", fontsize=13)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Summary and Conclusions
#
# **Model Architecture:** A 3-block CNN (Conv2d + BatchNorm + ReLU + MaxPool) followed
# by a 3-layer fully connected classifier with dropout regularization.
#
# **Results:**
# - The model was trained for 12 epochs using the Adam optimizer with a step learning
#   rate scheduler.
# - Batch normalization stabilized and accelerated training, while dropout helped
#   reduce overfitting between the training and validation sets.
# - Final test accuracy and the confusion matrix/classification report above summarize
#   overall and per-class performance.
#
# **Observations:**
# - The model performs very well on visually distinctive classes like *Trouser*,
#   *Bag*, *Sandal*, and *Ankle boot*.
# - The most common confusion occurs between visually similar upper-body garments —
#   *Shirt*, *T-shirt/top*, *Pullover*, and *Coat* — since these classes share similar
#   silhouettes in low-resolution grayscale images.
#
# **Possible Improvements:**
# - Data augmentation (random horizontal flips, small rotations/translations) to
#   improve generalization.
# - A deeper architecture or residual connections for higher capacity.
# - Hyperparameter tuning (learning rate, batch size, dropout rate) via a validation
#   grid/random search.
# - Ensembling multiple models to boost overall accuracy.

# %%
print("=" * 50)
print("FINAL RESULTS SUMMARY")
print("=" * 50)
print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
print(f"Final Test Accuracy:      {test_acc*100:.2f}%")
print(f"Final Test Loss:          {test_loss:.4f}")
print(f"Total Training Time:      {total_time/60:.2f} minutes")
print("=" * 50)
