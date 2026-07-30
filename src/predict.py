"""
Simple CLI to run the trained FashionCNN on a batch of Fashion-MNIST test
images and print predictions.

Usage:
    python src/predict.py --num-samples 8
"""

import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import CLASS_NAMES, FashionCNN


def main():
    parser = argparse.ArgumentParser(description="Run FashionCNN inference on test samples.")
    parser.add_argument("--weights", default="../models/best_fashion_cnn.pth",
                         help="Path to the trained model checkpoint (.pth)")
    parser.add_argument("--num-samples", type=int, default=8,
                         help="Number of test images to run predictions on")
    parser.add_argument("--data-dir", default="../data",
                         help="Directory to download/cache the Fashion-MNIST dataset")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use a reliable mirror in case the default host is unreachable
    datasets.FashionMNIST.mirrors = [
        "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
    ]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    test_dataset = datasets.FashionMNIST(
        root=args.data_dir, train=False, download=True, transform=transform
    )
    loader = DataLoader(test_dataset, batch_size=args.num_samples, shuffle=True)

    model = FashionCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    images, labels = next(iter(loader))
    images = images.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    for i in range(len(labels)):
        true_label = CLASS_NAMES[labels[i].item()]
        pred_label = CLASS_NAMES[preds[i].item()]
        mark = "✓" if true_label == pred_label else "✗"
        print(f"[{mark}] True: {true_label:15s} Predicted: {pred_label}")


if __name__ == "__main__":
    main()
