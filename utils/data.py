from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pathlib import Path


# Pytorch preprocessing
achille_preprocess = transforms.Compose([
    transforms.Pad(2),       # 28x28 → 32x32 
    transforms.ToTensor(),
])

achille_transform_train = transforms.Compose([
    transforms.Pad(2), 
    transforms.RandomAffine(degrees=0, translate=(4/32, 4/32)),  # random translation
    transforms.RandomHorizontalFlip(),  # horizontal flip
    transforms.ToTensor(),
])

class DownUpSample:
    def __init__(self, down_size=8, original=32):
        """
        down_size: size to downsample to (e.g., 8x8)
        """
        self.down_size = down_size
        self.original = original

    def __call__(self, x):
        """
        tensor: output of To Tensor
        Returns: tensor [C,H,W] after downsampling and upsampling
        """

        # Downsample to Upsample
        x = x.unsqueeze(0)
        x = F.interpolate(x, size=(self.down_size, self.down_size), mode='bilinear', align_corners=False)
        x = F.interpolate(x, size=(self.original, self.original), mode='bilinear', align_corners=False)

        return x.squeeze(0)  # [C,H,W]

achille_blurry_transform_train = transforms.Compose([
    transforms.Pad(2), 
    transforms.RandomAffine(degrees=0, translate=(4/32, 4/32)),  # random translation
    transforms.RandomHorizontalFlip(),  # horizontal flip
    transforms.ToTensor(),
    DownUpSample(down_size=8),
])

# Load and process datasets
def MNIST_dataset(is_train:bool, transforms, data_dir='./artifacts/data'):
    return datasets.MNIST(root=data_dir, train=is_train, download=True, transform=transforms)


# Useful for sanity check. 
def plot_example_1D(dataset):
    """Plot the first 5 images and their labels in a row."""
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        img, label = dataset[i]
        img_np = img.permute(1, 2, 0).numpy().squeeze()
        axes[i].imshow(img_np, cmap='gray')
        axes[i].set_title(str(label))
        axes[i].axis('off')
    plt.show()

def save_dataset_examples(train_dataset, blurry_train_dataset, test_dataset, out_dir: Path, n: int = 5):
    """
    Save n example images from each dataset (train, blurry train, test) to visualize transforms.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets_info = [
        ("train", train_dataset),
        ("blurry_train", blurry_train_dataset),
        ("test", test_dataset),
    ]

    for name, dataset in datasets_info:
        fig, axes = plt.subplots(1, n, figsize=(n * 2, 2))
        for i in range(n):
            img, label = dataset[i]
            img_np = img.permute(1, 2, 0).numpy().squeeze()

            axes[i].imshow(img_np, cmap="gray")
            axes[i].set_title(str(label))
            axes[i].axis("off")

        fig.suptitle(f"{name} samples", fontsize=14)
        plt.tight_layout()
        save_path = out_dir / f"{name}_examples.png"
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Saved {n} {name} dataset input examples to {save_path}")