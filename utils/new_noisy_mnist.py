import math
import torch
import torchvision
from torchvision import transforms
from typing import Optional
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Subset
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import GaussianBlur


def fibonacci_sphere(samples=1000, add_test=False, margin = 0.05):

    points = {}
    phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

    i = 0

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        x = (x + 1)/2 * (1 - 2*margin) + margin # After moving into [0, 1]^3, scale back down by the margin value, then shift 
        y = (y + 1)/2 * (1 - 2*margin) + margin
        z = (z + 1)/2 * (1 - 2*margin) + margin

        points[i] = (x, y, z)
        i = i + 1
        
    if add_test:
        points["test"] = (1, 1, 1)

    return points

transform_3ch = transforms.Compose([
    transforms.Grayscale(num_output_channels=3), 
    transforms.ToTensor(),
    ])

COLOUR_MAP = fibonacci_sphere(10)

    
class NoisyColorMNIST(torch.utils.data.Dataset):
    def __init__(self,
                 mnist_dataset,
                 theta: Optional[float] = None,
                 colour_map = COLOUR_MAP,
                 colour_noise_std=0,
                 R_input_noise_std=0,
                 G_input_noise_std=0,
                 B_input_noise_std=0,
                 preprocess_image_transform = transform_3ch,
                 channel_transforms: dict = {}  # New: {'R': func, 'G': func, 'B': func}
                 ):
        
        # Original dataset
        self.mnist = mnist_dataset
        self.transform = preprocess_image_transform # We can add additional transform 

        # Colour spurrious params
        self.theta = theta
        self.shape_colour_map = colour_map
        self.RGB_mapping_tensor = {
            k: torch.tensor(v, dtype=torch.float32).view(3,1,1)
            for k,v in self.shape_colour_map.items()
            }
        
        # Precalculate aspects of the probability draw
        self.num_classes = 10
        self.uniform_probs = torch.ones(self.num_classes) / self.num_classes


        # Noise Hyper Params
        self.colour_noise_std= colour_noise_std
        self.channel_noise_std = torch.tensor(
        [R_input_noise_std, G_input_noise_std, B_input_noise_std]).view(3,1,1)

        # Channel transforms
        self.channel_transforms = channel_transforms if channel_transforms else {}
        self.channel_idx = {'R':0, 'G':1, 'B':2}
        self.channel_transforms_idx = {
            self.channel_idx[channel_name]: func
            for channel_name, func in self.channel_transforms.items()
            }
        
        # Prep corr matrix
        self._precompute_colour_distributions()

    def __getitem__(self, index):
        '''
        Get an item from the dataset, applying the spurious colour logic
        Returns a tuple of (image, digit, colour)
        '''
        img, label = self.mnist[index]

        # Colour assignment
        probs = self.colour_probs[label]
        colour_int = Categorical(probs).sample().item()
        
        # Get colour map
        if self.colour_noise_std > 0:
            noise = torch.randn_like(self.RGB_mapping_tensor[colour_int]) * self.colour_noise_std
            MIN_VALUE = 0.001  # small nonzero floor to prevent (0, 0, 0)
            noisy_colour = torch.clamp(self.RGB_mapping_tensor[colour_int] + noise, min=MIN_VALUE, max=1.0)
        else:
            noisy_colour = self.color_map_tensor[colour_int]

        # Add channels and their transforms
        img_tensor = self.transform(img)
        for c_idx, func in self.channel_transforms_idx.items():
            img_tensor[c_idx] = func(img_tensor[c_idx])
        img_tensor = torch.clamp(img_tensor, 0.0, 1.0)

        # Transform greyscale image to RGB
        colorised_img = img_tensor * noisy_colour

        # Add input noise if needed. 
        if torch.any(self.channel_noise_std > 0):
            noise = torch.randn_like(colorised_img) * self.channel_noise_std
            colorised_img = torch.clamp(colorised_img + noise, 0, 1)

        return colorised_img, label
    
    def _precompute_colour_distributions(self):
        """Precompute probability vectors for each digit → colour mapping."""
        uniform = torch.ones(self.num_classes) / self.num_classes
        identity = torch.eye(self.num_classes)

        # θ * I + (1 - θ) * uniform
        probs = self.theta * identity + (1 - self.theta) * uniform

        # Check for a valid construction
        assert torch.allclose(
            probs.sum(dim=1), 
            torch.ones(self.num_classes), 
            atol=1e-5
        ), "Each row in colour probability matrix must sum to 1."
        
        self.colour_probs = probs

    def update_theta(self, new_theta):
        if not (0 <= new_theta <= 1):
            raise ValueError("theta must be between 0 and 1")
        self.theta = new_theta
        self._precompute_colour_distributions()

    def update_input_channel_noise(self, new_R_std=None, new_G_std=None, new_B_std=None):
        R = new_R_std if new_R_std is not None else self.channel_noise_std[0].item()
        G = new_G_std if new_G_std is not None else self.channel_noise_std[1].item()
        B = new_B_std if new_B_std is not None else self.channel_noise_std[2].item()

        self.channel_noise_std = torch.tensor([R, G, B]).view(3,1,1)
    
    def update_colour_noise(self, new_colour_noise_std):
        self.colour_noise_std=new_colour_noise_std
    
    def update_channel_transforms(self, new_transforms: dict):
        assert all(ch in ['R', 'G', 'B'] for ch in new_transforms.keys()), \
            "Channel keys must be 'R', 'G', or 'B'"

        self.channel_transforms = new_transforms

        # Precompute channel indices for efficiency
        self.channel_transforms_idx = {
            self.channel_idx[ch]: func
            for ch, func in self.channel_transforms.items()
        }

    def __len__(self):
        return len(self.mnist)
    
class SingleChannelColorMNIST(torch.utils.data.Dataset):
    """
    Wraps a NoisyColorMNIST dataset and zeroes all channels except the specified one.
    """
    def __init__(self, base_dataset, channel='R'):
        """
        Args:
            base_dataset: instance of NoisyColorMNIST
            channel: 'R', 'G', or 'B'
        """
        self.base_dataset = base_dataset
        assert channel in ['R', 'G', 'B'], "Channel must be 'R', 'G', or 'B'"
        self.channel = channel
        self.channel_idx = {'R':0, 'G':1, 'B':2}[channel]

    def __getitem__(self, index):
        img, label = self.base_dataset[index]  # img shape: [3,H,W]
        mask = torch.zeros_like(img)
        mask[self.channel_idx] = 1.0
        img = img * mask
        return img, label

    def __len__(self):
        return len(self.base_dataset)
    
def plot_samples(dataset, samples_per_class=5, title=None, figsize=(8, 8)):
    """
    Plots samples_per_class images per digit class from a dataset
    that returns (image, label) pairs.
    
    Args:
        dataset: PyTorch dataset returning (img, label)
        samples_per_class: number of samples to show per class
        title: optional figure title
        figsize: size of the matplotlib figure
    """
    # Collect samples
    samples = {i: [] for i in range(10)}  # assuming digits 0–9
    for i, (img, label) in enumerate(dataset):
        if len(samples[label]) < samples_per_class:
            samples[label].append(img)
        if all(len(v) == samples_per_class for v in samples.values()):
            break

    # Plot
    fig, axes = plt.subplots(10, samples_per_class, figsize=figsize)
    if title:
        fig.suptitle(title, fontsize=14)

    for cls in range(10):
        for i in range(samples_per_class):
            ax = axes[cls, i] if samples_per_class > 1 else axes[cls]
            img = samples[cls][i]
            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0)  # Convert C,H,W → H,W,C
            ax.imshow(img)
            ax.axis("off")
            if i == 0:
                ax.set_ylabel(str(cls), rotation=0, labelpad=15, fontsize=12)

    plt.tight_layout()
    plt.show()

def balanced_subset(dataset, total_samples):
    """
    Returns a class-balanced subset of a PyTorch dataset.

    Args:
        dataset: PyTorch dataset with `.targets` attribute (e.g., MNIST)
        total_samples: total number of samples in subset

    Returns:
        Subset(dataset, indices)
    """
    labels = dataset.targets
    num_classes = len(torch.unique(labels))
    samples_per_class = total_samples // num_classes

    subset_indices = []

    for c in torch.unique(labels):
        class_mask = (labels == c)
        class_indices = torch.nonzero(class_mask, as_tuple=False).squeeze().tolist()
        # Randomly sample
        sampled = torch.tensor(class_indices)[torch.randperm(len(class_indices))[:samples_per_class]]
        subset_indices.append(sampled)

    # Concatenate all indices
    subset_indices = torch.cat(subset_indices).tolist()

    return Subset(dataset, subset_indices)


def make_flip_channel(prob=0.5):
    """
    Returns a function that flips a channel horizontally with probability `prob`.
    """
    def flip(tensor):
        if torch.rand(1).item() < prob:
            return TF.hflip(tensor)
        return tensor
    return flip


def make_blur_channel(kernel_size=3, sigma=(0.1,2.0)):
    """
    Returns a function that applies Gaussian blur to a channel.
    Uses PyTorch tensors directly for efficiency.
    """
    blur = GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    def blur_func(tensor):
        # tensor shape: [H,W], add dummy channel dim for blur
        t = tensor.unsqueeze(0)
        t = blur(t)
        return t.squeeze(0)
    return blur_func

def make_swap_channel_with_dataset(dataset, channel, diff_label=True):
    """
    Returns a function that swaps channels with other images efficiently.

    Args:
        dataset: PyTorch dataset with .targets
        diff_label: if True, swap with image of diff label; else random
        channel: from which channel to sample from
    """
    targets = dataset.targets
    num_samples = len(dataset)
    num_classes = len(torch.unique(targets))

    # Precompute label -> indices mapping
    label_to_indices = {int(c): (targets == c).nonzero(as_tuple=True)[0] for c in range(num_classes)}
    diff_label_indices_dict = {
        label: torch.cat([idx for current_label, idx in label_to_indices.items() if current_label != label])
        for label in range(num_classes)
    }

    # Flatten indices for random label swap
    all_indices = torch.arange(num_samples)

    if diff_label:
        def swap_func(tensor, label):
            indices = diff_label_indices_dict[int(label)]
            idx = indices[torch.randint(len(indices), (1,)).item()]
            replacement_img, _ = dataset[idx]

            return replacement_img[channel]
    
    else:
        def swap_func(img_tensor, label):
            idx = all_indices[torch.randint(len(all_indices), (1,)).item()]
            replacement_img, _ = dataset[idx]

            return replacement_img[channel]

    return swap_func

def make_permute_channel(shape, perm=None):
    """
    Returns a function that permutes all pixels in a channel tensor [H,W].
    Permutation is precomputed once for efficiency.

    Args:
        shape: tuple of (H,W) of the channel
        seed: optional int for reproducibility
    """
    numel = shape[0] * shape[1]

    if perm is not None and perm.numel() != numel:
        raise ValueError(f"Provided perm has {perm.numel()} elements, expected {numel}")

    if perm is None:
        g = torch.Generator()
        perm = torch.randperm(numel, generator=g)

    def permute_func(tensor):
        return tensor.flatten()[perm].view_as(tensor)

    return permute_func

def random_subset(dataset, total_samples=3000, seed=42):
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=g)[:total_samples]
    return Subset(dataset, indices)