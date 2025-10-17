import os
import gc
import sys
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Subset
import torchvision
from torchvision import transforms

import pandas as pd
import numpy as np

import skorch
from skorch import NeuralNetClassifier
from skorch.callbacks import ProgressBar

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.models.mlp import BasicClassifierModule, BottleneckClassifierModule
from utils.models.baselines import LogisticRegressionModule # noqa: E402

from utils.callbacks import SaveModelInformationCallback, get_all_test_callbacks
from base_experiment import BaseExperiment, set_up_test_dataset, combine_experiment_histories, find_all_histories


# ----------------- Global Config -----------------
NUM_RUNS = 1
DATALOADER_NUM_WORKERS = 4
PRETRAINING_EPOCHS = 1
CLEAN_EPOCHS = 1
EXPERIMENT_DIR = Path("artifacts/experiment_results/PermutedMNIST")

# -------------------------------------------------

class PermutePixels:
    """Transform that flattens an image and applies a fixed pixel permutation."""
    def __init__(self, permutation: torch.Tensor):
        self.permutation = permutation

    def __call__(self, img):
        x = transforms.ToTensor()(img).view(-1)      # Flatten to (784,)
        x = x[self.permutation]                      # Apply permutation
        return x


def get_permutation(input_dim: int = 28*28, seed=None):
    """Generate a fixed random permutation for pixels."""
    if seed is not None:
        rng = np.random.default_rng(seed)
        return torch.tensor(rng.permutation(input_dim), dtype=torch.long)
    else:
        return torch.randperm(input_dim)


def make_fractional_balanced_subset(dataset_subset: Subset, fraction: float, seed=None):
    """
    Create a class-balanced subset of the dataset at a given fraction of its original size,
    then repeat (upsample) samples to match the original dataset size.
    
    Args:
        dataset_subset (Subset): A torch Subset of a torchvision dataset.
        fraction (float): Fraction of the dataset to keep (e.g., 0.1 for 10%).
        seed (int): Random seed for reproducibility.
        
    Returns:
        indexes
    """
    if not (0 < fraction <= 1):
        raise ValueError("Fraction must be between 0 and 1.")
    
    if seed is None:
        seed = np.random.randint(0, 2**32 - 1)
    rng = np.random.default_rng(seed)
    
    original_indices = np.array(dataset_subset.indices)
    labels = np.array([dataset_subset.dataset.targets[i].item() for i in original_indices])

    num_classes = len(np.unique(labels))
    total_target_size = int(len(original_indices) * fraction)
    num_per_class = total_target_size // num_classes

    if num_per_class == 0:
        raise ValueError(f"Fraction {fraction} is too small — not enough samples per class.")

    # Step 1: pick balanced subset
    selected_indices = []
    for c in np.unique(labels):
        class_idx = original_indices[labels == c]
        if len(class_idx) < num_per_class:
            raise ValueError(f"Not enough samples in class {c}: only {len(class_idx)} available.")
        chosen = rng.choice(class_idx, size=num_per_class, replace=False)
        selected_indices.extend(chosen)

    selected_indices = np.array(selected_indices)

    # Step 2: Upsample back to original size
    upsampled_indices = rng.choice(selected_indices, size=len(original_indices), replace=True)

    return upsampled_indices



class PermutedMNISTExperiment(BaseExperiment):
    """
    Experiment for Permuted MNIST variant:
    - Baseline training on target
    - Pretraining on permuted source
    - Fine-tuning from pretrained init
    """

    def __init__(
        self,
        model_cls,
        experiment_dir: Path,
        num_runs: int = 1,
        learning_rate: float = 0.005,
        batch_size: int = 128,
        device=None,
        optimizer_cls = torch.optim.Adam,
        criterion_cls = torch.nn.CrossEntropyLoss,
        activation: str = "relu",
        input_channels = 1,
        fraction_trainset = 1,
        **nn_kwargs
    ):
        super().__init__(experiment_dir, num_runs=num_runs, device=device)
        self.model_class = model_cls
        self.activation = activation
        self.lr = learning_rate
        self.batch_size = batch_size
        self.optimizer_cls = optimizer_cls
        self.criterion_cls = criterion_cls
        self.input_channels = input_channels

        self.fraction = fraction_trainset

        # Save config for reproducibility
        self._save_config(
            model_cls=model_cls,
            activation=activation,
            learning_rate=learning_rate,
            batch_size=batch_size,
            optimizer_cls=optimizer_cls,
            criterion_cls=criterion_cls,
            num_runs=num_runs,
            device=device,
            CLEAN_EPOCHS=CLEAN_EPOCHS,
            PRETRAINING_EPOCHS=PRETRAINING_EPOCHS,
            fraction_trainset= fraction_trainset
        )

        self.nn_kwargs = nn_kwargs


    def prepare_datasets(self):
        # ---- Load MNIST ----
        original_train = torchvision.datasets.MNIST(
            self.data_dir, train=True, download=True, transform=transforms.ToTensor()
        )
        original_test = torchvision.datasets.MNIST(
            self.data_dir, train=False, download=True, transform=transforms.ToTensor()
        )

        # ---- Load in target (hard vs complementary) ----
        target_train_subset, target_hard_subset = self.construct_hard_subset(original_train)

        # ---- Load in permuted source (hard vs complementary)----
        input_dim = target_train_subset[0][0].numel()
        perm = get_permutation(input_dim)
        source_perm_dataset = torchvision.datasets.MNIST(
            self.data_dir, train=True, download=True, transform=PermutePixels(perm)
        )
        source_train_subset, source_hard_subset = self.construct_hard_subset(source_perm_dataset)
        
        ## Get a fraction of the dataset - and pad it out so that it is the same length.
        if self.fraction != 1:
            target = set_up_test_dataset(target_train_subset)
            source = set_up_test_dataset(source_train_subset)

            subset_index = make_fractional_balanced_subset(target, self.fraction)
            target_train_subset = Subset(target, subset_index)
            source_train_subset = Subset(source, subset_index)

        # Set Up datasets
        self.target_dataset = set_up_test_dataset(target_train_subset)
        self.source_dataset = set_up_test_dataset(source_train_subset)


        # ---- Test datasets ----
        self.test_datasets = [
            ("MNIST_test_target", set_up_test_dataset(original_test)),
            ("MNIST_hard_target", set_up_test_dataset(target_hard_subset)),
            ("MNIST_hard_source", set_up_test_dataset(source_hard_subset)),
        ]

        # ---- Misc ----
        self.input_dim = input_dim
        self.dataset_classes = list(range(10))

        


    def build_model(self, finetuning: bool, logging_dir: Path) -> NeuralNetClassifier:
        assert self.test_datasets is not None, "prepare_datasets() must be called first."
        test_callbacks = get_all_test_callbacks(test_datasets=self.test_datasets, logging_dir_run=logging_dir)
        callbacks = [
            SaveModelInformationCallback(save_dir=str(logging_dir)),
            ProgressBar(),
            *test_callbacks
        ]

        return NeuralNetClassifier(
            module=self.model_class,
            lr=self.lr,
            optimizer=self.optimizer_cls,
            criterion=self.criterion_cls,
            device=self.device,
            callbacks=callbacks,
            train_split=None,
            classes=self.dataset_classes,
            module__activation=self.activation,
            module__input_dim=self.input_dim,
            module__input_channels =self.input_channels,
            iterator_train__num_workers=DATALOADER_NUM_WORKERS,
            iterator_train__shuffle=True,
            iterator_train__pin_memory=True,
            warm_start=finetuning,
            **self.nn_kwargs
        )


# -------------------------------------------------
if __name__ == "__main__":
    # Make unique experiment dir if needed
    base_dir = EXPERIMENT_DIR
    exp_dir = base_dir
    i = 1
    while exp_dir.exists():
        exp_dir = Path(f"{base_dir}_{i}")
        i += 1
    EXPERIMENT_DIR = exp_dir

    configs = [
        {"run_name": "MLP_w_depth_3", "model_cls": BasicClassifierModule},
        {"run_name": "Bottleneck_w_width_3", "model_cls": BottleneckClassifierModule},
        {"run_name": "LogReg", "model_cls": LogisticRegressionModule},
    ]
    for percent in [0.25, 0.5, 0.75]:
        for cfg in configs:
            run_name = f'{cfg["run_name"]}_percentdata_{percent}'
            experiment_dir = EXPERIMENT_DIR / run_name
            exp = PermutedMNISTExperiment(
                model_cls=cfg["model_cls"],
                experiment_dir=experiment_dir,
                num_runs=NUM_RUNS,
                input_channels=None,
                fraction_trainset=percent
            )
            exp.run_full(
                target_epochs=CLEAN_EPOCHS,
                source_epochs=PRETRAINING_EPOCHS,
                skip_baseline=False
            )

        # Combine results
        all_histories = find_all_histories(str(EXPERIMENT_DIR))
        combined_df = combine_experiment_histories(all_histories, save_dir=EXPERIMENT_DIR)
        print(combined_df.head())
