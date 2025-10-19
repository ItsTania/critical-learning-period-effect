import os
import gc
import sys
import itertools
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Subset
import torchvision

import skorch
from skorch import NeuralNetClassifier
from skorch.callbacks import ProgressBar

ROOT=Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.models.mlp import BasicClassifierModule
from utils.models.achille import Achille_MNIST_FC_No_BatchNorm

from utils.callbacks import SaveModelInformationCallback, get_all_test_callbacks
from utils.new_noisy_mnist import NoisyColorMNIST, transform_3ch, SingleChannelColorMNIST, balanced_subset, random_subset
from utils.data import save_dataset_examples_3ch

import pandas as pd
import numpy as np

from experiments.base_experiment import BaseExperiment, set_up_test_dataset, combine_experiment_histories, find_all_histories

NUM_RUNS=1

DATALOADER_NUM_WORKERS=4
PREFETCH=4

SOURCE_THETA = 0 # 0 is random while 1 is spurrious
TARGET_THETA = 0
EVAL_THETA = 0

PRETRAINING_EPOCHS= 0#500 
CLEAN_EPOCHS=1#100
EXPERIMENT_DIR = Path(f"artifacts/experiment_results/NoisyColourMNIST_gridsearch_target{TARGET_THETA}")

class ColorMNISTGridExperiment(BaseExperiment):
    """
    Experiment class for running a grid search on NoisyColorMNIST datasets.
    Only baseline training is run (no pretraining or fine-tuning).
    """

    def __init__(
            self, 
            model_cls,
            experiment_dir: Path, 
            theta: float = 0.0,
            input_noise=(0, 0, 0),
            colour_noise: float = 0.0,
            activation: str = "relu",
            num_runs: int = 1,
            device=None,
            optimizer_cls = torch.optim.Adam,
            criterion_cls = torch.nn.CrossEntropyLoss,
            input_channels = 1,
            learning_rate: float = 0.005,
            batch_size: int = 128,
            ):
        super().__init__(experiment_dir, num_runs=num_runs, device=device)

        self.theta = theta

        self.input_noise = input_noise
        self.colour_noise = colour_noise

        self.model_class = model_cls
        self.activation = activation
        self.lr = learning_rate
        self.batch_size = batch_size
        self.optimizer_cls = optimizer_cls
        self.criterion_cls = criterion_cls
        self.input_channels = input_channels
        



    def prepare_datasets(self):
        """Prepare the NoisyColorMNIST dataset for the grid search."""
        full_train_dataset = torchvision.datasets.MNIST(
            self.data_dir, 
            train=True, 
            download=True
            )
        
        ## Split to get the train split
        train_subset, hard_subset = self.construct_hard_subset(full_train_dataset)


        ## Make the train dataset
        self.target_dataset = NoisyColorMNIST(
            mnist_dataset=train_subset,
            theta=self.theta,
            colour_noise_std=self.colour_noise,
            R_input_noise_std=self.input_noise[0],
            G_input_noise_std=self.input_noise[1],
            B_input_noise_std=self.input_noise[2],
            preprocess_image_transform=transform_3ch
        )

        self.source_dataset = None

        ## Make the test dataset - gray
        MNIST_hard_target = NoisyColorMNIST(
            mnist_dataset=hard_subset,
            theta=self.theta,
            colour_noise_std=self.colour_noise,
            R_input_noise_std=self.input_noise[0],
            G_input_noise_std=self.input_noise[1],
            B_input_noise_std=self.input_noise[2],
            preprocess_image_transform=transform_3ch
        )

        mnist_train_3ch = torchvision.datasets.MNIST(
            self.data_dir, 
            train=True, 
            download=True, 
            transform=transform_3ch
            )
        _ , gray_hard_subset = self.construct_hard_subset(mnist_train_3ch)

        gray_mnist_test = torchvision.datasets.MNIST(
            self.data_dir, 
            train=False, 
            download=True, 
            transform=transform_3ch
            )

        MNIST_test_gray_3k = Subset(gray_mnist_test, torch.arange(3000))

        ## Make the test dataset - 1 channel
        mnist_test = torchvision.datasets.MNIST(
            self.data_dir, 
            train=False, 
            download=True)
        full_test = NoisyColorMNIST(
            mnist_test, 
            theta=self.theta, 
            colour_noise_std=self.colour_noise)
        subset_test = Subset(full_test, torch.arange(1000))
        R_channel_test = SingleChannelColorMNIST(subset_test, channel='R')
        G_channel_test = SingleChannelColorMNIST(subset_test, channel='G')
        B_channel_test = SingleChannelColorMNIST(subset_test, channel='B')

        MNIST_test_gray_1k = Subset(gray_mnist_test, torch.arange(1000))

        R_test = SingleChannelColorMNIST(MNIST_test_gray_1k, channel='R')
        G_test = SingleChannelColorMNIST(MNIST_test_gray_1k, channel='G')
        B_test = SingleChannelColorMNIST(MNIST_test_gray_1k, channel='B')

        self.test_datasets = [
            ("MNIST_test_target", set_up_test_dataset(full_test)),
            ("MNIST_hard_target", set_up_test_dataset(MNIST_hard_target)),
            ("MNIST_hard_gray", set_up_test_dataset(gray_hard_subset)),
            ("MNIST_subset_test_gray", set_up_test_dataset(MNIST_test_gray_3k)),
            ("MNIST_test_gray_R", set_up_test_dataset(R_test)),
            ("MNIST_test_gray_G", set_up_test_dataset(G_test)),
            ("MNIST_test_gray_B", set_up_test_dataset(B_test)),
            ("MNIST_test_R", set_up_test_dataset(R_channel_test)),
            ("MNIST_test_G", set_up_test_dataset(G_channel_test)),
            ("MNIST_test_B", set_up_test_dataset(B_channel_test)),
        ]

        # Set input dimension
        x0, y0 = self.target_dataset[0]
        self.input_dim = x0.numel()
        self.dataset_classes=list(range(10))

        # Save example images
        # save_dataset_examples_3ch(self.target_dataset, gray_hard_subset, full_test, self.experiment_dir)

    def build_model(self, finetuning: bool, logging_dir: Path) -> NeuralNetClassifier:
        # Get Callbacks
        assert self.test_datasets is not None, "Please run prepare datasets first."
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
            iterator_train__prefetch_factor=PREFETCH,
            iterator_train__persistent_workers=True,
            warm_start=finetuning,
        )

    def run_single_run(self, epochs: int = CLEAN_EPOCHS):
        """
        Run baseline training for this hyperparameter combination.
        """
        print(f"Starting experiment in {self.experiment_dir}")
        self.prepare_datasets()
        return self.run_baseline(epochs=epochs)
    

if __name__ == "__main__":
    # Check if EXPERIMENT_DIR exists, if so, add an integer suffix
    base_dir = EXPERIMENT_DIR
    exp_dir = base_dir
    i = 1
    while exp_dir.exists():
        exp_dir = Path(f"{base_dir}_{i}")
        i += 1
    EXPERIMENT_DIR = exp_dir
    del exp_dir

    # Get hyperparameters to search through

    thetas = [0.9, 0.95, 0.99, 0.995]
    input_noises = [(0.3, 0.3, 0.3)] #[(i,i,i) for i in [0.0, 0.1, 0.3, 0.4, 0.5]]
    colour_noises = [0.1] #[0.01, 0.05, 0.1, 0.5]
    models = [("MLP", BasicClassifierModule)] #[("MLP", BasicClassifierModule), ("AchilleNoBatchnorm", Achille_MNIST_FC_No_BatchNorm)]

    grid = list(itertools.product(input_noises, colour_noises, models, thetas))

    # Go through it
    for inp_noise, col_noise, (model_name, model), theta in grid:
        exp_name = f"{model_name}_theta{TARGET_THETA}_input{inp_noise}_colour{col_noise}"
        exp_run_dir = EXPERIMENT_DIR / Path(exp_name)

        exp = ColorMNISTGridExperiment(
            model_cls=model,
            experiment_dir=exp_run_dir,
            num_runs=NUM_RUNS,
            theta=theta,
            input_noise=inp_noise,
            colour_noise=col_noise,
            )

        exp._save_config(
            model=model_name,
            theta=theta,
            input_noise=inp_noise,
            colour_noise=col_noise
            )
        
        base_histories = exp.run_single_run(epochs=CLEAN_EPOCHS)

    all_histories = find_all_histories(str(EXPERIMENT_DIR))
    combined_df = combine_experiment_histories(all_histories, save_dir=EXPERIMENT_DIR)
    print(combined_df.head())
