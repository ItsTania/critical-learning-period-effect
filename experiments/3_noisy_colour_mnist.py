import os
import gc
import sys
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Subset
import torchvision

import skorch
from skorch import NeuralNetClassifier
from skorch.callbacks import ProgressBar

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.models.mlp import BasicClassifierModule, BottleneckClassifierModule
from utils.models.baselines import LogisticRegressionModule # noqa: E402
from utils.models.cnn import CNN # noqa: E402

from utils.callbacks import SaveModelInformationCallback, get_all_test_callbacks
from utils.colour_mnist import NoisyColorMNIST, transform_3ch
from utils.data import save_dataset_examples_3ch

import pandas as pd
import numpy as np

from base_experiment import BaseExperiment, set_up_test_dataset, combine_experiment_histories, find_all_histories

NUM_RUNS=1#5
DATALOADER_NUM_WORKERS=4
SOURCE_THETA = 1 # 0 is random while 1 is spurrious
TARGET_THETA = 0.999
EVAL_THETA = 0
STD_COLOUR_NOISE=0.07

PRETRAINING_EPOCHS= 2#500 
CLEAN_EPOCHS=2#100
EXPERIMENT_DIR = Path(f"artifacts/experiment_results/ColourMNIST_source{SOURCE_THETA}_target{TARGET_THETA}")


class NoisyColorMNISTExperiment(BaseExperiment):
    """
    Experiment for NoisyColorMNIST variant:
    - Baseline training
    - Pretraining on source (blurry)
    - Fine-tuning on target with pretrained init
    """

    def __init__(
        self,
        model_cls,
        experiment_dir: Path,
        activation: str = "relu",
        source_theta: float = 1.0,
        target_theta: float = 0.999,
        eval_theta: float = 0.0,
        num_runs: int = 1,
        learning_rate: float = 0.005,
        batch_size: int = 128,
        device=None,
        optimizer_cls = torch.optim.Adam,
        criterion_cls = torch.nn.CrossEntropyLoss,
        **nn_kwargs
    ):
        super().__init__(experiment_dir, num_runs=num_runs, device=device)
        self.source_theta = source_theta
        self.target_theta = target_theta
        self.eval_theta = eval_theta

        self.model_class = model_cls
        self.activation = activation
        self.lr = learning_rate
        self.batch_size = batch_size
        self.optimizer_cls = optimizer_cls
        self.criterion_cls = criterion_cls

        # save config
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
            PRETRAINING_EPOCHS=PRETRAINING_EPOCHS
        )

        self.nn_kwargs = nn_kwargs


    def prepare_datasets(self):
        # Load MNIST and split into MNIST Hard
        full_train_dataset = torchvision.datasets.MNIST(
            self.data_dir, 
            train=True, 
            download=True
            )
        train_subset, hard_subset = self.construct_hard_subset(full_train_dataset)
        
        assert len(train_subset) + len(hard_subset) == len(full_train_dataset)

        # ColorMNIST train datasets
        self.source_dataset = NoisyColorMNIST(
            train_subset, 
            theta=self.source_theta, 
            colour_noise_std=STD_COLOUR_NOISE
            )
        self.target_dataset = NoisyColorMNIST(
            train_subset, 
            theta=self.target_theta, 
            colour_noise_std=STD_COLOUR_NOISE
            )

        # Gray hard
        mnist_train_3ch = torchvision.datasets.MNIST(
            self.data_dir, 
            train=True, 
            download=True, 
            transform=transform_3ch
            )
        _ , gray_hard_subset = self.construct_hard_subset(mnist_train_3ch)

        # ColourMNIST Test datasets
        mnist_test = torchvision.datasets.MNIST(
            self.data_dir, 
            train=False, 
            download=True)
        test_target = set_up_test_dataset(
            NoisyColorMNIST(mnist_test, theta=self.target_theta, colour_noise_std=STD_COLOUR_NOISE)
            )
        test_hard_target = set_up_test_dataset(
            NoisyColorMNIST(hard_subset, theta=self.target_theta, colour_noise_std=STD_COLOUR_NOISE)
            )
        test_hard_eval = set_up_test_dataset(
            NoisyColorMNIST(hard_subset, theta=self.eval_theta, colour_noise_std=STD_COLOUR_NOISE)
            )
        test_hard_gray = set_up_test_dataset(
            gray_hard_subset
            )

        self.test_datasets = [
            ("MNIST_test_target", test_target),
            ("MNIST_hard_target", test_hard_target),
            ("MNIST_hard_eval", test_hard_eval),
            ("MNIST_hard_gray", test_hard_gray),
        ]

        # Input dimension
        x0, y0 = self.target_dataset[0]
        self.input_dim = x0.numel()
        self.dataset_classes=list(range(10))

        # Save example images
        save_dataset_examples_3ch(self.target_dataset, self.source_dataset, test_target, self.experiment_dir)

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
            iterator_train__num_workers=DATALOADER_NUM_WORKERS,
            iterator_train__shuffle=True,
            iterator_train__pin_memory=True,
            warm_start=finetuning,
            **self.nn_kwargs
        )


if __name__ == "__main__":
        # Check if EXPERIMENT_DIR exists, if so, add an integer suffix
    base_dir = EXPERIMENT_DIR
    exp_dir = base_dir
    i = 1
    while exp_dir.exists():
        exp_dir = Path(f"{base_dir}_{i}")
        i += 1
    EXPERIMENT_DIR = exp_dir

    # Define the configs we want to alter, changing only the model_cls for each row
    configs = [
        {
            "run_name": "MLP_w_depth_3",
            "model_cls": BasicClassifierModule,
        },
        {
            "run_name": "Bottleneck_w_width_3",
            "model_cls": BottleneckClassifierModule,
        },
        {
            "run_name": "LogisticRegression",
            "model_cls": LogisticRegressionModule,
        },
    ]

    for cfg in configs:
        experiment_dir = Path(EXPERIMENT_DIR) / str(cfg["run_name"])
        exp = NoisyColorMNISTExperiment(
            model_cls=cfg["model_cls"],
            experiment_dir=experiment_dir,
            num_runs=NUM_RUNS,
            source_theta=SOURCE_THETA,
            target_theta=TARGET_THETA
            )
        
        exp.run_full(
            target_epochs=CLEAN_EPOCHS,
            source_epochs=PRETRAINING_EPOCHS,
            skip_baseline=False
        )

    # Combine histories
    all_histories = find_all_histories(str(EXPERIMENT_DIR))
    combined_df = combine_experiment_histories(all_histories, save_dir=EXPERIMENT_DIR)
    print(combined_df.head())
