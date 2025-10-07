import sys
from pathlib import Path
from typing import Optional, List
import torch

import pandas as pd

from skorch import NeuralNetClassifier
from skorch.callbacks import ProgressBar

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.models.achille import Achille_MNIST_FC, Achille_MNIST_FC_No_BatchNorm, get_activation  # noqa: E402, F401
from utils.models.mlp import BasicClassifierModule, BottleneckClassifierModule # noqa: E402
from utils.models.baselines import LogisticRegressionModule # noqa: E402
from utils.models.cnn import CNN # noqa: E402
from utils.callbacks import SaveModelInformationCallback, get_all_test_callbacks  # noqa: E402
from utils.data import MNIST_dataset, achille_preprocess, achille_transform_train, achille_blurry_transform_train, save_dataset_examples  # noqa: E402

from experiments.base_experiment import BaseExperiment, set_up_test_dataset, combine_experiment_histories  # noqa: E402

EXPERIMENT_DIR = Path("artifacts/experiment_results/achille_repl_test")
DATALOADER_NUM_WORKERS = 4

# Experiment configurations
NUM_RUNS = 1 #10
TARGET_EPOCHS = 1 #180
SOURCE_EPOCHS = 1 #480


class AchilleExperiment(BaseExperiment):
    """
    Replicates Achille et al. (2019) MNIST blurry pretraining experiments.
    Allows testing different model architectures and hyperparameters.
    """

    def __init__(
        self,
        model_cls,
        experiment_dir: Path,
        activation: str = "relu",
        num_runs: int = 3,
        device: Optional[str] = None,
        learning_rate: float = 0.005,
        batch_size: int = 128,
        optimizer_cls = torch.optim.Adam,
        criterion_cls = torch.nn.CrossEntropyLoss,
        **nn_kwargs
    ):
        super().__init__(experiment_dir, num_runs=num_runs, device=device)
        self.model_cls = model_cls
        self.activation_fn = activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer_cls = optimizer_cls
        self.criterion_cls = criterion_cls

        # will be set in prepare_datasets
        self.test_loader = None

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
        )

        self.nn_kwargs = nn_kwargs

    # ----------------------------------------------------------------------
    # Dataset setup
    # ----------------------------------------------------------------------
    def prepare_datasets(self):
        data_dir = self.data_dir

        print(f"Loading Achille MNIST datasets from {data_dir}...")

        full_train_dataset = MNIST_dataset(
            is_train=True,
            transforms=achille_transform_train,
            data_dir=data_dir,
        )

        full_source_dataset = MNIST_dataset(
            is_train=True,
            transforms=achille_blurry_transform_train,
            data_dir=data_dir,
        )

        test_dataset = MNIST_dataset(
            is_train=False,
            transforms=achille_preprocess,
            data_dir=data_dir,
        )

        print(f"Constructing Hard MNIST Set Up {data_dir}...")

        self.target_dataset, target_MNIST_Hard = self.construct_hard_subset(full_train_dataset)
        self.source_dataset, source_MNIST_Hard = self.construct_hard_subset(full_source_dataset)

        # Ensure the right datasets are being assigned
        assert len(self.source_dataset) > len(source_MNIST_Hard)
        assert len(self.source_dataset) == len(self.target_dataset)

        self.test_datasets = [
            ("Achille_MNIST_test_target", set_up_test_dataset(test_dataset)),
            ("Achille_MNIST_hard_target", set_up_test_dataset(target_MNIST_Hard)),
            ("Achille_MNIST_hard_source", set_up_test_dataset(source_MNIST_Hard)),
            ]

        x0, _ = self.target_dataset[0]
        self.input_dim = x0.numel()
        self.dataset_classes=list(range(10))

        save_dataset_examples(self.target_dataset, self.source_dataset, test_dataset, self.experiment_dir)

    def build_model(self, finetuning: bool, logging_dir: Path) -> NeuralNetClassifier:
        # Set all the callbacks
        assert self.test_datasets is not None, "Please run prepare datasets first."
        test_callbacks = get_all_test_callbacks(test_datasets=self.test_datasets, logging_dir_run=logging_dir)
        callbacks = [
            SaveModelInformationCallback(save_dir=str(logging_dir)),
            ProgressBar(),
            *test_callbacks
        ]

        # Return model with correct configs set at the beginning of the experiment
        return NeuralNetClassifier(
            module=self.model_cls,
            lr=self.learning_rate,
            optimizer=self.optimizer_cls,
            criterion=self.criterion_cls,
            device=self.device,
            callbacks=callbacks,
            train_split=None,
            classes=self.dataset_classes,
            module__activation=self.activation_fn,
            module__input_dim=self.input_dim,
            iterator_train__num_workers=DATALOADER_NUM_WORKERS,
            iterator_train__pin_memory=True,
            warm_start=finetuning,  # important for finetuning
            **self.nn_kwargs
        )

    def run_full_achille(self, source_epochs:int, target_epochs:int, skip_baseline=False):
        print(f"Starting Achille replication experiment in {self.experiment_dir}")
        self.prepare_datasets()

        if not skip_baseline:
            print("Training target models from random init...")
            baseline_hist = self.run_baseline(target_epochs)

        print("Pretraining on blurry data...")
        pre_hist, ckpts = self.run_pretraining(source_epochs)

        print("Training target models from pretrained blurry checkpoints...")
        ft_hist = self.run_finetuning(ckpts, target_epochs)

        print("Achille replication experiment completed.")

        # Combine histories into a single summary DataFrame
        try:
            all_hist_fp = baseline_hist + pre_hist + ft_hist
            combine_experiment_histories(all_hist_fp, save_dir=self.experiment_dir)
        except:
            print("Failed to save summary df")
        return baseline_hist, pre_hist, ft_hist

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
            "run_name": "Original_Model",
            "model_cls": Achille_MNIST_FC,
        },
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
        {
            "run_name": "CNN",
            "model_cls": CNN,
        },
    ]

    for cfg in configs:
        experiment_dir = Path(EXPERIMENT_DIR) / str(cfg["run_name"])
        exp = AchilleExperiment(
            model_cls=cfg["model_cls"],
            experiment_dir=experiment_dir,
            num_runs=NUM_RUNS,
        )
        exp.run_full_achille(
            target_epochs=TARGET_EPOCHS,
            source_epochs=SOURCE_EPOCHS
        )