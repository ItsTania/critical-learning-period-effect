import os
import sys
import gc
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Subset
import torchvision
from torchvision import transforms

from tqdm import trange
import pandas as pd
import numpy as np

import skorch 
from skorch import NeuralNetClassifier 
from skorch.callbacks import ProgressBar

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.models.mlp import BasicClassifierModule, BottleneckClassifierModule 
from utils.models.achille import get_activation
from utils.callbacks import SaveModelInformationCallback, get_all_test_callbacks


# Experiment params - general
NUMBER_RUNS = 1
SKIP_BASELINE = False
DATALOADER_NUM_WORKERS = 4


SOURCE_ROTATIONS = [2.5, 5, 10, 20, 40]
TARGET_ROTATION = 0
EVAL_ROTATIONS = [360 - x for x in SOURCE_ROTATIONS]

if torch.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

MODEL = BottleneckClassifierModule
ACTIVATION = get_activation('relu')
PRETRAINING_EPOCHS = 2#500
CLEAN_EPOCHS = 2#100
LEARNING_RATE = 0.005
BATCH = 128
OPTIMIZER = torch.optim.Adam
CRITERION = torch.nn.CrossEntropyLoss


def get_rotation_transform(degrees: float):
    """Return a transform that rotates the MNIST image by `degrees` and converts to tensor."""
    return transforms.Compose([
        transforms.RandomRotation((degrees, degrees)),  # fixed angle
        transforms.ToTensor()
    ])

def train_MNIST_models_from_random_init(
        train_dataset, 
        test_datasets: List[Tuple[str, skorch.dataset.Dataset]], 
        logging_dir: Path, 
        num_epochs:int=CLEAN_EPOCHS, 
        dataset_classes=list(range(10)), 
        input_dim=784):
    """Train the baseline models from random initialisation."""
    list_of_model_histories: list[Path] = []
    for run in trange(NUMBER_RUNS, desc="Random Init Runs"):
        logging_dir_run = logging_dir / f"run_{run}"

        test_callbacks = get_all_test_callbacks(test_datasets=test_datasets, logging_dir_run=logging_dir_run)
        callbacks = [
            SaveModelInformationCallback(save_dir=str(logging_dir_run)),
            ProgressBar(),
            *test_callbacks,
        ]
        
        net = NeuralNetClassifier(
            module=MODEL,
            lr=LEARNING_RATE,
            optimizer=OPTIMIZER,
            criterion=CRITERION,
            device=DEVICE,
            callbacks=callbacks,
            train_split=None,
            classes=dataset_classes,
            module__activation=ACTIVATION,
            module__input_dim=input_dim,
            iterator_train__num_workers=DATALOADER_NUM_WORKERS,
            iterator_train__shuffle=True
        )

        # Start training
        net.fit(train_dataset, y=None, epochs=num_epochs)

        # Save history. 
        df = pd.DataFrame(net.history)
        df['run'] = run
        df.to_csv(str(logging_dir_run / "net_history.csv"))
        list_of_model_histories.append(logging_dir_run / "net_history.csv")

    return list_of_model_histories


def pretrain_MNIST_models(
        train_dataset, 
        test_datasets: List[Tuple[str, skorch.dataset.Dataset]], 
        logging_dir: Path, 
        num_epochs:int=PRETRAINING_EPOCHS, 
        dataset_classes=list(range(10)), 
        input_dim=784):
    ''' Train the source models on the 'degraded' training conditions.'''
    list_of_model_files: list[Path] = []
    list_of_model_histories: list[Path] = []

    for run in trange(NUMBER_RUNS, desc="Source Pretraining Runs"):
        logging_dir_run = logging_dir / f"run_{run}"

        test_callbacks = get_all_test_callbacks(test_datasets=test_datasets, logging_dir_run=logging_dir_run)
        callbacks = [
            SaveModelInformationCallback(save_dir=str(logging_dir_run)),
            ProgressBar(),
            *test_callbacks,
        ]
            
        net = NeuralNetClassifier(
            module=MODEL,
            lr=LEARNING_RATE,
            optimizer=OPTIMIZER,
            criterion=CRITERION,
            device=DEVICE,
            callbacks=callbacks,
            train_split=None,
            classes=dataset_classes,
            module__activation=ACTIVATION,
            module__input_dim=input_dim,
            iterator_train__num_workers=DATALOADER_NUM_WORKERS,
            iterator_train__shuffle=True
        )

        # Start Training.
        net.fit(train_dataset, y=None, epochs=num_epochs)

        # Save history.
        df = pd.DataFrame(net.history)
        df['run'] = run
        df.to_csv(str(logging_dir_run / "net_history.csv"))
        list_of_model_histories.append(logging_dir_run / "net_history.csv")
        
        # Save model
        model_path = logging_dir_run / 'pretrained_model_weights.pt'
        print(f"Saving model parameters to {str(model_path)}")
        torch.save(net.module_.state_dict(), str(model_path))
        list_of_model_files.append(model_path)

    return list_of_model_histories, list_of_model_files


def train_MNIST_model_from_pretrained_init(
        run, 
        pretrained_weights_fp: Path, 
        train_dataset, 
        test_datasets: List[Tuple[str, skorch.dataset.Dataset]],
        logging_dir: Path, 
        num_epochs:int=CLEAN_EPOCHS, 
        dataset_classes=list(range(10)),
        input_dim=784):
    logging_dir_run = logging_dir / f"run_{run}"

    test_callbacks = get_all_test_callbacks(test_datasets=test_datasets, logging_dir_run=logging_dir_run)
    callbacks = [
        SaveModelInformationCallback(save_dir=str(logging_dir_run)),
        ProgressBar(),
        *test_callbacks,
    ]
        
    net = NeuralNetClassifier(
        module=MODEL,
        lr=LEARNING_RATE,
        optimizer=OPTIMIZER,
        criterion=CRITERION,
        device=DEVICE,
        warm_start=True,
        callbacks=callbacks,
        train_split=None,
        classes=dataset_classes,
        module__activation=ACTIVATION,
        module__input_dim=input_dim,
        iterator_train__num_workers=DATALOADER_NUM_WORKERS,
        iterator_train__shuffle=True
    )
    net.initialize()
    print(f"Loading from {str(pretrained_weights_fp)}")
    state_dict = torch.load(str(pretrained_weights_fp), map_location=net.device)
    net.module_.load_state_dict(state_dict)
    net.fit(train_dataset, y=None, epochs=num_epochs)

    # Save history. 
    df = pd.DataFrame(net.history)
    df['run'] = run
    df.to_csv(str(logging_dir_run / "net_history.csv"))
    return logging_dir_run / "net_history.csv"


def set_up_test_dataset(dataset) -> skorch.dataset.Dataset:
    X_list, y_list = [], []
    for i in range(len(dataset)):
        x, y = dataset[i]
        X_list.append(x)
        y_list.append(y)
    X_tensor = torch.stack(X_list)
    y_tensor = torch.tensor(y_list, dtype=torch.long)
    test_dataset = skorch.dataset.Dataset(X_tensor, y_tensor)
    test_dataset.targets = y_tensor
    return test_dataset

if __name__ == "__main__":

    for SOURCE_ROTATION in SOURCE_ROTATIONS:
        EXPERIMENT_DIR = ROOT / Path(
        f"artifacts/experiment_results/rotationMNIST_source{SOURCE_ROTATION}_target{TARGET_ROTATION}"
    )
        print(f"Starting rotation experiment run {EXPERIMENT_DIR}!")
        randominit_logging_dir = EXPERIMENT_DIR / 'target_w_random_init'
        pretrained_models_logging_dir = EXPERIMENT_DIR / 'pretraining_on_source'
        noisyinit_logging_dir = EXPERIMENT_DIR / "target_w_source_init"
        os.makedirs(randominit_logging_dir, exist_ok=True)
        os.makedirs(pretrained_models_logging_dir, exist_ok=True)
        os.makedirs(noisyinit_logging_dir, exist_ok=True)

        # Load MNIST datasets
        data_dir = ROOT / "artifacts" / "data"
        hard_indices = np.load(os.path.join(data_dir, "hard_indices.npy"))
        complementary_indices = np.load(os.path.join(data_dir, "new_train_indices.npy"))

        # Apply rotation transforms - source
        original_source_train_dataset = torchvision.datasets.MNIST(
            data_dir, train=True, download=True, transform=get_rotation_transform(SOURCE_ROTATION)
        )
        source_train_dataset = Subset(original_source_train_dataset, complementary_indices)
        source_MNIST_hard_subset = Subset(original_source_train_dataset, hard_indices)
        
        # Apply rotation transforms - target
        original_target_train_dataset = torchvision.datasets.MNIST(
            data_dir, train=True, download=True, transform=get_rotation_transform(TARGET_ROTATION)
        )
        target_train_dataset = Subset(original_target_train_dataset, complementary_indices)
        target_MNIST_hard_subset = Subset(original_target_train_dataset, hard_indices)

        target_MNIST_test = torchvision.datasets.MNIST(
            data_dir, train=False, download=True, transform=get_rotation_transform(TARGET_ROTATION)
        )

        # Determine input dimension dynamically
        x0, y0 = target_train_dataset[0]
        input_dim = x0.numel()

        # Prepare test datasets
        test_datasets = [
            ("MNIST_test_target", set_up_test_dataset(target_MNIST_test)),
            ("MNIST_hard_target", set_up_test_dataset(target_MNIST_hard_subset)),
            ("MNIST_hard_source", set_up_test_dataset(source_MNIST_hard_subset)),
        ]

        for eval_rotation in EVAL_ROTATIONS: 
            tmp = torchvision.datasets.MNIST(
                data_dir, train=True, download=True, transform=get_rotation_transform(eval_rotation))
            tmp_MNIST_hard_subset = Subset(original_target_train_dataset, hard_indices)
            test_datasets.append((f"MNIST_hard_{eval_rotation}", set_up_test_dataset(source_MNIST_hard_subset)))

        # ---------------- Baseline ----------------
        if not SKIP_BASELINE:
            random_init_model_histories = train_MNIST_models_from_random_init(
                train_dataset=target_train_dataset, 
                logging_dir=randominit_logging_dir,
                test_datasets=test_datasets,
                input_dim=input_dim
            )
            gc.collect()

        # ---------------- Pretraining ----------------
        pretrain_model_histories, pretrain_model_params = pretrain_MNIST_models(
            train_dataset=source_train_dataset,
            logging_dir=pretrained_models_logging_dir,
            test_datasets=test_datasets,
            input_dim=input_dim
        )
        gc.collect()

        # ---------------- Fine-tune from pretrained ----------------
        model_history_noisy_init = []
        for run, model_params in enumerate(pretrain_model_params):
            net_history = train_MNIST_model_from_pretrained_init(
                run=run,
                train_dataset=target_train_dataset, 
                pretrained_weights_fp=model_params, 
                logging_dir=noisyinit_logging_dir,
                test_datasets=test_datasets,
                input_dim=input_dim
            )
            model_history_noisy_init.append(net_history)
