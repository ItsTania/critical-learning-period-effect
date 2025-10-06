import os
import sys
import gc
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Subset
import torchvision

from tqdm import trange
import pandas as pd
import numpy as np

import skorch 
from skorch import NeuralNetClassifier 
from skorch.helper import predefined_split
from skorch.callbacks import ProgressBar

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.models.mlp import BasicClassifierModule, BottleneckClassifierModule 
from utils.models.achille import get_activation
from utils.callbacks import SaveModelInformationCallback, get_all_test_callbacks
from utils.colour_mnist import ColorMNIST

# Experiment params - general
NUMBER_RUNS = 2
EXPERIMENT_DIR = ROOT / Path("artifacts/experiment_results/multi-channel-example")
SKIP_BASELINE=False
DATALOADER_NUM_WORKERS=4

SOURCE_THETA = 1 # 0 is spurrious while 1 is random
TARGET_THETA = 0.999
EVAL_THETA = 0


if torch.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'


# Experiment params - taken from Achilles
MODEL=BasicClassifierModule
ACTIVATION=get_activation('relu')
PRETRAINING_EPOCHS= 2#500 
CLEAN_EPOCHS=2#50
LEARNING_RATE=0.005
BATCH=128
OPTIMIZER=torch.optim.Adam
CRITERION=torch.nn.CrossEntropyLoss



# Train baseline models
def train_MNIST_models_from_random_init(
        train_dataset, 
        test_datasets: List[Tuple[str, skorch.dataset.Dataset]], 
        logging_dir: Path, 
        num_epochs:int=CLEAN_EPOCHS, 
        dataset_classes=list(range(10)), 
        input_dim=784):
    ''' Train the basline models from random initialisation'''

    list_of_model_histories: list[Path] = []
    for run in trange(NUMBER_RUNS, desc="Random Init Runs"):
        logging_dir_run = logging_dir / f"run_{run}"
        
        test_callbacks = get_all_test_callbacks(test_datasets=test_datasets, logging_dir_run=logging_dir_run)
        callbacks = [
            SaveModelInformationCallback(save_dir=str(logging_dir_run)),
            ProgressBar(),
            *test_callbacks,]
        
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

# Pretrain models on the blurry data
def pretrain_MNIST_models(
        train_dataset, 
        test_datasets: List[Tuple[str, skorch.dataset.Dataset]], 
        logging_dir: Path, 
        num_epochs:int=PRETRAINING_EPOCHS, 
        dataset_classes=list(range(10)), 
        input_dim=784):
    ''' Train the source models on the 'degraded' training conditions.'''

    # Return paths to models for ease.
    list_of_model_files: list[Path] = []
    list_of_model_histories: list[Path] = []

    for run in trange(NUMBER_RUNS, desc="Degraded Pretraining Runs"):
        logging_dir_run = logging_dir / f"run_{run}"
        
        test_callbacks = get_all_test_callbacks(test_datasets=test_datasets, logging_dir_run=logging_dir_run)
        callbacks = [
            SaveModelInformationCallback(save_dir=str(logging_dir_run)),
            ProgressBar(),
            *test_callbacks,]
            
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

# Load blurry data parameters and further train
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

    # Define callbacks
    test_callbacks = get_all_test_callbacks(test_datasets=test_datasets, logging_dir_run=logging_dir_run)
    callbacks = [
        SaveModelInformationCallback(save_dir=str(logging_dir_run)),
        ProgressBar(),
        *test_callbacks,]
        
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

    # Save history. Both redundant as this is saved through the callback. But am coding quickly at the moment and prefer to have redundancies. 
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

    # Stack into tensors
    X_tensor = torch.stack(X_list) # shape (N, H, W) or (N, C, H, W)
    y_tensor = torch.tensor(y_list, dtype=torch.long)

    # Step 3: Wrap in TensorDataset
    test_dataset = skorch.dataset.Dataset(X_tensor, y_tensor)
    test_dataset.targets = y_tensor
    return test_dataset

if __name__ == "__main__": 

    # Set root directory for logging and ensure they exist
    print(f"Starting experiment run {EXPERIMENT_DIR}!")
    randominit_logging_dir = EXPERIMENT_DIR / 'target_w_random_init'
    os.makedirs(randominit_logging_dir, exist_ok=True)
    pretrained_models_logging_dir = EXPERIMENT_DIR / 'pretraining_on_source'
    os.makedirs(pretrained_models_logging_dir, exist_ok=True)
    noisyinit_logging_dir = EXPERIMENT_DIR / "target_w_source_init"
    os.makedirs(noisyinit_logging_dir, exist_ok=True)

    # Load relevant datasets
    print("Loading data...")
    data_dir = ROOT / "artifacts" / "data"

    # Split into subset
    Original_MNIST_train = torchvision.datasets.MNIST(data_dir, train=True, download=True)
    hard_indices = np.load(os.path.join(data_dir, "hard_indices.npy"))
    complementary_indices = np.load(os.path.join(data_dir, "new_train_indices.npy"))
    MNIST_train = Subset(Original_MNIST_train, complementary_indices)
    MNIST_hard_subset = Subset(Original_MNIST_train, hard_indices)
    assert len(MNIST_train) + len(MNIST_hard_subset) == len(Original_MNIST_train)

    # Transform the source and train datasets
    source_train_dataset = ColorMNIST(MNIST_train, theta=SOURCE_THETA)
    target_train_dataset = ColorMNIST(MNIST_train, theta=TARGET_THETA)

    # Get the input dimension
    x0, y0 = target_train_dataset[0]
    input_dim = x0.numel()

    # Set up the test datasets
    MNIST_test = torchvision.datasets.MNIST(data_dir, train=False, download=True)
    tmp_test_dataset = ColorMNIST(MNIST_test, theta=EVAL_THETA)
    test_dataset = set_up_test_dataset(tmp_test_dataset)

    tmp_hard_test_dataset = ColorMNIST(MNIST_hard_subset, theta=EVAL_THETA)
    hard_test_dataset = set_up_test_dataset(tmp_hard_test_dataset)


    test_datasets=[
        ("MNIST_test", test_dataset),
        ("MNIST_hard", hard_test_dataset)
        ]


    # Sanity check! Save examples of the train, blurry, and test
    # save_dataset_examples(target_train_dataset, source_train_dataset, test_dataset, EXPERIMENT_DIR)

    # Train models! Random init
    if SKIP_BASELINE:
        print("Skipping Baseline!")
    else:
        random_init_model_histories = train_MNIST_models_from_random_init(
            train_dataset=target_train_dataset, 
            logging_dir=randominit_logging_dir,
            test_datasets=test_datasets,
            input_dim=input_dim
            )

        gc.collect()
        print(f"Completed training {NUMBER_RUNS} runs on baseline models starting from random initialisation. Net histories can be found:")
        print(random_init_model_histories)
        
    # Train models! Pretrain
    pretrain_model_histories, pretrain_model_params = pretrain_MNIST_models(
        train_dataset=source_train_dataset, ## Blurry dataset here!
        logging_dir=pretrained_models_logging_dir,
        test_datasets=test_datasets,
        input_dim=input_dim
        )

    gc.collect()
    print(f"Completed pretraining {NUMBER_RUNS} models on source data for {PRETRAINING_EPOCHS} epochs. Net histories can be found:")
    print(pretrain_model_histories)
    
    print("Training on pretrained initialisations")
    # Train models! Pretrain init
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
    
    print(f"Completed training {NUMBER_RUNS} runs on baseline models starting from noisy-pretraining initialisation. Net histories can be found:")
    print(model_history_noisy_init)