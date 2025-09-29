import os
import sys
import gc
from pathlib import Path

import torch
from tqdm import trange # type: ignore
import pandas as pd # type: ignore

from skorch import NeuralNetClassifier # type: ignore
from skorch.helper import predefined_split  # type: ignore
from skorch.callbacks import ProgressBar # type: ignore

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.models.achille import Achille_MNIST_FC, Achille_MNIST_FC_No_BatchNorm, get_activation # noqa: E402
from utils.callbacks import SaveModelInformationCallback, valid_acc_epoch_logger, get_model_checkpoints  # noqa: E402
from utils.data import MNIST_dataset, achille_preprocess, achille_transform_train, achille_blurry_transform_train, save_dataset_examples  # noqa: E402

# Experiment params - general
NUMBER_RUNS = 1 #5
EXPERIMENT_DIR = ROOT / Path("artifacts/experiment_results/example")
SKIP_BASELINE=True

if torch.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'


# Experiment params - taken from Achilles
MODEL=Achille_MNIST_FC_No_BatchNorm
ACTIVATION=get_activation('relu')
PRETRAINING_EPOCHS= 1#480 # In the paper they test [40 * x for x in range(12)]
CLEAN_EPOCHS=1 #180
LEARNING_RATE=0.005
BATCH=512#128
OPTIMIZER=torch.optim.Adam
CRITERION=torch.nn.CrossEntropyLoss



# Train baseline models
def train_MNIST_models_from_random_init(train_dataset, test_dataset, logging_dir: Path, num_epochs:int=CLEAN_EPOCHS, dataset_classes=list(range(10))):
    list_of_model_histories: list[Path] = []
    for run in trange(NUMBER_RUNS, desc="Random Init Runs"):
        logging_dir_run = logging_dir / f"run_{run}"
        net = NeuralNetClassifier(
            module=MODEL,
            lr=LEARNING_RATE,
            optimizer=OPTIMIZER,
            criterion=CRITERION,
            device=DEVICE,
            callbacks=[
                valid_acc_epoch_logger,
                SaveModelInformationCallback(save_dir=str(logging_dir_run)), 
                get_model_checkpoints(str(logging_dir_run/ "checkpoints")),
                ProgressBar()],
            train_split=predefined_split(test_dataset),
            classes=dataset_classes,
            module__activation=ACTIVATION,
            iterator_train__num_workers=4,
            iterator_valid__num_workers=4,
            )
        net.heldout_test_dataset=test_dataset
        net.fit(train_dataset, y=None, epochs=num_epochs)

        # Save history. Both redundant as this is saved through the callback. But am coding quickly at the moment and prefer to have redundancies. 
        df = pd.DataFrame(net.history)
        df['run'] = run
        df.to_csv(str(logging_dir_run / "net_history.csv"))
        list_of_model_histories.append(logging_dir_run / "net_history.csv")

    return list_of_model_histories

# Pretrain models on the blurry data
def pretrain_MNIST_models(train_dataset, test_dataset, logging_dir: Path, num_epochs:int=PRETRAINING_EPOCHS, dataset_classes=list(range(10))):
    list_of_model_files: list[Path] = []
    list_of_model_histories: list[Path] = []

    for run in trange(NUMBER_RUNS, desc="Degraded Pretraining Runs"):
        logging_dir_run = logging_dir / f"run_{run}"
        net = NeuralNetClassifier(
            module=MODEL,
            lr=LEARNING_RATE,
            optimizer=OPTIMIZER,
            criterion=CRITERION,
            device=DEVICE,
            callbacks=[
                valid_acc_epoch_logger,
                SaveModelInformationCallback(save_dir=str(logging_dir_run)), 
                get_model_checkpoints(str(logging_dir_run/ "checkpoints")),
                ProgressBar()],
            train_split=predefined_split(test_dataset),
            classes=dataset_classes,
            module__activation=ACTIVATION,
            iterator_train__num_workers=4,
            iterator_valid__num_workers=4,
            )
        net.heldout_test_dataset=test_dataset
        net.fit(train_dataset, y=None, epochs=num_epochs)

        # Save history.
        df = pd.DataFrame(net.history)
        df['run'] = run
        df.to_csv(logging_dir_run / "net_history.csv")
        list_of_model_histories.append(logging_dir_run / "net_history.csv")
        
        # Save model
        model_path = logging_dir_run / 'pretrained_model_weights.pt'
        print(f"Saving model parameters to {str(model_path)}") # Both redundant as this is saved through the callback. But am coding quickly at the moment and prefer to have redundancies. 
        torch.save(net.module_.state_dict(), str(model_path))
        list_of_model_files.append(model_path)

    return list_of_model_histories, list_of_model_files

# Load blurry data parameters and further train
def train_MNIST_model_from_pretrained_init(run, pretrained_weights_fp: Path, train_dataset, test_dataset, logging_dir: Path, num_epochs:int=CLEAN_EPOCHS, dataset_classes=list(range(10))):
    logging_dir_run = logging_dir / f"run_{run}"
    net = NeuralNetClassifier(
        module=MODEL,
        lr=LEARNING_RATE,
        optimizer=OPTIMIZER,
        criterion=CRITERION,
        device=DEVICE,
        warm_start=True, 
        callbacks=[
            valid_acc_epoch_logger,
            SaveModelInformationCallback(save_dir=str(logging_dir_run)), 
            get_model_checkpoints(str(logging_dir_run/ "checkpoints")),
            ProgressBar()],
        train_split=predefined_split(test_dataset),
        classes=dataset_classes,
        module__activation=ACTIVATION,
        iterator_train__num_workers=4,
        iterator_valid__num_workers=4,
        )
    net.heldout_test_dataset=test_dataset
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

if __name__ == "__main__": 

    # Set root directory for logging and ensure they exist
    print(f"Starting experiment run {EXPERIMENT_DIR}!")
    randominit_logging_dir = EXPERIMENT_DIR / 'random_init'
    os.makedirs(randominit_logging_dir, exist_ok=True)
    pretrained_models_logging_dir = EXPERIMENT_DIR / 'pretraining_on_blur'
    os.makedirs(pretrained_models_logging_dir, exist_ok=True)
    noisyinit_logging_dir = EXPERIMENT_DIR / "blur_pretrained_init"
    os.makedirs(noisyinit_logging_dir, exist_ok=True)

    # Load relevant datasets
    print("Loading data...")
    data_dir = ROOT / "artifacts" / "data"
    train_dataset = MNIST_dataset(is_train=True, transforms=achille_transform_train, data_dir=data_dir)
    blurry_train_dataset = MNIST_dataset(is_train=True, transforms=achille_blurry_transform_train, data_dir=data_dir)
    test_dataset = MNIST_dataset(is_train=False, transforms=achille_preprocess, data_dir=data_dir)

    # Sanity check! Save examples of the train, blurry, and test
    save_dataset_examples(train_dataset, blurry_train_dataset, test_dataset, EXPERIMENT_DIR)

    # Train models! Random init
    if SKIP_BASELINE:
        print("Skipping Baseline!")
    else:
        random_init_model_histories = train_MNIST_models_from_random_init(
            train_dataset=train_dataset, 
            test_dataset=test_dataset, 
            logging_dir=randominit_logging_dir
            )

        gc.collect()
        print(f"Completed training {NUMBER_RUNS} runs on baseline models starting from random initialisation. Net histories can be found:")
        print(random_init_model_histories)

    # Train models! Pretrain
    pretrain_model_histories, pretrain_model_params = pretrain_MNIST_models(
        train_dataset=blurry_train_dataset, ## Blurry dataset here!
        test_dataset=test_dataset, 
        logging_dir=pretrained_models_logging_dir
        )
    
    gc.collect()
    print(f"Completed pretraining {NUMBER_RUNS} models on noisy data for {PRETRAINING_EPOCHS} epochs. Net histories can be found:")
    print(pretrain_model_histories)

    # Train models! Pretrain init
    model_history_noisy_init = []
    for run, model_params in enumerate(pretrain_model_params):
        net_history = train_MNIST_model_from_pretrained_init(
            run=run,
            train_dataset=train_dataset, 
            test_dataset=test_dataset,
            pretrained_weights_fp=model_params, 
            logging_dir=noisyinit_logging_dir
            )
        model_history_noisy_init.append(net_history)
    
    print(f"Completed training {NUMBER_RUNS} runs on baseline models starting from noisy-pretraining initialisation. Net histories can be found:")
    print(random_init_model_histories)