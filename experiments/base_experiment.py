import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
import torch
from torch.utils.data import Subset
import pandas as pd
import numpy as np
import gc
import json

import skorch
from skorch import NeuralNetClassifier

class BaseExperiment(ABC):
    """
    General abstract class that defines an MNIST variant experiment:
    baseline training → pretraining → fine-tuning.
    """

    def __init__(self, experiment_dir: Path, num_runs=1, device=None, data_dir=Path("artifacts/data")):
        self.experiment_dir = experiment_dir
        self.num_runs = num_runs
        self.device = device or self._get_device()

        self.data_dir = data_dir
        self._prepare_log_dirs()

        # These must be filled by subclasses
        self.target_dataset = None
        self.source_dataset = None
        self.test_datasets = None
        self.input_dim = None

    @abstractmethod
    def prepare_datasets(self):
        """Prepare self.train_dataset, self.source_dataset, self.test_datasets, self.input_dim."""
        pass

    @abstractmethod
    def build_model(self, finetuning:bool, logging_dir:Path) -> NeuralNetClassifier:
        """Return a configured skorch NeuralNetClassifier."""
        pass

    def _save_config(self, **kwargs):
        """Save experiment configuration as a JSON file for reproducibility."""
        cfg = kwargs.copy()
        cfg_serializable = {k: str(v) for k, v in cfg.items()}
        config_path = self.experiment_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(cfg_serializable, f, indent=4)

    def _get_device(self):
        if torch.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def _prepare_log_dirs(self):
        self.randominit_dir = self.experiment_dir / "target_w_random_init"
        self.pretrain_dir = self.experiment_dir / "pretraining_on_source"
        self.finetune_dir = self.experiment_dir / "target_w_source_init"
        for d in [self.randominit_dir, self.pretrain_dir, self.finetune_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _save_history(self, net, out_path: Path, run=None):
        df = pd.DataFrame(net.history)
        if run is not None:
            df["run"] = run
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path)

    def run_baseline(self, epochs: int):
        histories = []
        for run in range(self.num_runs):
            # Set the logging_dir for the run
            logging_dir_run = self.randominit_dir / f"run_{run}"

            # Set and fit the model
            net = self.build_model(finetuning=False, logging_dir=logging_dir_run)
            net.fit(self.target_dataset, y=None, epochs=epochs)

            # Save
            hist = logging_dir_run / "net_history.csv"
            self._save_history(net, hist, run)
            histories.append(hist)
        gc.collect()
        return histories

    def run_pretraining(self, epochs: int):
        hist_paths, ckpt_paths = [], []
        for run in range(self.num_runs):
            # Set the logging_dir for the run
            logging_dir_run = self.pretrain_dir / f"run_{run}"

            # Define the model
            net = self.build_model(finetuning=False, logging_dir=logging_dir_run)

            # Fit the model
            net.fit(self.source_dataset, y=None, epochs=epochs)

            # Save
            hist = logging_dir_run / "net_history.csv"
            self._save_history(net, hist, run)
            ckpt = logging_dir_run / "pretrained.pt"
            torch.save(net.module_.state_dict(), ckpt)
            hist_paths.append(hist)
            ckpt_paths.append(ckpt)
        gc.collect()
        return hist_paths, ckpt_paths

    def run_finetuning(self, pretrained_ckpts: List[Path], epochs: int):
        finetune_histories = []
        for run, ckpt_path in enumerate(pretrained_ckpts):
            # Set the logging_dir for the run
            logging_dir_run = self.finetune_dir / f"run_{run}"

            # Initialise model
            net = self.build_model(finetuning=True, logging_dir=logging_dir_run)
            net.initialize()
            state = torch.load(str(ckpt_path), map_location=net.device)
            net.module_.load_state_dict(state)
            if not net.warm_start:
                print("Warm start must be True to further fine-tune and not reinitialise the model")
                print("Setting warm start to True")
                net.warm_start = True

            # Fit model
            net.fit(self.target_dataset, y=None, epochs=epochs)

            # Save
            hist = logging_dir_run / "net_history.csv"
            self._save_history(net, hist, run)
            finetune_histories.append(hist)
        gc.collect()
        return finetune_histories

    def run_full(self, 
                 target_epochs:int, 
                 source_epochs:int, 
                 skip_baseline=False):
        print(f"Starting experiment in {self.experiment_dir}")
        self.prepare_datasets()
        if not skip_baseline:
            print("Baseline training...")
            base_hist = self.run_baseline(target_epochs)
            print(base_hist)
        print("Pretraining...")
        pre_hist, ckpts = self.run_pretraining(source_epochs)
        print(pre_hist)
        print("Fine-tuning...")
        ft_hist = self.run_finetuning(ckpts, target_epochs)
        print(ft_hist)

    def construct_hard_subset(self, original_dataset):
        """ Splits the original Train MNIST split into (new) train MNISR, hard MNIST"""

        hard_indices_fp = os.path.join(self.data_dir, "hard_indices.npy")
        new_train_indices_fp = os.path.join(self.data_dir, "new_train_indices.npy")
        
        if not os.path.exists(hard_indices_fp) or not os.path.exists(new_train_indices_fp):
            raise FileNotFoundError(
                f"Hard split index files not found in {self.data_dir}. "
                "Expected 'hard_indices.npy' and 'new_train_indices.npy'."
            )
        
        hard_indices = np.load(hard_indices_fp)
        new_train_indices = np.load(new_train_indices_fp)

        # Build subsets
        new_train_subset = Subset(original_dataset, new_train_indices)
        hard_subset = Subset(original_dataset, hard_indices)

        # Consistency check
        total_len = len(new_train_subset) + len(hard_subset)
        assert total_len == len(original_dataset), (
            f"Split mismatch: {len(new_train_subset)} + {len(hard_subset)} != {len(original_dataset)}"
        )

        return new_train_subset, hard_subset

def set_up_test_dataset(dataset, device=None) -> skorch.dataset.Dataset:
    X_list, y_list = [], []
    for i in range(len(dataset)):
        x, y = dataset[i]
        X_list.append(x)
        y_list.append(y)

    # Stack into tensors
    X_tensor = torch.stack(X_list) # shape (N, H, W) or (N, C, H, W)
    y_tensor = torch.tensor(y_list, dtype=torch.long)

    if device is not None:
        X_tensor = X_tensor.to(device)
        y_tensor = y_tensor.to(device)

    # Step 3: Wrap in TensorDataset
    test_dataset = skorch.dataset.Dataset(X_tensor, y_tensor)
    test_dataset.targets = y_tensor
    return test_dataset

def combine_experiment_histories(
    histories: List[Path],
    save_dir,
    save_name: str = "combined_results.csv",
    read_config: bool=False
    ) -> pd.DataFrame:

    all_histories = []
    for hist_path in histories:
        try:
            history_df = pd.read_csv(hist_path)
        except Exception as e:
            print(f"Failed to read {hist_path}: {e}")
            continue
        # Add initial Scores
        df = add_initial_scores_from_logits(history_df, hist_path.parent)

        # Add identifiers
        df["run_name"] = hist_path.parent.name
        df["initialisation"] = hist_path.parent.parent.name
        df["experiment_group_name"] = hist_path.parent.parent.parent.name

        if read_config:
            config_path = hist_path.parent.parent / "config.json"
            if config_path.exists():
                try:
                    with open(config_path, "r") as f:
                        config = json.load(f)
                    # Add all config keys to df with "identifier__" prefix
                    for k, v in config.items():
                        # Try to convert numeric strings to float if possible
                        try:
                            v_num = float(v)
                            df[f"identifier__{k}"] = v_num
                        except:
                            df[f"identifier__{k}"] = v
                except Exception as e:
                    print(f"Failed to read config {config_path}: {e}")
            else:
                print(f"Config file not found at {config_path}")

        all_histories.append(df)

    if all_histories:
        combined_df = pd.concat(all_histories, ignore_index=True)
        if save_dir is None:
            print("Not saving...")
        else:
            combined_df.to_csv(save_dir / save_name, index=False)
    else:
        combined_df = pd.DataFrame()
        print("No histories to combine.")
    return combined_df

def add_initial_scores_from_logits(history_df: pd.DataFrame, run_dir: Path, score_subdir: str = "logits") -> pd.DataFrame:
    """
    For a given run directory, read initial_score.txt from all subfolders of `logits` and add to history_df.

    Args:
        history_df: pd.DataFrame for a single run (history CSV)
        run_dir: Path to the run directory (e.g., run_0)
        score_subdir: Name of the folder containing subfolders with initial_score.txt

    Returns:
        pd.DataFrame: history_df with added columns like initial_<score_name>_acc and initial_<score_name>_loss
    """
    logits_dir = run_dir / score_subdir
    if not logits_dir.exists():
        print(f"No logits directory found at {logits_dir}")
        return history_df

    for subfolder in logits_dir.iterdir():
        if not subfolder.is_dir():
            continue
        score_file = subfolder / "initial_score.txt"
        score_name = subfolder.name
        acc_col = f"initial_{score_name}_acc"
        loss_col = f"initial_{score_name}_loss"
        acc, loss = None, None

        if score_file.exists():
            try:
                text = score_file.read_text()
                # Extract Accuracy
                acc_match = re.search(r"Accuracy:\s*([0-9.]+)", text)
                if acc_match:
                    acc = float(acc_match.group(1))
                # Extract Loss
                loss_match = re.search(r"Loss:\s*([0-9.]+)", text)
                if loss_match:
                    loss = float(loss_match.group(1))
            except Exception as e:
                print(f"Failed to read {score_file}: {e}")

        history_df[acc_col] = acc
        history_df[loss_col] = loss

    return history_df

def find_all_histories(experiment_dir: str, history_filename: str = "net_history.csv") -> List[Path]:
    """
    Recursively find all history CSV files in an experiment directory.

    Args:
        experiment_dir (str): Path to the root experiment directory.
        history_filename (str): Filename to look for (default: 'net_history.csv').

    Returns:
        List[Path]: List of Paths to history CSV files.
    """
    exp_dir = Path(experiment_dir)
    if not exp_dir.exists():
        raise ValueError(f"Experiment directory {experiment_dir} does not exist.")

    history_files = list(exp_dir.rglob(history_filename))
    if not history_files:
        print(f"No history files named '{history_filename}' found in {experiment_dir}")
    return history_files