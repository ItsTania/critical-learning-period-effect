import os
from datetime import datetime
import pickle
from skorch.callbacks import Checkpoint, EpochScoring, Callback # type: ignore
import torch
import numpy as np


## Save model weights for further investigation
def save_model_trigger(net, factor=5):
    """
    Returns True if the current epoch number is a power of 2
    or divisible by another number (currently 5).
    """
    n = net.history[-1, 'epoch']  + 1 # current epoch, 0-based

    if (n & (n - 1)) == 0:
        return True
    if (n % factor == 0):
        return True
    return False

def get_model_checkpoints(dirname):
    file_format = os.path.join(dirname, 'model_epoch_{last_epoch[epoch]}.pt')
    return Checkpoint(f_params=file_format, monitor=save_model_trigger)

## Log model performance on accuracy at every epoch
valid_acc_epoch_logger = EpochScoring(
    scoring="accuracy",
    lower_is_better=False,
    on_train=False,   # use validation data
    name="valid_acc"
)

train_acc_epoch_logger = EpochScoring(
    scoring="accuracy",
    lower_is_better=False,
    on_train=True,       # compute on training data
    name="train_acc",
)

## Save the model performance at the end
def default_namer(net):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        f"nethistory_"
        f"{timestamp}.pkl"
    )
class SaveHistoryCallback(Callback):
    def __init__(self, namer=default_namer, save_dir='net_histories', prefix='experiment_name'):
        self.save_dir = save_dir
        self.prefix = prefix
        self.namer = namer
        os.makedirs(save_dir, exist_ok=True)

    def on_train_end(self, net, **kwargs):
        # Construct filename based on module and optimizer hyperparameters
        fname = f"{self.prefix}_{self.namer(net)}"
        path = os.path.join(self.save_dir, fname)

        # Save history object
        with open(path, 'wb') as f:
            pickle.dump(net.history, f)
        print(f"Saved history to {path}")

## Save initial model checkpoint
class SaveModelInformationCallback(Callback):
    def __init__(self, save_dir='model_checkpoint'):
        self.save_dir = save_dir

    def on_train_begin(self, net, **kwargs):
        # Initialise logging dir
        logging_dir = os.path.join(self.save_dir, "initial")
        os.makedirs(logging_dir, exist_ok=True)

        # Run validation
        val_ds = getattr(net, 'heldout_test_dataset', None)
        if val_ds is not None:
            val_score = net.score(val_ds, y=val_ds.targets)
            with open(os.path.join(logging_dir,'initial_score.txt'), 'a') as f:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write("Initial Evaluation:\n")
                f.write(f"Validation Score: {val_score}\n")
                f.write(f"Start Time: {str(timestamp)}\n")
                f.write("="*30 + "\n")

        # Save parameters using skorch
        net.save_params(
            f_params=os.path.join(logging_dir,'initial_model.pkl'), 
            )
        print(f"Saved history to {logging_dir}")
    
    def on_train_end(self, net, **kwargs):
        # Initialise logging dir
        logging_dir = os.path.join(self.save_dir, "final")
        os.makedirs(logging_dir, exist_ok=True)

        # Save parameters using skorch
        net.save_params(
            f_params=os.path.join(logging_dir,'final_model.pkl'), 
            f_optimizer=os.path.join(logging_dir,'final_opt.pkl'), 
            f_history=os.path.join(logging_dir,'training_history.json')
            )
        
        # Double safe
        with open(os.path.join(self.save_dir,'training_history_pickled.pkl'), 'wb') as f:
            pickle.dump(net.history, f)
            
        print(f"Saved history to {logging_dir}")


class SkorchTestPerformanceLogger(Callback):
    """A callback to log test performance metrics and save logits & model periodically."""

    def __init__(self,
                 test_loader,
                 experiment_dir,
                 metric_log_name="test",
                 linear_every_n_batches=None,
                 log_base_scheduler=None,
                 model_dir="model_checkpoints",
                 logits_dir="logs"):
        self.test_loader = test_loader
        self.metric_log_name = metric_log_name

        # Logging schedule
        self.linear_every_n_batches = linear_every_n_batches
        self.log_base_scheduler = log_base_scheduler
        self.next_log_batch = 1

        if self.linear_every_n_batches is None and self.log_base_scheduler is None:
            self.linear_every_n_batches = 1  # default: log every batch

        # Directories
        self.model_dir = os.path.join(experiment_dir, model_dir)
        self.logits_dir = os.path.join(experiment_dir, logits_dir, f"{metric_log_name}_logits")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.logits_dir, exist_ok=True)

    def should_run_logging(self, batch_idx):
        """Scheduling Logic: Return True if we should log at this batch index."""
        if self.linear_every_n_batches is not None:
            if batch_idx % self.linear_every_n_batches == 0:
                return True

        if self.log_base_scheduler is not None and batch_idx >= self.next_log_batch:
            self.next_log_batch *= self.log_base_scheduler
            return True

        return False

    def on_train_begin(self, net, X, y):
        print(f"[{self.metric_log_name}] Logging to: {self.logits_dir}")
        print("Running initial test at batch 0...")
        self._run_test_and_log(net, batch_idx=0)

    def on_batch_end(self, net, X, y, training, **kwargs):
        if not training:
            return  # skorch triggers on_batch_end at the end of training and validation. We don't want to double up. 
        batch_idx = net.history[-1]['batches'][-1]['batch']  # skorch records current batch
        if self.should_run_logging(batch_idx):
            self._run_test_and_log(net, batch_idx)

    def _run_test_and_log(self, net, batch_idx):
        net.module_.eval()
        all_logits, all_labels = [], []

        with torch.no_grad():
            for xb, yb in self.test_loader:
                xb, yb = xb.to(net.device), yb.to(net.device)
                logits = net.module_(xb)
                all_logits.append(logits.cpu())
                all_labels.append(yb.cpu())

        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)
        preds = torch.argmax(logits, dim=1)

        acc = (preds == labels).float().mean().item()

        # Log to history
        net.history.record(f'{self.metric_log_name}_acc', acc)

        # Save logits and model checkpoint
        logits_file = os.path.join(self.logits_dir, f"batch_{batch_idx}.pt")
        torch.save({"logits": logits, "labels": labels}, logits_file)

        model_file = os.path.join(self.model_dir, f"model_batch_{batch_idx}.pt")
        net.save_params(f_params=model_file)