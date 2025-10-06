import os
from datetime import datetime
import pickle
import skorch
from skorch.callbacks import Checkpoint, EpochScoring, Callback 
import torch
from pathlib import Path
from typing import List, Tuple
from torch.utils.data import DataLoader

## Save model weights for further investigation
def save_model_trigger(net):
    """
    Returns True if the current epoch number is a power of 2
    or divisible by another number (currently 5).
    """
    factor=5
    n = net.history[-1, 'epoch']  + 1 # current epoch, 0-based

    if (n & (n - 1)) == 0:
        return True
    if (n % factor == 0):
        return True
    return False

def get_model_checkpoints(dirname):
    file_format = os.path.join(dirname, 'model_epoch_{last_epoch[epoch]}.pt')
    return Checkpoint(dirname=str(dirname), f_params=file_format, monitor=save_model_trigger)

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
    ''' Save the initial and final model and its full history'''
    def __init__(self, save_dir='model_checkpoint'):
        self.save_dir = save_dir

    def on_train_begin(self, net, **kwargs):
        # Initialise logging dir
        logging_dir = os.path.join(self.save_dir, "initial")
        os.makedirs(logging_dir, exist_ok=True)

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
                 logits_dir="logits",
                 num_classes=10):
        
        # Run once after initialisation
        self.has_run = False

        # Metric
        self.test_loader = test_loader
        self.metric_log_name = metric_log_name
        self.N = len(self.test_loader.dataset)
        self.num_classes = num_classes

        # Logging schedule
        self.checkpoint_fn = checkpoint_fn if checkpoint_fn is not None else (lambda net: True)

        # Directories
        self.logits_dir = os.path.join(experiment_dir, logits_dir, f"{metric_log_name}_logits")
        os.makedirs(self.logits_dir, exist_ok=True)

    def on_train_begin(self, net, X, y):
        #print(f"[{self.metric_log_name}] Logging to: {self.logits_dir}")
        if self.has_run:
            return
        
        self.has_run = True
        acc, loss = self._run_test_and_log(net, epoch=0, save_to_disk=True) #skorch logs the first epoch as 1
        
        # Log to text file
        with open(os.path.join(self.logits_dir,'initial_score.txt'), 'a') as f:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write("Initial Evaluation:\n")
                f.write(f"Accuracy: {acc}\n")
                f.write(f"Loss: {loss}\n")
                f.write(f"Start Time: {str(timestamp)}\n")
                f.write("="*30 + "\n")

    def on_epoch_end(self, net, **kwargs): 
        epoch = net.history[-1, 'epoch']
        acc, loss = self._run_test_and_log(net, epoch=epoch, save_to_disk=self.checkpoint_fn(epoch))

        net.history.record(f"{self.metric_log_name}_acc", acc)
        net.history.record(f"{self.metric_log_name}_loss", loss)

    def _run_test_and_log(self, net, epoch, save_to_disk):
        net.module_.eval()
        logits_all = torch.empty(self.N, self.num_classes)
        labels_all = torch.empty(self.N, dtype=torch.long)

        start = 0
        with torch.no_grad():
            for xb, yb in self.test_loader:
                b = xb.size(0)

                # Forward pass
                xb, yb = xb.to(net.device), yb.to(net.device)
                logits = net.module_(xb)

                # Save tensors
                logits_all[start:start+b] = logits.cpu()
                labels_all[start:start+b] = yb.cpu()
                start += b

        if save_to_disk:
            logits_file = os.path.join(self.logits_dir, f"epoch_{epoch}.pt")
            torch.save(logits_all, logits_file)
        
        y_pred = logits_all.argmax(dim=1)
        acc = (y_pred == labels_all).float().mean().item()
        loss = net.criterion_(logits_all, labels_all).item()
        return acc, loss

def checkpoint_fn(epoch):
    return (epoch % 10 == 0) or ((epoch & (epoch - 1)) == 0)

class ModelCheckpointLogger(Callback):
    """Callback to save model checkpoints with flexible pathing."""

    def __init__(self, experiment_dir, model_dir="model_checkpoints", checkpoint_fn=None):
        self.model_dir = os.path.join(experiment_dir, model_dir)
        os.makedirs(self.model_dir, exist_ok=True)
        self.checkpoint_fn = checkpoint_fn if checkpoint_fn is not None else lambda net: True

    def on_epoch_end(self, net, **kwargs):
        epoch = net.history[-1, 'epoch']
        if self.checkpoint_fn(epoch):
            model_file = os.path.join(self.model_dir, f"model_epoch_{epoch}.pt")
            net.save_params(f_params=model_file)
            #print(f"[Checkpoint] Saved model at epoch {epoch} -> {model_file}")

def get_all_test_callbacks(
        test_datasets: List[Tuple[str, skorch.dataset.Dataset]],
        logging_dir_run: Path, 
        checkpoint_fn=checkpoint_fn,
        batch_size: int = 128,
        num_workers: int = 4
) -> List[Callback]:
    '''Convienence function to get all the relevant callbacks to log logits throughout training'''
    callbacks=[]
    callbacks.append(ModelCheckpointLogger(experiment_dir=logging_dir_run, checkpoint_fn=checkpoint_fn))
    for name, dataset in test_datasets:
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        callbacks.append(
            SkorchTestPerformanceLogger(
                test_loader=test_loader,
                experiment_dir=logging_dir_run,
                metric_log_name=name
            )
        )

    return callbacks