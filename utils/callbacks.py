import os
from datetime import datetime
import pickle
from skorch.callbacks import Checkpoint, EpochScoring, Callback # type: ignore


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