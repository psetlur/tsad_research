import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


def setup_logger_and_checkpoint(name, project, monitor):
    logger = WandbLogger(project=project, name=name, log_model=True)
    model_checkpoint = setup_checkpoint(name, monitor)
    return logger, model_checkpoint


def setup_checkpoint(name, monitor):
    os.makedirs(f"checkpoints/{name}/", exist_ok=True)
    return pl.callbacks.ModelCheckpoint(
        monitor=monitor,
        dirpath=f"checkpoints/{name}/",
        filename=f"model",
        save_top_k=1,
        mode="min",
    )


class EarlyStopping(object):
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')

    def reset(self):
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
