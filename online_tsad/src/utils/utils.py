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
        filename=f"model_epoch" + "-{" + monitor + ":.2f}",
        save_top_k=1,
        # save_last=True,
        mode="min",
    )
