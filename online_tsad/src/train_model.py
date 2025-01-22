import os
import numpy as np

import pytorch_lightning as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch import Trainer

from utils import setup_checkpoint, setup_logger_and_checkpoint
from models import load_model
from pytorch_lightning.callbacks import EarlyStopping


def train_model(args, m_config, train_dataloader, trainval_dataloader, a_config):
    path = "checkpoints/training"
    if os.path.exists(path):
        for l in os.listdir(path):
            os.remove(os.path.join(path, l))

    if args.wandb:
        logger, ckpt_callback = setup_logger_and_checkpoint(
            name="training",
            project="Auto-TSAD",
            monitor=args.ckpt_monitor,
        )
    else:
        ckpt_callback = setup_checkpoint(
            name="training",
            monitor=args.ckpt_monitor,
        )
        logger = CSVLogger("logs", name="training", version='fixed_grid')

    model = load_model(m_config, a_config)
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=10
    )
    trainer = pl.Trainer(
        strategy=args.strategy,
        max_epochs=m_config["epochs"],
        callbacks=[ckpt_callback, early_stop_callback],
        log_every_n_steps=10,
        logger=logger,
        num_sanity_val_steps=0,
        deterministic=True,
    )
    trainer.fit(model, train_dataloader, trainval_dataloader)
    model = model.load_from_checkpoint(ckpt_callback.best_model_path)
    return model
