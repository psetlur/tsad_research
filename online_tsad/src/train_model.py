import os
import pytorch_lightning as pl
from lightning.pytorch.loggers import CSVLogger
from utils import setup_checkpoint, setup_logger_and_checkpoint
from pytorch_lightning.callbacks import EarlyStopping
from models.encoder import Encoder


def train_model(args, m_config, train_dataloader, trainval_dataloader, a_config):
    path = "checkpoints/training/"
    model = Encoder(args=args, ts_input_size=m_config.get("ts_input_size"), lr=m_config.get("lr"), a_config=a_config)
    if os.path.exists(path):
        ckpt = os.listdir(path)
        if len(ckpt) > 0:
            return model.load_from_checkpoint(path + ckpt[0])

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
        logger = CSVLogger("logs", name="training", version=args.trail)

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=20
    )
    trainer = pl.Trainer(
        accelerator="auto",
        devices='auto',
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
