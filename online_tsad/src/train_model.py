import os
import pytorch_lightning as pl
from lightning.pytorch.loggers import CSVLogger
from utils import setup_checkpoint, setup_logger_and_checkpoint
from pytorch_lightning.callbacks import EarlyStopping
from models.encoder import Encoder


def train_model(args, m_config, train_dataloader, trainval_dataloader):
    # The directory structure matches what utils.setup_checkpoint creates with name="training"
    checkpoint_dir = "checkpoints/training/"
    # The specific checkpoint file expected based on utils.setup_checkpoint
    expected_checkpoint_file = "model.ckpt"
    resume_checkpoint_path = None

    potential_ckpt_path = os.path.join(checkpoint_dir, expected_checkpoint_file)

    # Check if the specific checkpoint file exists
    if os.path.exists(potential_ckpt_path) and os.path.isfile(potential_ckpt_path):
        resume_checkpoint_path = potential_ckpt_path
        print(f"Found checkpoint: {resume_checkpoint_path}. Attempting to resume training.")
    else:
        print(f"Checkpoint file '{potential_ckpt_path}' not found. Starting training from scratch.")
        # Ensure the directory exists even if starting fresh (setup_checkpoint also does this)
        os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Instantiate the Model ---
    # The Trainer will load the state from the checkpoint into this model instance if ckpt_path is provided
    model = Encoder(args=args, ts_input_size=m_config.get("ts_input_size"), lr=m_config.get("lr")).to(args.device)

    # --- Configure Loggers and Callbacks using utils.py ---
    if args.wandb:
        # setup_logger_and_checkpoint handles both logger and checkpoint callback creation
        logger, ckpt_callback = setup_logger_and_checkpoint(
            name="training",
            project="Auto-TSAD", # Or your specific project name from args if available
            monitor=args.ckpt_monitor,
            # Note: utils.setup_logger_and_checkpoint internally calls utils.setup_checkpoint
            # which sets dirpath="checkpoints/training/" and filename="model"
        )
        print(f"Using Wandb logger and checkpoint callback setup via utils.")
    else:
        # Use utils.setup_checkpoint directly for the checkpoint callback
        ckpt_callback = setup_checkpoint(
            name="training", # This will create checkpoints/training/model.ckpt
            monitor=args.ckpt_monitor,
            # Note: setup_checkpoint sets mode="min" by default, ensure this matches your monitor
        )
        # Use standard CSVLogger if not using wandb
        logger = CSVLogger("logs", name="training", version=args.trail)
        print(f"Using CSV logger and checkpoint callback setup via utils.")

    # Standard EarlyStopping callback
    early_stop_callback = EarlyStopping(
        monitor="val_loss", # Make sure this matches ckpt_callback's monitor if they should align
        mode="min", # Should match ckpt_callback's mode
        patience=100 # Keep original patience or adjust as needed
    )

    # --- Adjust Total Epochs for Continued Training ---
    # Ensure m_config["epochs"] reflects the TOTAL desired epochs (e.g., 800)
    total_epochs = m_config["epochs"]
    print(f"Training will run up to a total of {total_epochs} epochs.")
    if resume_checkpoint_path:
         print(f"Resuming from checkpoint, training will continue until epoch {total_epochs}.")


    # --- Configure the Trainer ---
    trainer = pl.Trainer(
        accelerator='auto',
        devices='auto', # Or specify like [0] if needed
        strategy=args.strategy,
        max_epochs=total_epochs, # Total epochs for the entire training process
        callbacks=[ckpt_callback, early_stop_callback],
        log_every_n_steps=10,
        logger=logger,
        num_sanity_val_steps=0,
        deterministic=True, # Kept from original code
    )

    # --- Start Training (or Resuming) ---
    print(f"Starting trainer.fit()...")
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=trainval_dataloader,
        ckpt_path=resume_checkpoint_path # Pass the path here! If None, starts fresh.
    )

    # --- Load the Best Model After Training ---
    # After fitting (either from scratch or resumed), load the best checkpoint found *during this run*
    # The best_model_path should point to 'checkpoints/training/model.ckpt' if it was the best
    print(f"Training finished.")
    best_model_path_from_callback = ckpt_callback.best_model_path
    print(f"Best model saved path according to callback: {best_model_path_from_callback}")

    if best_model_path_from_callback and os.path.exists(best_model_path_from_callback):
         print(f"Loading best model from: {best_model_path_from_callback}")
         # Use load_from_checkpoint on the class to get a fresh instance with the best weights
         # Need to pass original args and potentially m_config again if needed by load_from_checkpoint
         best_model = Encoder.load_from_checkpoint(
             best_model_path_from_callback,
             # If your Encoder's __init__ or hparams need these, pass them. Otherwise, remove.
             args=args,
             ts_input_size=m_config.get("ts_input_size"),
             lr=m_config.get("lr")
         )
         # Put model on the correct device after loading
         best_model.to(args.device)
         return best_model
    else:
        print(f"Warning: No best model path found or path does not exist: '{best_model_path_from_callback}'. Returning the model state at the end of training.")
        # Fallback: return the model as it is after the last training step
        model.eval() # Set to evaluation mode
        return model


