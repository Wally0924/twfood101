import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.model_selection import KFold

from dataloader import KFoldFood101DataModule, TestFood101DataModule
from model import SwinFoodClassifier
from trainer import KFoldTrainer


if __name__ == '__main__':
    pl.seed_everything(42) # for reproducibility
    torch.set_float32_matmul_precision('high')

    # --- Configuration ---
    DATA_DIR = '/home/rvl1421/MIngWei/NTUT_Deep_Learning/taiwanese-food-101/tw_food_101/tw_food_101' # Change if you want data stored elsewhere
    DATA_CSV_FILE = 'tw_food_101_train.csv'
    MODEL_NAME = 'swin_base_patch4_window7_224.ms_in22k' # Example SwinV2 model from timm pre-trained on ImageNet-22k
    IMAGE_SIZE = 224 # Must match the model input size
    BATCH_SIZE = 32   # Adjust based on your GPU memory
    NUM_WORKERS = 6   # Adjust based on your system cores
    LEARNING_RATE = 3e-4 # Starting point, might need tuning
    MAX_EPOCHS = 80    # Adjust as needed
    ACCELERATOR = "gpu" if torch.cuda.is_available() else "cpu"
    DEVICES = [0] if ACCELERATOR == "gpu" else 1 # Use GPU 0 if available, otherwise CPU
    PRECISION = 16 if ACCELERATOR == "gpu" else 32 # Use mixed precision on GPU for speed/memory savings
    FOLDS = 3

    # --- Initialization ---
    data_module = KFoldFood101DataModule(
        data_dir=DATA_DIR,
        ds_file=DATA_CSV_FILE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        image_size=IMAGE_SIZE,
        num_folds=FOLDS,
    )

    # model = SwinFoodClassifier(
    #     model_name=MODEL_NAME,
    #     num_classes=data_module.num_classes,
    #     learning_rate=LEARNING_RATE,
    #     pretrained=True
    # )

    # --- Callbacks ---
    # Log to TensorBoard
    logger = TensorBoardLogger("tb_logs", name="swin_food101")

    # Save the best model based on validation accuracy
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',       # Metric to monitor
        dirpath='checkpoints/',  # Directory to save checkpoints
        filename='swin-food101-{epoch:02d}-{val_acc:.2f}',
        save_top_k=1,            # Save only the best model
        mode='max',              # Maximize validation accuracy
        save_last=True           # Optionally save the last epoch's checkpoint
    )

    # Stop training early if validation loss doesn't improve
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=5, # Number of epochs with no improvement after which training will be stopped.
        verbose=True,
        mode='min'
    )

    # --- Trainer ---
    # trainer = pl.Trainer(
    #     accelerator=ACCELERATOR,
    #     devices=DEVICES,
    #     max_epochs=MAX_EPOCHS,
    #     precision=PRECISION if ACCELERATOR == "gpu" else 32,
    #     logger=logger,
    #     callbacks=[checkpoint_callback, early_stopping_callback],
    #     log_every_n_steps=0, # Log metrics every 10 training steps
    #     # accumulate_grad_batches=2, # Optional: Simulate larger batch size if memory constrained
    #     # gradient_clip_val=0.5,     # Optional: Gradient clipping
    # )

    trainer = KFoldTrainer(
        model_class=SwinFoodClassifier,
        data_module=data_module,
        model_kwargs={'model_name': MODEL_NAME, 'num_classes': data_module.num_classes, 'learning_rate': LEARNING_RATE, 'pretrained': True},
        # trainer_kwargs={'accelerator': ACCELERATOR, 'devices': DEVICES, 'max_epochs': MAX_EPOCHS, 'precision': PRECISION if ACCELERATOR == "gpu" else 32, 'logger': logger, 'callbacks': [checkpoint_callback, early_stopping_callback], 'log_every_n_steps': 0}
        trainer_kwargs={'accelerator': ACCELERATOR, 'devices': DEVICES, 'max_epochs': MAX_EPOCHS, 'precision': PRECISION if ACCELERATOR == "gpu" else 32}
    )   

    # --- Start Training ---
    print(f"Starting training with model: {MODEL_NAME} on {ACCELERATOR}")
    results = trainer.run()

    # --- Optional: Test the best model ---
    # print("\nStarting testing using the best model checkpoint...")
    # # trainer.test(datamodule=data_module, ckpt_path='best') # 'best' loads the best checkpoint automatically
    # # Or load a specific checkpoint:
    # best_model_path = checkpoint_callback.best_model_path
    # if best_model_path:
    #      print(f"Loading best model from: {best_model_path}")
    #      trainer.test(model=model, datamodule=data_module, ckpt_path=best_model_path)
    # else:
    #     print("No best model checkpoint found, testing with last model state.")
    #     trainer.test(model=model, datamodule=data_module)

    # print("\nTraining and testing finished.")
    # print(f"Best model checkpoint saved at: {checkpoint_callback.best_model_path}")
    # print(f"TensorBoard logs saved in: {logger.log_dir}")