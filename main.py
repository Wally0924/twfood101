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

    # --- Configuration ---
    DATA_DIR = '/home/rvl1421/MIngWei/NTUT_Deep_Learning/taiwanese-food-101/tw_food_101/tw_food_101' # Change if you want data stored elsewhere
    DATA_CSV_FILE = 'tw_food_101_train.csv'
    MODEL_NAME = 'swin_base_patch4_window7_224.ms_in22k' # Example SwinV2 model from timm pre-trained on ImageNet-22k
    IMAGE_SIZE = 224 # Must match the model input size
    BATCH_SIZE = 16   # Adjust based on your GPU memory
    NUM_WORKERS = 8   # Adjust based on your system cores
    LEARNING_RATE = 3e-4 # Starting point, might need tuning
    MAX_EPOCHS = 80    # Adjust as needed
    ACCELERATOR = "gpu" if torch.cuda.is_available() else "cpu"
    DEVICES = [0] if ACCELERATOR == "gpu" else 1 # Use GPU 0 if available, otherwise CPU
    PRECISION = "16-mixed" if ACCELERATOR == "gpu" else 32 # Use mixed precision on GPU for speed/memory savings
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

    trainer = KFoldTrainer(
        model_class=SwinFoodClassifier,
        data_module=data_module,
        model_kwargs={'model_name': MODEL_NAME, 'num_classes': data_module.num_classes, 'learning_rate': LEARNING_RATE, 'pretrained': True},
        trainer_kwargs={'accelerator': ACCELERATOR, 'devices': DEVICES, 'max_epochs': MAX_EPOCHS, 'precision': PRECISION if ACCELERATOR == "gpu" else 32}
    )

    # --- Start Training ---
    print(f"Starting training with model: {MODEL_NAME} on {ACCELERATOR}")
    results = trainer.run()