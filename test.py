import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from model import SwinFoodClassifier
from dataloader import TestFood101DataModule
from tqdm import tqdm
import pandas as pd


def process_pred_result_to_csv(results):
    prediction = []

    for data in tqdm(results):
        preds = data.cpu().numpy()
        prediction.extend(preds)


    results_df = pd.DataFrame({
        'Category': prediction
    })

    results_df.to_csv('test_predictions.csv', index=True)
    print("Test predictions saved to 'test_predictions.csv'")

def main():
    DATA_DIR = '/home/rvl1421/MIngWei/NTUT_Deep_Learning/taiwanese-food-101/tw_food_101/tw_food_101' # Change if you want data stored elsewhere
    DATA_CSV_FILE = 'tw_food_101_test_list.csv'
    MODEL_NAME = '/home/rvl/Documents/tw_food_101/checkpoints/swin-food101-epoch=19-val_acc=0.91.ckpt'
    IMAGE_SIZE = 224 # Must match the model input size
    BATCH_SIZE = 16   # Adjust based on your GPU memory
    NUM_WORKERS = 8   # Adjust based on your system cores
    ACCELERATOR = "gpu" if torch.cuda.is_available() else "cpu"
    DEVICES = [0] if ACCELERATOR == "gpu" else 1 # Use GPU 0 if available, otherwise CPU
    PRECISION = 16 if ACCELERATOR == "gpu" else 32 # Use mixed precision on GPU for speed/memory savings

    data_module = TestFood101DataModule(
        data_dir=DATA_DIR,
        ds_file=DATA_CSV_FILE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        image_size=IMAGE_SIZE,
    )

    model = SwinFoodClassifier.load_from_checkpoint(MODEL_NAME)
    model.eval()
    model.freeze()

    logger = TensorBoardLogger("output", name="pred_swin_food101")

    trainer = pl.Trainer(
            accelerator=ACCELERATOR,
            devices=DEVICES,
            precision=PRECISION if ACCELERATOR == "gpu" else 32,
            logger=logger,
            log_every_n_steps=0, # Log metrics every 10 training steps
        )
        
    # 運行測試
    test_results = trainer.predict(model, dataloaders=data_module)
    process_pred_result_to_csv(test_results)

if __name__ == '__main__':
    main()