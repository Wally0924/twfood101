import pytorch_lightning as pl
import warnings
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch

class KFoldTrainer:
    """演示如何使用KFoldDataModule進行K-fold交叉驗證的示例類"""
    
    def __init__(
        self,
        model_class,
        data_module,
        model_kwargs: dict = {},
        trainer_kwargs: dict = {}
    ):
        self.model_class = model_class
        self.data_module = data_module
        self.model_kwargs = model_kwargs
        self.trainer_kwargs = trainer_kwargs
        self.results = []
    
    def run(self) -> list[dict]:
        """執行K-fold交叉驗證"""
        warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
        warnings.filterwarnings("ignore", category=UserWarning, message="Attempting to use hipBLASLt on an unsupported architecture!")
        torch.set_float32_matmul_precision('high')
        # for fold in range(self.data_module.num_folds):
        #     print(f"Training fold {fold+1}/{self.data_module.num_folds}")
            
        #     # 設置當前折疊
        #     self.data_module.set_fold(fold)
            
        #     # 初始化模型
        #     model = self.model_class(**self.model_kwargs)
            
        #     # 初始化訓練器
        #     trainer = pl.Trainer(**self.trainer_kwargs, 'logger': logger, 'callbacks': [checkpoint_callback, early_stopping_callback], 'log_every_n_steps': 0)
            
        #     # 訓練模型
        #     trainer.fit(model, datamodule=self.data_module)
            
        #     # 驗證模型
        #     val_results = trainer.validate(model, datamodule=self.data_module)
            
        #     # 保存結果
        #     self.results.append({
        #         'fold': fold,
        #         'val_results': val_results
        #     })
        
        # return self.results
        for fold in range(self.data_module.num_folds):
            print(f"Training fold {fold+1}/{self.data_module.num_folds}")
            self.data_module.set_fold(fold)

            model = self.model_class(**self.model_kwargs)

            # ✅ 獨立的 logger 路徑
            logger = TensorBoardLogger("tb_logs", name="swin_food101")

            # ✅ 獨立的 EarlyStopping (重設狀態)
            early_stopping_callback = EarlyStopping(
                monitor='val_loss',
                patience=3, # Number of epochs with no improvement after which training will be stopped.
                verbose=True,
                mode='min'
            )

            checkpoint_callback = ModelCheckpoint(
                monitor='val_acc',       # Metric to monitor
                dirpath='checkpoints/',  # Directory to save checkpoints
                filename='swin-food101-{epoch:02d}-{val_acc:.2f}',
                save_top_k=1,            # Save only the best model
                mode='max',              # Maximize validation accuracy
                save_last=True           # Optionally save the last epoch's checkpoint
            )

            trainer_kwargs = {
                **self.trainer_kwargs,
                'logger': logger,
                'callbacks': [checkpoint_callback, early_stopping_callback],
                'log_every_n_steps': 0
            }

            trainer = pl.Trainer(**trainer_kwargs)
            trainer.fit(model, datamodule=self.data_module)

            val_results = trainer.validate(model, datamodule=self.data_module)

            self.results.append({
                'fold': fold,
                'val_results': val_results
            })