import pytorch_lightning as pl

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
        for fold in range(self.data_module.num_folds):
            print(f"Training fold {fold+1}/{self.data_module.num_folds}")
            
            # 設置當前折疊
            self.data_module.set_fold(fold)
            
            # 初始化模型
            model = self.model_class(**self.model_kwargs)
            
            # 初始化訓練器
            trainer = pl.Trainer(**self.trainer_kwargs)
            
            # 訓練模型
            trainer.fit(model, datamodule=self.data_module)
            
            # 驗證模型
            val_results = trainer.validate(model, datamodule=self.data_module)
            
            # 保存結果
            self.results.append({
                'fold': fold,
                'val_results': val_results
            })
        
        return self.results