import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import timm # PyTorch Image Models library
import torchmetrics

class SwinFoodClassifier(pl.LightningModule):
    def __init__(self, model_name: str = 'swinv2_base_patch4_window7_224.ms_in22k', num_classes: int = 101, learning_rate: float = 1e-4, pretrained: bool = True):
        super().__init__()
        self.save_hyperparameters() # Saves args to self.hparams

        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # Load pretrained Swin Transformer V2 model
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=self.num_classes) # num_classes=0 removes the original classifier

        # Get the number of input features for the original classifier
        # num_ftrs = self.model.get_classifier().in_features # More general way to get classifier input features

        # Replace the classifier head with a new one for Food-101
        self.model.reset_classifier(num_classes=self.num_classes) # timm's way to replace head


        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        return loss, preds, labels

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self._common_step(batch, batch_idx)
        acc = self.train_accuracy(preds, labels)
        # Use prog_bar=True to display metrics in the progress bar
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._common_step(batch, batch_idx)
        acc = self.val_accuracy(preds, labels)
        self.log('val_loss', loss, on_epoch=True, logger=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, logger=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        """
        使用 trainer.test() 進行測試評估，會自動計算您在 test_step 中定義的指標
        """
        loss, preds, labels = self._common_step(batch, batch_idx)
        acc = self.test_accuracy(preds, labels)
        self.log('test_loss', loss, on_epoch=True, logger=True)
        self.log('test_acc', acc, on_epoch=True, logger=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        """
        獲取預測結果
        """
        imgs = batch
        outputs = self(imgs)
        preds = torch.argmax(outputs, dim=1)
        return preds

    def configure_optimizers(self):
        # AdamW is often recommended for transformer models
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        # Optional: Add a learning rate scheduler
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5, last_epoch=-1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss", # Metric to monitor for scheduler
                "frequency": 1 # Check every epoch
            },
        }