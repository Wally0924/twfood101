from pathlib import Path

from torch.utils.data import Dataset, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm
from PIL import Image as IM, ImageFile
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

class TWFoodDataset(Dataset): #Read data & preprocess101
    def __init__(self, images: list, labels: list=[], transform=None):
        # Read the CSV file into a DataFrame
        self.images = images 
        self.labels = labels
        self.transform = transform

    def __len__(self):
        # Return the total number of samples
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = IM.open(image_path)

        if image.mode == 'P':
            image = image.convert('RGBA')
            image = image.convert('RGB')
        else:
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)
        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        else:
            return image


class KFoldFood101DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = './data', ds_file: str= 'tw_food_101_train.csv', batch_size: int = 32, num_workers: int = 4, image_size: int = 224, num_folds: int = 3):
        super().__init__()
        self.data_dir = data_dir
        self.ds_file = ds_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.num_classes = 101 # Food-101 has 101 classes
        self.num_folds = num_folds

        # Define transformations
        # Get normalization statistics from timm model config if available, otherwise use ImageNet defaults
        # For Swin V2, ImageNet defaults are usually fine.
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.TrivialAugmentWide(), # Good general augmentation
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(self.image_size), # Resize slightly larger
                transforms.CenterCrop(self.image_size), # Crop to target size
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        self.current_fold = 0

    def _setup_folds(self) -> None:
        """設置K-fold索引"""
        # 創建KFold物件
        self.kfold = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)

        # 獲取所有折疊的訓練和驗證索引
        self.folds = []

        for train_idx, val_idx in self.kfold.split(self.dataset):
            self.folds.append({
                'train': train_idx,
                'val': val_idx
            })

    def prepare_data(self):
        # Load train data
        train_df = pd.read_csv(Path(self.data_dir).joinpath(self.ds_file), header=None)
        self.train_images = [Path(self.data_dir).joinpath(row[2]) for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Loading train image paths") if Path(self.data_dir).joinpath(row[2]).exists()]
        self.train_labels = [int(row[1]) for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Loading train labels") if Path(self.data_dir).joinpath(row[2]).exists()]

    def setup(self, stage: str | None = None):
        if stage == 'fit' or stage is None:
            self.dataset = TWFoodDataset(
                images=self.train_images,
                labels=self.train_labels,
                transform=self.data_transforms['train']
            )
            self._setup_folds()

        if stage != 'fit' and stage is not None:
             print(f"Stage {stage} not explicitly handled in setup.")

    def _get_fold_dataloader(self, fold_indices: np.ndarray, shuffle: bool) -> DataLoader:
        """創建指定折疊索引的DataLoader"""
        subset = Subset(self.dataset, fold_indices)
        return DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def train_dataloader(self) -> DataLoader:
        """返回當前折疊的訓練DataLoader"""
        train_indices = self.folds[self.current_fold]['train']
        return self._get_fold_dataloader(train_indices, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """返回當前折疊的驗證DataLoader"""
        val_indices = self.folds[self.current_fold]['val']
        return self._get_fold_dataloader(val_indices, shuffle=False)

    def set_fold(self, fold_index: int) -> None:
        """設置當前使用的折疊"""
        if fold_index < 0 or fold_index >= self.num_folds:
            raise ValueError(f"Fold index must be between 0 and {self.num_folds-1}")
        self.current_fold = fold_index

class TestFood101DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = './data', ds_file: str='tw_food_101_test_list.csv', batch_size: int = 32, num_workers: int = 4, image_size: int = 224):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.num_classes = 101 # Food-101 has 101 classes

        # Define transformations
        # Get normalization statistics from timm model config if available, otherwise use ImageNet defaults
        # For Swin V2, ImageNet defaults are usually fine.
        self.data_transforms = {
            'test': transforms.Compose([
                transforms.Resize(self.image_size), # Resize slightly larger
                transforms.CenterCrop(self.image_size), # Crop to target size
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

    def prepare_data(self):
        # Load test data
        test_df = pd.read_csv(Path(self.data_dir).joinpath(self.test_ds_file), header=None)
        self.test_images = [Path(self.data_dir).joinpath(row[1]) for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Loading test image paths") if Path(self.data_dir).joinpath(row[1]).exists()]

    def setup(self, stage: str | None = None):
        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_dataset = TWFoodDataset(
                images=self.test_images,
                labels=None,
                transform=self.data_transforms['test']
            )

        if stage != 'test' and stage is not None:
             print(f"Stage {stage} not explicitly handled in setup.")

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=True if self.num_workers > 0 else False)
