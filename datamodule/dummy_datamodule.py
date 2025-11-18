from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, Optional
import torch
from lightning import LightningDataModule

# =========================
# Dummy I2T Dataset
# =========================

class DummyI2TDataset(Dataset):
    """
    Dummy I2T dataset that returns random images and synthetic captions.
    Used only for verifying the training pipeline.
    """

    def __init__(self, tokenizer: AutoTokenizer, length: int = 1000, max_length: int = 32):
        super().__init__()
        self.tokenizer = tokenizer
        self.length = length
        self.max_length = max_length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Random image
        pixel_values = torch.randn(3, 224, 224)

        # Synthetic caption text
        text = f"this is a dummy caption number {idx}"
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = enc.input_ids[0]            # [T]
        attention_mask = enc.attention_mask[0]  # [T]
        labels = input_ids.clone()             # typical causal LM training: labels = input_ids

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# =========================
# Lightning DataModule
# =========================

class LLaVADataModule(LightningDataModule):
    """
    LightningDataModule for I2T training.

    It creates DummyI2TDataset for train/val/test
    and returns corresponding dataloaders.
    """

    def __init__(
        self,
        tokenizer_name: str,
        train_length: int = 1000,
        val_length: int = 200,
        test_length: int = 200,
        max_length: int = 32,
        batch_size: int = 4,
        num_workers: int = 4,
    ):
        super().__init__()
        self.tokenizer_name = tokenizer_name
        self.train_length = train_length
        self.val_length = val_length
        self.test_length = test_length
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.tokenizer: Optional[AutoTokenizer] = None
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def prepare_data(self) -> None:
        """
        Download tokenizer or any external resources if needed.
        This is called only from a single process.
        """
        AutoTokenizer.from_pretrained(self.tokenizer_name)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Create datasets for each stage: 'fit', 'validate', 'test', 'predict'.
        """
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        if stage == "fit" or stage is None:
            self.train_dataset = DummyI2TDataset(
                tokenizer=self.tokenizer,
                length=self.train_length,
                max_length=self.max_length,
            )
            self.val_dataset = DummyI2TDataset(
                tokenizer=self.tokenizer,
                length=self.val_length,
                max_length=self.max_length,
            )

        if stage == "validate":
            self.val_dataset = DummyI2TDataset(
                tokenizer=self.tokenizer,
                length=self.val_length,
                max_length=self.max_length,
            )

        if stage == "test" or stage is None:
            self.test_dataset = DummyI2TDataset(
                tokenizer=self.tokenizer,
                length=self.test_length,
                max_length=self.max_length,
            )

    def train_dataloader(self) -> DataLoader:
        """
        Return train dataloader.
        """
        assert self.train_dataset is not None, "Train dataset is not initialized. Call setup('fit') first."
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Return validation dataloader.
        """
        assert self.val_dataset is not None, "Validation dataset is not initialized. Call setup('fit') first."
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Return test dataloader.
        """
        assert self.test_dataset is not None, "Test dataset is not initialized. Call setup('test') first."
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
