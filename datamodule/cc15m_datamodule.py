import os
from typing import Optional, Dict, Any, Callable, List, Union, Sequence

import torch
from torch.utils.data import DataLoader
from lightning import LightningDataModule

import webdataset as wds
from transformers import AutoTokenizer


class CCWebDatasetDataModule(LightningDataModule):
    """
    LightningDataModule for CC-style datasets (e.g., CC3M + CC12M) using WebDataset.

    It can take multiple shard patterns or lists, e.g.:

        train_shards = [
            "/data/cc3m/train-{00000..00999}.tar",
            "/data/cc12m/train-{00000..01999}.tar",
        ]

    WebDataset will expand and stream all of them as one dataset.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        train_shards: Union[str, Sequence[str]],
        val_shards: Optional[Union[str, Sequence[str]]] = None,
        test_shards: Optional[Union[str, Sequence[str]]] = None,
        image_transform: Optional[Callable] = None,
        max_length: int = 32,
        batch_size: int = 4,
        num_workers: int = 4,
        shuffle_buffer: int = 10_000,
        resampled: bool = False,
        num_training_step: int = 1000000,
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer (pad_token must be set or will be set here).
            train_shards: String pattern or list/tuple of patterns/urls for training shards.
            val_shards: String pattern or list/tuple for validation shards.
            test_shards: String pattern or list/tuple for test shards.
            image_transform: Callable that takes a PIL image and returns a tensor [C,H,W].
            max_length: Max token length for caption text.
            batch_size: Batch size for all splits.
            num_workers: Number of DataLoader workers.
            shuffle_buffer: Buffer size used in WebDataset.shuffle.
            resampled: If True, use ResampledShards for infinite sampling (useful for very large datasets).
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.train_shards = train_shards
        self.val_shards = val_shards
        self.test_shards = test_shards
        self.image_transform = image_transform
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_buffer = shuffle_buffer
        self.resampled = resampled
        self.num_training_step = num_training_step

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Ensure pad_token exists
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # --------- Small helper --------- #

    @staticmethod
    def _normalize_shards(shards: Union[str, Sequence[str]]) -> Union[str, List[str]]:
        """
        Normalize shard specification to either a string pattern or a list of strings.

        WebDataset supports:
            - a single brace pattern string, e.g. "train-{00000..09999}.tar"
            - a list of explicit shard URLs / patterns
        """
        if isinstance(shards, (list, tuple)):
            return list(shards)
        return shards

    def prepare_data(self) -> None:
        """
        Called only on one process. No heavy work needed here, since shards already exist on disk.
        """
        pass

    # --------- WebDataset pipeline builders --------- #

    def _build_webdataset(
        self,
        shards: Union[str, Sequence[str]],
        is_train: bool = True,
    ) -> wds.WebDataset:
        """
        Build a WebDataset pipeline for given shards.

        This assumes that each sample has:
            - "jpg": image file
            - "txt": caption text file
        """
        shards_norm = self._normalize_shards(shards)

        if self.resampled:
            dataset = wds.WebDataset(
                wds.ResampledShards(shards_norm),
                shardshuffle=is_train,
                nodesplitter=wds.split_by_node,
            )
        else:
            dataset = wds.WebDataset(
                shards_norm,
                shardshuffle=is_train,
                nodesplitter=wds.split_by_node,
            )

        if is_train:
            dataset = dataset.shuffle(self.shuffle_buffer)

        # Decode images as PIL
        dataset = dataset.decode("pil")

        # Extract (image, text) tuple from keys "jpg" and "txt"
        dataset = dataset.to_tuple("jpg", "txt")

        # Map to our training dict format
        dataset = dataset.map(self._preprocess_sample)

        return dataset

    def _preprocess_sample(self, sample: List[Any]) -> Dict[str, torch.Tensor]:
        """
        Convert a (PIL.Image, raw_text) sample into model-ready tensors.
        """
        image, caption = sample  # image: PIL, caption: str or bytes

        # Ensure caption is string
        if isinstance(caption, bytes):
            caption = caption.decode("utf-8", errors="ignore")

        # Apply image transform
        if self.image_transform is not None:
            pixel_values = self.image_transform(image)  # [C, H, W]
        else:
            import torchvision.transforms as T

            default_transform = T.Compose(
                [
                    T.Resize((224, 224)),
                    T.ToTensor(),  # [0, 1]
                ]
            )
            pixel_values = default_transform(image)

        # Tokenize caption
        enc = self.tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = enc.input_ids[0]            # [T]
        attention_mask = enc.attention_mask[0]  # [T]
        labels = input_ids.clone()              # typical causal LM training

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    # --------- Setup for different stages --------- #

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Create WebDataset pipelines for each stage: 'fit', 'validate', 'test'.
        """
        if stage == "fit" or stage is None:
            assert self.train_shards is not None, "train_shards must be provided for stage 'fit'."
            self.train_dataset = self._build_webdataset(self.train_shards, is_train=True)

            if self.val_shards is not None:
                self.val_dataset = self._build_webdataset(self.val_shards, is_train=False)

        if stage == "validate":
            if self.val_shards is not None and self.val_dataset is None:
                self.val_dataset = self._build_webdataset(self.val_shards, is_train=False)

        if stage == "test" or stage is None:
            if self.test_shards is not None:
                self.test_dataset = self._build_webdataset(self.test_shards, is_train=False)

    # --------- Dataloaders --------- #

    def train_dataloader(self) -> DataLoader:
        """
        Return WebDataset-based train dataloader.
        """
        assert self.train_dataset is not None, "Train dataset is not initialized. Call setup('fit') first."

        loader = wds.WebLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        loader = loader.with_length(self.num_training_step)

        return loader

    def val_dataloader(self) -> DataLoader:
        """
        Return WebDataset-based validation dataloader.
        """
        if self.val_dataset is None:
            if self.val_shards is None:
                raise ValueError("val_shards is not provided, cannot build val_dataloader.")
            self.val_dataset = self._build_webdataset(self.val_shards, is_train=False)

        loader = wds.WebLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return loader

    def test_dataloader(self) -> DataLoader:
        """
        Return WebDataset-based test dataloader.
        """
        if self.test_dataset is None:
            if self.test_shards is None:
                raise ValueError("test_shards is not provided, cannot build test_dataloader.")
            self.test_dataset = self._build_webdataset(self.test_shards, is_train=False)

        loader = wds.WebLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return loader