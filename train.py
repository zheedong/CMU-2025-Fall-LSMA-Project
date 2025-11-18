"""
Simple LLaVA-style Image-to-Text Trainer using Lightning.

- Loads an arbitrary causal LM from Hugging Face (AutoModelForCausalLM)
- Uses an injected vision encoder to extract visual features
- Applies a learnable pooling layer on vision tokens
- Projects pooled vision features to LM hidden size and prepends them as a prefix token
- Trains with standard causal LM loss (labels = shifted input_ids)
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from lightning import LightningModule, Trainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

from model.dummy_vision_encoder import DummyVisionEncoder
from datamodule.dummy_datamodule import DummyDataModule

# =========================
# Vision pooling & LLaVA module
# =========================

class VisionPoolingHead(nn.Module):
    """
    Learnable attention pooling over vision tokens.
    Input:  x [B, T, H]
    Output: pooled [B, H]
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H]
        scores = self.attn(x)  # [B, T, 1]
        attn_weights = torch.softmax(scores, dim=1)  # softmax over T
        pooled = (attn_weights * x).sum(dim=1)  # [B, H]
        return pooled


class LLaVATrainModule(LightningModule):
    """
    Simple LLaVA-style LightningModule.

    Expected config example:
    config = {
        "lm_model_name": "gpt2",
        "lr": 1e-4,
        "weight_decay": 0.01,
        "warmup_steps": 1000,
        "vision_hidden_size": 1024,  # feature dim of vision_model output
        "freeze_vision": True,
        "freeze_lm": False,
    }

    The vision_model is injected from outside and should output either:
    - [B, V_DIM]
    - [B, T, V_DIM]
    or a dict with "last_hidden_state" / "pooler_output" etc.
    """

    def __init__(self, config: Dict[str, Any], vision_model: nn.Module, text_tokenizer: AutoTokenizer, text_model: AutoModelForCausalLM):
        super().__init__()
        self.save_hyperparameters(ignore=["vision_model"])
        self.config = config

        self.text_tokenizer = text_tokenizer
        self.text_model = text_model

        lm_hidden_size = self.text_model.config.hidden_size

        # 2) Vision backbone + learnable pooling + projection
        self.vision_model = vision_model
        vision_hidden_size = config["vision_hidden_size"]

        # Learnable pooling head on top of vision tokens
        self.vision_pool = VisionPoolingHead(vision_hidden_size)

        # Projection from vision_hidden_size to LM hidden size
        self.vision_proj = nn.Linear(vision_hidden_size, lm_hidden_size)

        # Freeze only the vision backbone if requested
        if config.get("freeze_vision", False):
            for p in self.vision_model.parameters():
                p.requires_grad = False

        # Pooling & projection are always trainable
        for p in self.vision_pool.parameters():
            p.requires_grad = True
        for p in self.vision_proj.parameters():
            p.requires_grad = True

        # Optionally freeze language model
        if config.get("freeze_lm", False):
            for p in self.lm_model.parameters():
                p.requires_grad = False

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode images into a single pooled feature vector per image.

        This method:
        - Runs the vision backbone (optionally frozen)
        - Handles several common output formats (dict, tuple, plain tensor)
        - Applies a learnable attention pooling over token features
        """
        # Run vision backbone with or without gradient based on freeze_vision
        with torch.set_grad_enabled(not self.config.get("freeze_vision", False)):
            vision_out = self.vision_model(pixel_values)

        # Extract sequence features [B, T, V_DIM] from various output formats
        if isinstance(vision_out, dict):
            if "last_hidden_state" in vision_out:
                seq_feats = vision_out["last_hidden_state"]  # [B, T, V_DIM]
            elif "pooler_output" in vision_out:
                pooled = vision_out["pooler_output"]  # [B, V_DIM]
                seq_feats = pooled.unsqueeze(1)  # [B, 1, V_DIM]
            else:
                # Fallback: take the first value
                seq_feats = next(iter(vision_out.values()))
                if seq_feats.dim() == 2:
                    seq_feats = seq_feats.unsqueeze(1)
        else:
            # Non-dict output: assume tensor or object with last_hidden_state
            if hasattr(vision_out, "last_hidden_state"):
                seq_feats = vision_out.last_hidden_state  # [B, T, V_DIM]
            else:
                feats = vision_out
                if feats.dim() == 2:
                    # [B, V_DIM] -> [B, 1, V_DIM]
                    seq_feats = feats.unsqueeze(1)
                else:
                    # Assume already [B, T, V_DIM]
                    seq_feats = feats

        # seq_feats: [B, T, V_DIM]
        pooled_feats = self.vision_pool(seq_feats)  # [B, V_DIM]
        return pooled_feats

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass:
        - Encode images into a pooled vision feature
        - Project to LM hidden size
        - Prepend vision embedding as one prefix token
        - Call LM with inputs_embeds, attention_mask, and labels
        """
        # 1) Encode image and project to LM hidden size
        vision_feats = self.encode_image(pixel_values)    # [B, V_DIM]
        vision_embeds = self.vision_proj(vision_feats)    # [B, H]
        vision_embeds = vision_embeds.unsqueeze(1)        # [B, 1, H]

        # 2) Token embeddings from LM
        input_embeds = self.lm_model.get_input_embeddings()(input_ids)  # [B, T, H]

        # 3) Concatenate vision prefix and text embeddings
        inputs_embeds = torch.cat([vision_embeds, input_embeds], dim=1)  # [B, 1+T, H]

        # 4) Extend attention mask for the prefix token
        bsz = attention_mask.size(0)
        prefix_mask = torch.ones(
            bsz, 1, dtype=attention_mask.dtype, device=attention_mask.device
        )
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)  # [B, 1+T]

        # 5) Extend labels; ignore prefix position in the loss
        if labels is not None:
            prefix_labels = torch.full(
                (bsz, 1),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            labels = torch.cat([prefix_labels, labels], dim=1)  # [B, 1+T]

        # 6) LM forward (LM internally computes causal LM loss)
        outputs = self.lm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """
        Training step for I2T:
        batch: {
            "pixel_values": [B, 3, H, W],
            "input_ids": [B, T],
            "attention_mask": [B, T],
            "labels": [B, T]
        }
        """
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = self(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """
        Validation step with the same loss as training.
        """
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = self(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """
        Configure optimizer and (optional) linear warmup scheduler.
        """
        lr = self.config.get("lr", 1e-4)
        weight_decay = self.config.get("weight_decay", 0.01)
        warmup_steps = self.config.get("warmup_steps", 0)

        # Only trainable parameters
        params = [p for p in self.parameters() if p.requires_grad]

        optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)

        # If no warmup or trainer not yet available, return optimizer only
        if warmup_steps <= 0:
            return optimizer

        # Linear warmup + linear decay scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


# =========================
# Main: training with DataModule
# =========================

if __name__ == "__main__":
    # Example config
    config = {
        "lm_model_name": "gpt2",
        "lr": 1e-4,
        "weight_decay": 0.01,
        "warmup_steps": 100,
        "vision_hidden_size": 1024,
        "freeze_vision": False,
        "freeze_lm": False,
    }

    # 1) Build DataModule (will internally create tokenizer and datasets)
    datamodule = DummyDataModule(
        tokenizer_name=config["lm_model_name"],
        train_length=1000,
        val_length=200,
        test_length=200,
        max_length=32,
        batch_size=4,
        num_workers=4,
    )

    # 2) Build vision encoder
    vision_encoder = DummyVisionEncoder(
        vision_hidden_size=config["vision_hidden_size"]
    )

    # Build Text Tokenizer
    text_tokenizer = AutoTokenizer.from_pretrained(config["lm_model_name"])
    if text_tokenizer.pad_token is None:
        # Many causal LMs have no pad_token, so we reuse eos_token
        text_tokenizer.pad_token = text_tokenizer.eos_token

    # Build Text Model
    text_model = AutoModelForCausalLM.from_pretrained(config["lm_model_name"])

    # 3) Build LLaVA-style module
    model = LLaVATrainModule(
        config=config,
        vision_model=vision_encoder,
        text_tokenizer=text_tokenizer,
        text_model=text_model,
    )

    # 4) Lightning trainer
    trainer = Trainer(
        max_epochs=3,
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        log_every_n_steps=10,
    )

    # 5) Train with DataModule
    trainer.fit(model, datamodule=datamodule)