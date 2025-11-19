"""
Simple LLaVA-style Image-to-Text Trainer using Lightning.

- Loads an arbitrary causal LM from Hugging Face (AutoModelForCausalLM)
- Uses an injected vision encoder to extract visual features
- Treats vision patches as a token sequence (no pooling)
- Projects vision token features to LM hidden size and prepends them as a prefix token sequence
- Supports freezing the LM or fine-tuning with LoRA
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
from peft import LoraConfig, get_peft_model

from einops import rearrange

# from model.dummy_vision_encoder import DummyVisionEncoder
from model.vision_encoders import VisionEncoderWrapper
# from datamodule.dummy_datamodule import LLaVADataModule
from datamodule.cc15m_datamodule import CCWebDatasetDataModule


class LLaVATrainModule(LightningModule):
    """
    Simple LLaVA-style LightningModule.

    Expected config example:
    config = {
        "lm_model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "lr": 1e-4,
        "weight_decay": 0.01,
        "warmup_steps": 1000,
        "vision_hidden_size": 2048,  # feature dim of vision_model output
        "freeze_vision": True,
        "freeze_lm": False,
        "use_lora": True,
        "vision_layer": -1,         # which layer index to use from vision_model outputs (if applicable)
    }

    The vision_model is injected from outside and should output either:
    - [B, T_v, V_DIM]
    - [B, C, H, W] (which will be reshaped to [B, T_v, V_DIM])
    - or a dict/object that contains "last_hidden_state" etc.

    The text_model can be a plain AutoModelForCausalLM or a PEFT LoRA-wrapped model.

    Vision tokens are projected and used as a prefix token sequence for the LM
    (no pooling to a single vector).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        vision_model: nn.Module,
        text_tokenizer: AutoTokenizer,
        text_model: AutoModelForCausalLM,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["vision_model", "text_tokenizer", "text_model"])
        self.config = config

        # Text tokenizer & LM (possibly LoRA-wrapped)
        self.text_tokenizer = text_tokenizer
        self.text_model = text_model

        lm_hidden_size = self.text_model.config.hidden_size

        # Vision backbone + projection
        self.vision_model = vision_model
        vision_hidden_size = config["vision_hidden_size"]

        # Linear projection for each vision token: V_DIM -> LM_HIDDEN
        self.vision_proj = nn.Linear(vision_hidden_size, lm_hidden_size)

        # Freeze only the vision backbone if requested
        if config.get("freeze_vision", False):
            for p in self.vision_model.parameters():
                p.requires_grad = False

        # Projection is always trainable
        for p in self.vision_proj.parameters():
            p.requires_grad = True

        # Optionally freeze language model (only when not using LoRA)
        if config.get("freeze_lm", False) and not config.get("use_lora", False):
            for p in self.text_model.parameters():
                p.requires_grad = False

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode images into a sequence of vision token features.

        This method:
        - Runs the vision backbone (optionally frozen)
        - Selects a layer if the vision model returns a list of features
        - Converts 4D features to [B, T_v, V_DIM] if necessary
        - Returns a tensor of shape [B, T_v, V_DIM]
        """
        # Run vision backbone with or without gradient based on freeze_vision
        with torch.set_grad_enabled(not self.config.get("freeze_vision", False)):
            if self.config.get("freeze_vision", False):
                self.vision_model.eval()
            vision_out = self.vision_model(pixel_values)

        # If the vision model returns a list or tuple of layers, select one
        vision_layer_idx = self.config.get("vision_layer", -1)
        if isinstance(vision_out, (list, tuple)):
            vision_out = vision_out[vision_layer_idx]

        # If 4D, try to reshape to [B, T_v, V_DIM]
        # Adjust this logic to match your VisionEncoderWrapper output format.
        if isinstance(vision_out, torch.Tensor) and vision_out.dim() == 4:
            # Example expected shape from VisionEncoderWrapper: [B, 1, num_patches, dim]
            # Rearrange to [B, num_patches, dim]
            # If your actual shape is [B, C, H, W], you should instead use:
            #   vision_out = rearrange(vision_out, "b c h w -> b (h w) c")
            try:
                vision_out = rearrange(vision_out, "b 1 patch dim -> b patch dim")
            except Exception:
                # Fallback: assume [B, C, H, W]
                vision_out = rearrange(vision_out, "b c h w -> b (h w) c")

        # Now handle dict / object-style outputs
        if isinstance(vision_out, dict):
            if "last_hidden_state" in vision_out:
                seq_feats = vision_out["last_hidden_state"]  # [B, T_v, V_DIM]
            elif "pooler_output" in vision_out:
                pooled = vision_out["pooler_output"]         # [B, V_DIM]
                seq_feats = pooled.unsqueeze(1)              # [B, 1, V_DIM]
            else:
                # Fallback: take the first value
                seq_feats = next(iter(vision_out.values()))
                if seq_feats.dim() == 2:
                    seq_feats = seq_feats.unsqueeze(1)
        else:
            # Non-dict output: assume tensor or object with last_hidden_state
            if hasattr(vision_out, "last_hidden_state"):
                seq_feats = vision_out.last_hidden_state      # [B, T_v, V_DIM]
            else:
                feats = vision_out
                if feats.dim() == 2:
                    # [B, V_DIM] -> [B, 1, V_DIM]
                    seq_feats = feats.unsqueeze(1)
                else:
                    # Assume already [B, T_v, V_DIM]
                    seq_feats = feats

        # seq_feats: [B, T_v, V_DIM]
        assert seq_feats.dim() == 3, f"Expected [B, T_v, V_DIM], got {seq_feats.shape}"
        return seq_feats

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass:
        - Encode images into a sequence of vision token features [B, T_v, V_DIM]
        - Project to LM hidden size: [B, T_v, H]
        - Prepend vision embeddings as a prefix token sequence
        - Call LM with inputs_embeds, attention_mask, and labels
        """
        # 1) Encode image to vision token sequence and project to LM hidden size
        vision_tokens = self.encode_image(pixel_values)       # [B, T_v, V_DIM]
        vision_embeds = self.vision_proj(vision_tokens)       # [B, T_v, H]

        # 2) Token embeddings from LM
        input_embeds = self.text_model.get_input_embeddings()(input_ids)  # [B, T, H]

        # 3) Concatenate vision prefix and text embeddings
        inputs_embeds = torch.cat([vision_embeds, input_embeds], dim=1)   # [B, T_v+T, H]

        # 4) Extend attention mask for the vision tokens
        bsz = attention_mask.size(0)
        num_vision_tokens = vision_embeds.size(1)
        vision_mask = torch.ones(
            bsz, num_vision_tokens, dtype=attention_mask.dtype, device=attention_mask.device
        )
        attention_mask = torch.cat([vision_mask, attention_mask], dim=1)  # [B, T_v+T]

        # 5) Extend labels; ignore vision positions in the loss
        if labels is not None:
            vision_labels = torch.full(
                (bsz, num_vision_tokens),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            labels = torch.cat([vision_labels, labels], dim=1)            # [B, T_v+T]

        # 6) LM forward (LM internally computes causal LM loss)
        outputs = self.text_model(
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
        if warmup_steps <= 0 or self.trainer is None:
            return optimizer

        # Linear warmup + linear decay scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.config["num_training_step"],
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
    # Set matmul precision for A100 Tensor Cores
    torch.set_float32_matmul_precision("high")

    # Example config for LLaMA 3 + LoRA
    config = {
        # Replace with the exact LLaMA 3 checkpoint you have access to
        "lm_model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "lr": 1e-4,
        "weight_decay": 0.01,
        "warmup_steps": 100,
        "vision_hidden_size": 2048,
        "freeze_vision": False,
        "freeze_lm": False,   # Set True if you want to freeze LM entirely (when not using LoRA)
        "use_lora": True,     # Set True to enable LoRA, False for plain LM
        "vision_layer": -1,   # Which layer index to use from the vision encoder outputs
        "num_workers": 8,
        "shuffle_buffer": 10000,
        "resampled": False,
        "max_length": 32,
        "training_epochs": 1,
        "batch_size": 16,
        "devices": 2,
        # LoRA hyperparameters
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
    }
    # Set number of training steps
    # Use CC3M only
    num_training_step = 3310000 * config["training_epochs"] // (config["batch_size"] * config['devices'])
    config["num_training_step"] = num_training_step

    # 1) Build vision encoder
    # vision_encoder = DummyVisionEncoder(
    #     vision_hidden_size=config["vision_hidden_size"]
    # )
    vision_encoder = VisionEncoderWrapper(
        vision_encoder_name="vggt",
    )

    # Freeze vision encoder
    for param in vision_encoder.parameters():
        param.requires_grad = False

    # 2) Build text tokenizer
    text_tokenizer = AutoTokenizer.from_pretrained(
        config["lm_model_name"],
        use_fast=True,
    )
    # Ensure pad_token exists
    if text_tokenizer.pad_token is None:
        if text_tokenizer.eos_token is not None:
            text_tokenizer.pad_token = text_tokenizer.eos_token
        else:
            text_tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # 3) Build DataModule
    # Dummy DataModule Example
    # datamodule = LLaVADataModule(
    #     tokenizer=text_tokenizer,
    #     train_length=1000,
    #     val_length=200,
    #     test_length=200,
    #     max_length=32,
    #     batch_size=4,
    #     num_workers=4,
    # )
    
    datamodule = CCWebDatasetDataModule(
        tokenizer=text_tokenizer,
        train_shards="data/cc3m/{00002..00331}.tar",
        val_shards="data/cc3m/{00000..00001}.tar",
        # image_transform=image_transform,
        max_length=config["max_length"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle_buffer=config["shuffle_buffer"],
        resampled=config["resampled"],
        num_training_step=config["num_training_step"],
    )

    # 4) Build base text model (LLaMA 3)
    text_model = AutoModelForCausalLM.from_pretrained(
        config["lm_model_name"],
        dtype=torch.bfloat16,  # matches bf16-mixed
    )

    # Resize embeddings if we added new special tokens
    text_model.resize_token_embeddings(len(text_tokenizer))

    # 5) Optionally wrap with LoRA
    if config.get("use_lora", False):
        # Typical target modules for LLaMA-style models
        lora_target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        lora_config = LoraConfig(
            r=config.get("lora_r", 8),
            lora_alpha=config.get("lora_alpha", 16),
            lora_dropout=config.get("lora_dropout", 0.05),
            target_modules=lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        text_model = get_peft_model(text_model, lora_config)
        text_model.print_trainable_parameters()

    # 6) Build LLaVA-style module
    model = LLaVATrainModule(
        config=config,
        vision_model=vision_encoder,
        text_tokenizer=text_tokenizer,
        text_model=text_model,
    )

    # 7) Lightning trainer
    trainer = Trainer(
        max_epochs=config["training_epochs"],
        devices=config["devices"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        log_every_n_steps=1,
    )

    # 8) Train with DataModule
    trainer.fit(model, datamodule=datamodule)