"""Sanity tests for BCTDataset and get_bct_dataloader."""

import torch
import pytest


def _make_config(model_name="sshleifer/tiny-gpt2", max_length=128, mix_instruct=False):
    return {
        "model": {"name": model_name, "output_attentions": False, "output_hidden_states": False},
        "lora": {"r": 1, "lora_alpha": 2, "lora_dropout": 0.0,
                 "target_modules": ["c_attn"], "bias": "none"},
        "training": {"epochs": 1, "learning_rate": 1e-4, "grad_clip": 1.0,
                     "log_every_n_steps": 1, "save_dir": None},
        "data": {
            "bct_root": "datasets/sycophancy_bct",
            "mix_instruct": mix_instruct,
            "batch_size": 1,
            "max_length": max_length,
        },
    }


def test_dataloader_shapes():
    from data.attct_datasets import get_bct_dataloader
    config = _make_config()
    dl = get_bct_dataloader(config, split="train")
    batch = next(iter(dl))

    assert set(batch.keys()) == {"input_ids", "attention_mask", "labels"}
    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["input_ids"].dtype == torch.long
    assert batch["labels"].dtype == torch.long


def test_labels_masked():
    """Question tokens must be masked to -100; at least some response tokens must be visible."""
    from data.attct_datasets import get_bct_dataloader
    config = _make_config()
    dl = get_bct_dataloader(config, split="train")
    batch = next(iter(dl))

    labels = batch["labels"][0]
    n_masked  = (labels == -100).sum().item()
    n_visible = (labels != -100).sum().item()
    assert n_masked  > 0, "Expected some masked (question) tokens"
    assert n_visible > 0, "Expected some visible (response) tokens"


def test_sft_loss_forward():
    """SFTLoss should run a forward pass and return a scalar loss."""
    from data.attct_datasets import get_bct_dataloader
    from losses.losses import SFTLoss
    from transformers import AutoModelForCausalLM
    from peft import get_peft_model, LoraConfig, TaskType

    config = _make_config()
    dl = get_bct_dataloader(config, split="train")
    batch = next(iter(dl))

    model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=1, lora_alpha=2,
        target_modules=["c_attn"], bias="none",
    ))
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

    loss_fn = SFTLoss()
    result = loss_fn(logits=outputs.logits, labels=batch["labels"])

    assert "loss" in result
    assert result["loss"].ndim == 0, "Loss should be a scalar"
    assert torch.isfinite(result["loss"]), "Loss should be finite"


def test_bct_trainer_step():
    """BCTTrainer._step should run without error and return a finite loss."""
    from data.attct_datasets import get_bct_dataloader
    from losses.losses import SFTLoss
    from train import BCTTrainer
    from transformers import AutoModelForCausalLM
    from peft import get_peft_model, LoraConfig, TaskType
    import wandb

    wandb.init(mode="disabled")

    config = _make_config()
    dl = get_bct_dataloader(config, split="train")

    model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=1, lora_alpha=2,
        target_modules=["c_attn"], bias="none",
    ))

    device = torch.device("cpu")
    loss_fn = SFTLoss()
    trainer = BCTTrainer(model, dl, loss_fn, config, device)

    batch = next(iter(dl))
    result = trainer._step(batch)

    assert "loss" in result
    assert torch.isfinite(result["loss"])
