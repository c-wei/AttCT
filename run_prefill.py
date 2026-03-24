print("LOADED FILE:", __file__)

from datasets import load_dataset
print("LOAD_DATASET EXISTS:", load_dataset)

import torch
import wandb
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

from train import Trainer
from losses.losses import AttentionConsistencyLoss
from data.prefill_dataset import get_prefill_dataloader, load_wildjailbreak_prompts

with open("./configs/prefill_act.yaml") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = config["model"]["name"]

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

lora_cfg = config["lora"]
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    attn_implementation="eager",
)

model = get_peft_model(model, LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=lora_cfg["r"],
    lora_alpha=lora_cfg["lora_alpha"],
    lora_dropout=lora_cfg["lora_dropout"],
    target_modules=lora_cfg["target_modules"],
    bias=lora_cfg["bias"],
))
model.print_trainable_parameters()
model = model.to(device)

# hf_dataset = load_dataset("allenai/wildjailbreak", split="train", streaming=True)
# limit = config["data"].get("limit")

# prompts = []
# for item in hf_dataset:
#     content = item["content"]
#     text = content[0] if isinstance(content, list) else content
#     prompts.append(text)
#     if limit is not None and len(prompts) >= limit:
#         break

# print(f"Loaded {len(prompts)} harmful prompts from wildjailbreak")

# split = int(0.9 * len(prompts))
# train_prompts = prompts[:split]
# eval_prompts  = prompts[split:]

# train_dl = get_prefill_dataloader(
#     train_prompts, tokenizer,
#     batch_size=config["training"].get("batch_size", 1),
# )
# eval_dl = get_prefill_dataloader(
#     eval_prompts, tokenizer, shuffle=False,
#     batch_size=config["training"].get("batch_size", 1),
# )

# print(f"Train pairs: {len(train_dl.dataset)} | Eval pairs: {len(eval_dl.dataset)}")

train_prompts, eval_prompts = load_wildjailbreak_prompts(
    limit=config["data"].get("limit"),
)

#Loss
loss_cfg = config["loss"]
loss_fn = AttentionConsistencyLoss(
    weight=loss_cfg.get("weight", 1.0),
    output_hidden_states=False,
)

wandb.init(project="AttCT", name="prefill_act_wildjailbreak", config=config)

#Train
trainer = Trainer(
    model=model,
    dataloader=train_dl,
    loss_fn=loss_fn,
    config=config,
    device=device,
    ref_model=None,
    tokenizer=tokenizer,
    log_io_path="logs/prefill_act/io_log.jsonl",
)
trainer.train()

trainer.eval_loss(eval_dl)
wandb.finish()