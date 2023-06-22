import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from utils.utils import print_trainable_parameters, add_eos_text, group_texts
from datasets import Dataset, load_dataset
import transformers
import functools

model_id = "databricks/dolly-v2-3b"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb_config, device_map={"": 0}
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)

data = load_dataset("text", data_files="/content/drive/MyDrive/chat_history.txt")
data = data.map(functools.partial(add_eos_text, tokenizer=tokenizer), batched=True)
data = data.map(
    lambda samples: tokenizer(samples["text"]), batched=True, remove_columns=["text"]
)
lm_datasets = data.map(
    group_texts,
    batched=True,
    batch_size=250,
    num_proc=4,
)

# needed for gpt-neo-x tokenizer
tokenizer.pad_token = tokenizer.eos_token

trainer = transformers.Trainer(
    model=model,
    train_dataset=lm_datasets["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        warmup_steps=100,
        max_steps=500,
        learning_rate=1e-4,
        fp16=True,
        logging_steps=1,
        output_dir="/content/drive/MyDrive/Lora_outputs",
        optim="paged_adamw_8bit",
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()
model.save_pretrained("/content/drive/MyDrive/Lora_outputs/final_model")
