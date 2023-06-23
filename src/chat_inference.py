from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch


# load model with quantized weights and LORA weights
peft_model_id = "/content/drive/MyDrive/Lora_outputs/checkpoint-2000"
config = PeftConfig.from_pretrained(peft_model_id)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path, quantization_config=bnb_config, device_map={"": 0}
)

model = PeftModel.from_pretrained(model, peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Map to gpu
device = "cuda:0"
model = model.to(device)
model.eval()

# sample prediction
text = "Deepak: Hello\nPriya:"
inputs = tokenizer(text, return_tensors="pt").to(device)

# Prediction using sampling method, can also do beam search heres
outputs = model.generate(
    **inputs,
    max_new_tokens=40,
    early_stopping=True,
    do_sample=True,
    top_k=3,
    no_repeat_ngram_size=2
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
