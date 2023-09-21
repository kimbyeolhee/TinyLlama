import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig

def create_bnb_config():
    """
    load LLM in 4bit quantization. it can divide the used memory by 4.
    it makes the model to import on smaller GPU.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

def create_peft_config(modules):
    """
    Create Parameter Efficient Fine-Tuning config
    :param modules: Names of the modules to apply LoRA
    """
    return LoraConfig(
        r=16, # dimension of the low-rank approximation
        lora_alpha=64, # parameter for scaling the loss
        target_modules=modules, # modules to apply LoRA
        lora_dropout=0.1, # dropout rate for LoRA
        bias="none",
        task_type="CAUSAL_LM"
    )