import os
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.create_config import create_bnb_config


load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")



def load_model_and_tokenizer(model_name_or_path: str):
    n_gpus = torch.cuda.device_count()
    max_memory = f"{24576}MB"
    bnb_config = create_bnb_config()

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        device_map= "auto",
        max_memory = {i: max_memory for i in range(n_gpus)},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_auth_token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token # Llama model은 eos token을 pad token으로 사용함
    tokenizer.padding_side = "right"
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer("PY007/TinyLlama-1.1B-Chat-v0.1")