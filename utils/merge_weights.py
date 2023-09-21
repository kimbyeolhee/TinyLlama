import os
import torch
from dotenv import load_dotenv
from peft import AutoPeftModelForCausalLM

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

def merge_weights(model, tokenizer, config, save_path):
    """
    merge the weights from LoRA with the original model.
    reload the base model in FP16 precision and the peft library to merge everyhing.
    """
    model = AutoPeftModelForCausalLM.from_pretrained(
        save_path,
        device_map="auto",
        return_dict=True,
        torch_dtype=torch.bfloat16
    )
    model = model.merge_and_unload()
    merged_save_path = os.path.join(config.setup.output_dir, config.model.name_or_path.split("/")[-1], config.setup.experiment_name, "merged")
    os.makedirs(merged_save_path, exist_ok=True)
    model.save_pretrained(merged_save_path, safe_serialization=True)

    # tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path, use_auth_token=HF_TOKEN)
    tokenizer.save_pretrained(merged_save_path)