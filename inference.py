import os
import sys
import argparse

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

from pathlib import Path
from omegaconf import OmegaConf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


def inference(input_text, config):
    tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path)
    model = AutoModelForCausalLM.from_pretrained(config.model.name_or_path)

    tokenized_input = tokenizer(input_text, return_tensors="pt")
    output = model.generate(input_ids=tokenized_input["input_ids"], attention_mask=tokenized_input["attention_mask"], max_length=50, num_beams=5, early_stopping=True)

    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="finetune a TinyLlama")
    parser.add_argument("--config", "-c", type=str, default="inference_config")
    args = parser.parse_args()

    config = OmegaConf.load(
        f"{wd}/configs/{args.config}.yaml"
    )

    input_text = "Can you provide a one-word answer to describe the role of the liver in activating alveolar macrophages during acute pancreatitis-induced lung injury?"
    output = inference(input_text, config)

    print(output)