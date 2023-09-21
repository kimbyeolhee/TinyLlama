import os
import sys
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
from utils.model_loader import load_model_and_tokenizer
from dataset import get_dataset
from utils.LoRA import find_all_linear_names
from utils.setup import set_device, seed_everything

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

def train(model, tokenizer, dataset, output_dir):
    # Enabling gradient checkpointing to reduce memory usage for training
    model.gradient_checkpointing_enable()

    # using PEFT's prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)

    # get LoRA module names
    modules = find_all_linear_names(model)
    print("📌")
    print(modules)

def main(config, device):
    model, tokenizer = load_model_and_tokenizer(config.model.name_or_path)
    dataset = get_dataset(model, tokenizer, config.setup.seed)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="finetune a TinyLlama")
    parser.add_argument("--config", "-c", type=str, default="finetune_base_config")
    args = parser.parse_args()

    config = OmegaConf.load(
        f"{wd}/configs/{args.config}.yaml"
    )

    device = set_device(config.setup.device_number)
    seed_everything(config.setup.seed)

    main(config, device)