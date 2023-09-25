import os
import sys
import argparse

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

from pathlib import Path
from omegaconf import OmegaConf
from peft import get_peft_model, prepare_model_for_kbit_training
from utils.model_loader import load_model_and_tokenizer
from dataset import get_dataset
from utils.LoRA import find_all_linear_names
from utils.setup import seed_everything
from utils.create_config import create_peft_config
import torch
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from utils.merge_weights import merge_weights

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))



def train(model, tokenizer, dataset, config):
    # Enabling gradient checkpointing to reduce memory usage for training
    model.gradient_checkpointing_enable()

    # using PEFT's prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)

    # get modules to apply LoRA
    modules = find_all_linear_names(model)

    # create PEFT config
    peft_config = create_peft_config(modules)
    model = get_peft_model(model, peft_config)

    # training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=config.train.per_device_train_batch_size,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        warmup_steps=config.train.warmup_steps,
        # max_steps=500,
        output_dir=config.setup.output_dir,
        num_train_epochs=config.train.num_train_epochs,
        learning_rate = config.train.learning_rate,
        fp16=config.train.fp16,
        optim = config.train.optim,
    )

    # train
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    model.config.use_cache = False

    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    print("-------- Training Done! ----------")
    print(metrics)

    # save model
    save_path = os.path.join(config.setup.output_dir, config.model.name_or_path.split("/")[-1], config.setup.experiment_name, "8bit")
    os.makedirs(save_path, exist_ok=True)
    trainer.model.save_pretrained(save_path, safe_serialization=True) # json/encoder.py's def default needs to support arbirary iterators

    # Reload model in FP16 and merge it with LoRA weights
    merge_weights(model, tokenizer, config, save_path)

    # delete memory cache
    del model
    del trainer
    torch.cuda.empty_cache()


def main(config, device):
    model, tokenizer = load_model_and_tokenizer(config.model.name_or_path)
    dataset = get_dataset(model, tokenizer, config)

    train(model, tokenizer, dataset, config)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="finetune a TinyLlama")
    parser.add_argument("--config", "-c", type=str, default="finetune_base_config")
    args = parser.parse_args()

    config = OmegaConf.load(
        f"{wd}/configs/{args.config}.yaml"
    )
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    print("Current cuda device: ", torch.cuda.current_device())
    print("Count of using GPUs: ", torch.cuda.device_count())

    seed_everything(config.setup.seed)

    main(config, device)