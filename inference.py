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
    # max_length 는 Prompt(LM의 초기 입력값)을 포함한 최대 길이를 지정하지만 max_new_tokens 는 Prompt를 제외한 ‘생성한' 문장의 최대 길이를 지정
    output = model.generate(
        input_ids=tokenized_input["input_ids"], 
        attention_mask=tokenized_input["attention_mask"], 
        max_new_tokens=config.inference.max_new_tokens, 
        temperature=config.inference.temperature,
        early_stopping=True)

    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="finetune a TinyLlama")
    parser.add_argument("--config", "-c", type=str, default="inference_config")
    args = parser.parse_args()

    config = OmegaConf.load(
        f"{wd}/configs/{args.config}.yaml"
    )

    INTRO = "You are an expert in the field of Open Target and Drug Discovery. You are asked to answer the following question."
    QUESTION_KEY = "### Question:"
    CONTEXT_KEY = "### Context:"
    ANSWER_KEY = "### Answer:"

    intro = f"{INTRO}"
    question = f"{QUESTION_KEY}\n{config.input.question}"
    input_context = f"{CONTEXT_KEY}\n{config.input.context}" if config.input.context != "None" else None
    answer = f"{ANSWER_KEY}\n"

    parts = [intro, question, input_context, answer]
    input_text = "\n\n".join([part for part in parts if part])
    
    output = inference(input_text, config)

    print(output)