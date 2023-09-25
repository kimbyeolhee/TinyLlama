import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from functools import partial
from datasets import load_dataset
from transformers import AutoTokenizer

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

wd = Path(__file__).parent.parent.resolve()

def create_prompt_formats(sample):
    """Concatenate dataset's column "question", "context", "answer" into new line
    :param sample: Sample of dataset(dictionary)
    """
    INTRO = "You are an expert in the field of Open Target and Drug Discovery. You are asked to answer the following question."
    QUESTION_KEY = "### Question:"
    CONTEXT_KEY = "### Context:"
    ANSWER_KEY = "### Answer:"
    END_KEY = "### End"

    intro = f"{INTRO}"
    question = f"{QUESTION_KEY}\n{sample['question']}"
    input_context = f"{CONTEXT_KEY}\n{sample['context']}" if sample["context"] else None
    answer = f"{ANSWER_KEY}\n{sample['answer']}"
    end = f"{END_KEY}"

    parts = [part for part in [intro, question, input_context, answer, end] if part] # context가 없는 경우 사용하지 않음

    formatted_prompt = "\n\n".join(parts)

    sample["formatted_prompt"] = formatted_prompt

    return sample

def preprocess_batch(batch, tokenizer, max_length):
    """Tokenize batch of samples"""
    return tokenizer(
        batch["formatted_prompt"],
        max_length=max_length,
        truncation=True,
    )

def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed: int, dataset):
    """Prepare for training
    :param tokenizer: Tokenizer for Model
    :param max_length: Max length of input sequence
    """

    # log로 Preprocessing dataset...

    # Add prompt ver in "formatted_prompt" column
    dataset = dataset.map(create_prompt_formats)

    # Apply tokenizing to each batch and remove unnecessary columns
    _preprocessing_function = partial(preprocess_batch, tokenizer=tokenizer, max_length=max_length)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["question", "context", "answer"],
    )

    # Filter out samples that are longer than max_length
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)

    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)

    return dataset

def get_max_length(model):
    """Return max_length from model's config"""
    max_length = None
    conf = model.config

    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(conf, length_setting, None)
        if max_length is not None:
            print(f"Max length from {length_setting}: {max_length}")
            break
    if max_length is None:    
        max_length = 1024
        print(f"Max length set to {max_length}")
    return max_length
    

    
def get_dataset(model, tokenizer, config):
    """Return dataset for training"""
    dataset = load_dataset(config.data.name_or_path, split="train", use_auth_token=HF_TOKEN)
    max_length = get_max_length(model)
    dataset = preprocess_dataset(tokenizer, max_length, config.setup.seed, dataset)

    return dataset

    

if __name__ == "__main__":
    load_dataset()
    