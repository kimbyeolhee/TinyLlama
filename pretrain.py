import os
import glob
import sys
import time
import argparse
from pathlib import Path
from typing import Optional, Tuple, Union
import math
import torch
from omegaconf import OmegaConf


###
from Llama.model import GPT
from Llama.model_config import Config

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

def main(fabric, train_data_dir, val_data_dir, resume):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain a TinyLlama")
    parser.add_argument("--config", "-c", type=str, default="pretrain_base_config")
    args = parser.parse_args()

    config = OmegaConf.load(
        f"{wd}/configs/{args.config}.yaml"
    )
    
    devices = config.device_num
    model_config = Config.from_name(config.model_name)

    model = GPT(model_config)
    print("🎆")
    print(model)