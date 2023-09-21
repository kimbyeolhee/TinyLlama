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
import lightning as L
from lightning.fabric.strategies import FSDPStrategy, XLAStrategy
from torch.utils.data import DataLoader
from functools import partial

###
from TinyLlama.model import GPT, Config

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
    GPT(model_config)
    # if devices > 1:
    #     strategy = FSDPStrategy(
    #         auto_wrap_policy={Block},
    #     )