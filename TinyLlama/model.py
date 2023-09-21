import os
import math
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import Self
# from flash_attn import flash_attn_func
from TinyLlama.config import Config
# from xformers.ops import SwiGLU
# from TinyLlama.fused_rotary_embedding import apply_rotary_emb_func
RoPECache = Tuple[torch.Tensor, torch.Tensor]
KVCache = Tuple[torch.Tensor, torch.Tensor]
FlashAttention2Available = RequirementCache("flash-attn>=2.0.0.post1")

class GPT(nn.Module):
    def __init__(self,config:Config) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False) # lm_head는 GPT의 마지막 레이어로, GPT의 출력을 다시 vocab_size만큼의 차원으로 변환해줌
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd), # wte는 vocab_size만큼의 차원을 n_embd만큼의 차원으로 변환해주는 레이어
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)), 
                ln_f=config.norm_class(config.embd, eps=config.norm_eps), # ln_f는 LayerNorm
            )
        )
        self.rope_cache: Optional[RoPECache] = None # Rotary Positional Embedding을 저장하는 변수
        self.mask_cache: Optional[torch.Tensor] = None # 마스크를 저장하는 변수
        self.kv_cache: List[KVCache] = [] # key와 value를 저장하는 변수

    def _init_weights(self, module: nn.Module, n_layer) -> None:
        """
        get.apply(gpt._init_weights)처럼 사용
        """
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / module.weight.size(1))) # module.weight.size(1)은 n_embd
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5/ module.weight.size(1))) # module.weight.size(1)은 n_embd
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        
        for name, p in module.parameters():
            if (name == "proj.weight" and isinstance(module, LLaMAMLP)) or (name == "w3.weight" and isinstance(module, SwiGLU)): # Swiglu 사용 시 fc2 layer의 이름이 w3.weight임
                nn.init.normal_(p, mean=0.0, std=1 / math.sqrt(p.shape[-1]) / n_layer)


class Block(nn.Module):
    def __init__(self, config:Config) -> None:
        super().__init__()
        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        if not config.shared_attention_norm:
            self.norm_2 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.mlp = config.mlp_class(config)
        self.config = config


class CausalSelfAttention(nn.Module):
    def __init__(self, config:Config) -> None:
        super().__init__()
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        self.attn = nn.Linear(config.n_embd, shape, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.config = config

    def forward(
            self,
            x: torch.Tensor,
            rope: RoPECache,
            max_seq_length: int,
            mask: Optional[torch.Tensor] = None,
            input_pos: Optional[torch.Tensor] = None,
            kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        B, T, C = x.size() # B: batch size, T: sequence length, C: embedding size(n_embd)

        qkv = self.attn(x) 