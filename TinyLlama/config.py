from dataclasses import dataclass
from typing import Any, Literal, Optional, Type

import torch
from typing_extensions import Self

import TinyLlama.model
from TinyLlama.utils import find_multiple

@dataclass
class Config:
    org: str = "Lightning"
    name: str = "lit-GPT"
    block_size: int = 4096
    vocab_size: int = 50254
    padding_multiple: int = 512
    padded_vocab_size: Optional[int] = None
    n_layer: int = 16
    n_head: int = 32
    n_embd: int = 4096
    rotary_percentage: float = 0.25
    parallel_residual: bool = True
    bias: bool = True
    n_query_groups: Optional[int] = None
    shared_attention_norm: bool = False
    _norm_class: Literal["LayerNorm", "RMSNorm"] = "LayerNorm"
    norm_eps: float = 1e-5
    _mlp_class: Literal["GptNeoxMLP", "LLaMAMLP"] = "GptNeoxMLP"
    intermediate_size: Optional[int] = None
    condense_ratio: int = 1

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0
        
        # padded_vocab_size가 None이면 padded_vocab_size를 vocab_size를 넘지 않는 padding_multiple의 최소 배수로 설정
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size, self.padding_multiple)
        # n_query_groups가 None이 아니면 n_query_groups가 n_head의 약수여야 함
        # None이면 n_query_groups를 n_head 값으로 설정 
        if self.n_query_groups is not None:
            assert self.n_head % self.n_query_groups == 0
        else:
            self.n_query_groups = self.n_head
        # intermediate_size가 None이면 intermediate_size를 n_embd의 4배로 설정
        # mlp_class가 LLaMAMLP이면 intermediate_size가 None이면 ValueError 발생
        if self.intermediate_size is None:
            if self._mlp_class == "LLaMAMLP":
                raise ValueError("LLaMAMLP requires intermediate_size to be set")
            self.intermediate_size = 4 * self.n_embd


    @property
    def head_size(self) -> int:
        """
        embedding size를 head 수로 나눈 값
        """
        return self.n_embd // self.n_head

    @classmethod
    def from_name(cls, name:str, **kwargs:Any) -> "Self":
        conf_dict = name_to_config[name].copy()
        conf_dict
        return cls(**conf_dict)

    @property
    def mlp_class(self) -> Type:
        return getattr(TinyLlama.model, self._mlp_class)

    @property
    def norm_class(self) -> Type:
        if self._norm_class == "RMSNorm":
            from TinyLlama.rmsnorm import RMSNorm
            return RMSNorm
        elif self._norm_class == "FusedRMSNorm":
            from TinyLlama.rmsnorm import FusedRMSNorm
            return FusedRMSNorm
        return getattr(torch.nn, self._norm_class)

configs = [
    # https://huggingface.co/stabilityai/stablelm-base-alpha-3b/blob/main/config.json
    dict(org="stabilityai", name="stablelm-base-alpha-3b", padding_multiple=512), 
]

llama_2 = [
    # https://huggingface.co/meta-llama/Llama-2-7b-hf/blob/main/config.json
    dict(
        org="meta-llama",
        name="Llama-2-7b{}-hf",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=32,
        n_head=32,
        n_embd=4096,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,
    ),    
    dict(
        org="meta-llama",
        name="CodeLlama-2-7b-hf",
        block_size=4096,
        vocab_size=32016,
        padded_vocab_size=32016,
        padding_multiple=64,
        n_layer=32,
        n_head=32,
        n_embd=4096,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,
    ),
]
for c in llama_2:
    for kind in ("", "-chat"):
        copy = c.copy()
        copy["name"] = c["name"].format(kind)
        configs.append(copy)

tiny_LLaMA = [
     
    # https://twitter.com/cwolferesearch/status/1691929174175264858
    dict(
        org="StatNLP-research",
        name="tiny_LLaMA_1b",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=22,
        n_head=32,
        n_embd=2048,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5, #Llama 2 use 1e-5. Llama 1 use 1e-6
        _mlp_class="LLaMAMLP",
        intermediate_size=5632,
        n_query_groups=4,
    ),
]
configs.extend(tiny_LLaMA)

name_to_config = {config["name"]: config for config in configs}