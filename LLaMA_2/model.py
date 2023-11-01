import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, List

from transformers import AutoTokenizer

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32 # Number of heads for the queries
    n_kv_heads: Optional[int] = None # Number of heads for the keys and values
    vocab_size: int = -1 # will be set when we load tokenizer
    multiple_of: int = 256 # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # Needed for KV cache
    max_batch_size: int =32
    max_seq_len: int = 2048

    device: str = "cpu"

def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions
    """
    # 논문에 의하면 dimension of embedding은 짝수여야 함
    assert head_dim % 2 == 0, "head_dim must be even"

    # theta parameter 생성
    ## 논문에 따르면 theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ..., dim/2]
    theta_numerator = torch.arange(0, head_dim, 2).float() # shape: (head_dim/2) -> tensor([0., 2., 4., ..., head_dim-2]) -> torch.Size([64])
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device) # shape: (head_dim/2) -> tensor([1.0000e+00, 8.6596e-01, 7.4989e-01, ..., 1.1548e-04])

    # Position parameter 생성 ("m")
    m = torch.arange(seq_len, device=device) # shape: (seq_len) -> tensor([0, 1, 2, ..., seq_len-1]) # torch.Size([4096])

    # 각각의 theta에 대해 각각의 position(m)을 곱해줌 
    # torch.outer(a, b) => size of a: n and size of b: m => size of output: (n, m)
    freqs = torch.outer(m, theta).float() # shape: (seq_len, head_dim / 2) -> torch.Size([4096, 64])

    # 복소수(complex number)로 변환: c = R * exp(i * m * theta), 이때 R은 논문에 따라 1로 설정
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs) # shape: (seq_len, head_dim / 2) -> torch.Size([4096, 64])

    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # 실수와 허수로 분리
    # shape: (Batch, Seq_len, H, Head_Dim) -> (Batch, Seq_len, H, Head_Dim) # torch.Size([4, 1, 32, 64])
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    # 인자로 전달받은 freq_complex tensor를 x_complex tensor와 곱해주기 위해서는 freq_complex tensor shape가 Batch dimension과 Head Dimension 자리를 갖고 있어야함
    # shape: (Seq_len, Head_dim/2) -> (1, Seq_len, 1, Head_dim/2) # torch.Size([1, 64]) => torch.Size([1, 1, 1, 64])
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(0)

    # x_complex와 freqs_complex를 곱해줌
    # shape: (B, Seq_len, H, Head_dim/2) * (1, Seq_len, 1, Head_dim/2) -> (B, Seq_len, H, Head_dim/2)
    x_rotated = x_complex * freqs_complex # torch.Size([4, 1, 32, 64])

    # 복소수 형태를 다시 실수 형태로 변환
    # shape: (B, Seq_len, H, Head_dim/2) -> (B, Seq_len, H, Head_dim/2, 2)
    x_out = torch.view_as_real(x_rotated) # torch.Size([4, 1, 32, 64, 2])

    # shape: (B, Seq_len, H, Head_dim/2, 2) -> (B, Seq_len, H, Head_dim)
    x_out = x_out.reshape(*x.shape) # torch.Size([4, 1, 32, 128])

    return x_out.type_as(x).to(device)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor):
        return self.weight * self._norm(x.float()).type_as(x)


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads # number of times each key and value is repeated
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads*self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads*self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads*self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads*self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape # shape: (batch_size, seq_len=1, dim)
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x) # shape: (B, seq_len=1, h_query * head_dim), (B, seq_len=1, h_KV * head_dim), (B, seq_len=1, h_KV * head_dim)

        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim) # shape: (B, seq_len=1, h_query, head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim) # shape: (B, seq_len=1, h_KV, head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim) # shape: (B, seq_len=1, h_KV, head_dim)

        # Query와 Key에 Rotary Positional Embedding을 적용
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device) # shape: (B, seq_len=1, h_query, head_dim) # torch.Size([4, 1, 32, 128])
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device) # shape: (B, seq_len=1, h_KV, head_dim) # torch.Size([4, 1, 32, 128])



class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = self.dim // self.n_heads

        self.attention = SelfAttention(args) # 📌
        # self.feed_forward = FeedForward(args)

        # Nomralization beforem the attention block
        self.attention_norm = RMSNorm(self.dim, eps=args.norm_eps)
        # # Mormalization before the feed forward block
        # self.ffn_norm = RMSNorm(self.dim, eps=args.norm_eps)
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_complex
        )
        # out = h + self.feed_forward.forward(self.ffn_norm(h))

        # return out


class Transformer(nn.Module):
    def __init__(self, args:ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set! load tokenizer before creating model"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(TransformerBlock(args))

        # self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, 
                                                              self.args.max_seq_len * 2, # LLaMA1은 2048, LLaMA2는 4096로 학습함
                                                              device=self.args.device
                                                              )

    def forward(self, tokens: torch.Tensor, start_pos: int):
        batch_size, seq_len = tokens.shape # shape: (batch_size, seq_len)

        assert seq_len ==1, "Only one token at a time supported"
        h = self.tok_embeddings(tokens) # shape: (batch_size, seq_len, dim) # torch.Size([4, 1, 4096])

        # Retrieve the pairs (m, theta) corresponding to the postion
        freqs_complex = self.freqs_complex[start_pos : start_pos+seq_len] # shape: (seq_len, dim/2) # torch.Size([1, 64])
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)

def text_completion(model: nn.Module,
                    tokenizer: AutoTokenizer, 
                    prompts: List[str], 
                    temperature: float=0.6, 
                    top_p: float=0.9, 
                    max_gen_len: Optional[int]=None,
                    args: Optional[ModelArgs]=None):
    if max_gen_len is None:
        max_gen_len = args.max_seq_len
    # Convert each prompt into tokens
    prompt_tokens = [tokenizer.encode(prompt) for prompt in prompts]
    batch_size = len(prompt_tokens)

    assert batch_size <= args.max_batch_size, f"batch size must be less than or equal to {args.max_batch_size}"
    
    max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
    assert max_prompt_len <= args.max_seq_len, f"prompt length must be less than or equal to {args.max_seq_len}"
    total_len = min(args.max_seq_len, max_gen_len + max_prompt_len)

    pad_id = tokenizer.pad_token_id
    tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=args.device)
    for k, t in enumerate(prompt_tokens):
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=args.device)
    
    eos_reached = torch.tensor([False] * batch_size, device=args.device)
    prompt_tokens_mask = tokens != pad_id # True if the token is a prompt token, False otherwise
    cur_iterator = tqdm(range(1, total_len), desc="Generating text")
    for cur_pos in cur_iterator:
        with torch.no_grad():
            print("🚗 cur_pos --> forward: ", cur_pos)
            logits = model.forward(tokens[:, cur_pos-1:cur_pos], cur_pos)  # ⭐HERE⭐
        if temperature > 0:
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p) # 📌
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1)
        
        next_token = next_token.reshape(-1)
        # replace token if it is a padding token
        next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == tokenizer.eos_token_id)
        if all(eos_reached):
            break

    out_tokens, out_text = [], []
    for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
        if tokenizer.eos_id in current_prompt_tokens:
            eos_idx = current_prompt_tokens.index(tokenizer.eos_id)
            current_prompt_tokens = current_prompt_tokens[:eos_idx]
        out_tokens.append(current_prompt_tokens)
        out_text.append(tokenizer.decode(current_prompt_tokens)) 
    return out_tokens, out_text


def sample_top_p(self, probs, p):
    # (B, vocab_size)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)

    mask = probs_sum - probs_sort > p

    probs_sort[mask] = 0.0

    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, dim=-1, index=next_token)
    return next_token


if __name__ == "__main__":
    args = ModelArgs()

    # using llama2 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("PY007/TinyLlama-1.1B-Chat-v0.2", bos_token = "</s>")
    args.vocab_size = tokenizer.vocab_size

    # check the tensor shape of Rotary Positional Embedding
    model = Transformer(args)

    prompts = [
        "Hello, how are you?",
            "I'm fine, thank you. And you?",
            "I'm fine too.",
            "That's good to hear."
    ]

    out_tokens, out_text = text_completion(model, tokenizer, prompts, args=args)
    for i in range(len(out_text)):
        print(f"🤖: {out_text[i]}")
