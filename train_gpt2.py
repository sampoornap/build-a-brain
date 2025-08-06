from dataclasses import dataclass
import torch
from torch.nn import functional as F
import torch.nn as nn



class Block(nn.Module): # a single transformer block

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPT2Config: # stores model hyperparameters
    block_size: int = 128
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # gpt 2 is a decoder only transformer

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)







