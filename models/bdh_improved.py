import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bdh_base import Attention

class BDHImproved(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.n_layer = config.n_layer
        self.mult = config.mlp_internal_dim_multiplier
        self.N = (self.n_embd * self.mult) // self.n_head
        
        self.embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.ln = nn.LayerNorm(config.n_embd)
        
        self.encoder = nn.Parameter(torch.Tensor(self.n_head, self.n_embd, self.N))
        self.encoder_v = nn.Parameter(torch.Tensor(self.n_head, self.n_embd, self.N))
        self.decoder = nn.Parameter(torch.Tensor(self.n_head * self.N, self.n_embd))
        
        nn.init.normal_(self.encoder, std=0.02)
        nn.init.normal_(self.encoder_v, std=0.02)
        nn.init.normal_(self.decoder, std=0.02)
        
        self.attn = Attention(self.n_head, self.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        x = self.embed(idx).unsqueeze(1)
        x = self.ln(x)
        
        for _ in range(self.n_layer):
            x_latent = x @ self.encoder
            # GELU instead of ReLU
            x_sparse = F.gelu(x_latent)
            
            y = self.attn(x_sparse, x_sparse, x)
            y = self.ln(y)
            
            y_latent = y @ self.encoder_v
            # GELU instead of ReLU
            y_sparse = F.gelu(y_latent)
            
            xy = x_sparse * y_sparse
            xy = self.drop(xy)
            
            y_mlp = (
                xy.transpose(1, 2)
                .reshape(B, 1, T, self.n_head * self.N)
                @ self.decoder
            )
            
            y_out = self.ln(y_mlp)
            x = self.ln(x + y_out)
            
        x = x.view(B, T, self.n_embd)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
            
        return logits, loss
