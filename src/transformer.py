import torch
import esm
from esm import (
    Alphabet,
    FastaBatchedDataset,
    ProteinBertModel,
    pretrained,
    MSATransformer,
)
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import einsum
from torch.nn.functional import softmax

class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, r_ff, p_drop=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model*r_ff)
        self.dropout = nn.Dropout(p_drop)
        self.linear2 = nn.Linear(d_model*r_ff, d_model)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear1.bias)

        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
    
    def forward(self, x):
        residual = x  
        x = self.norm(x)
        x = self.linear2(self.dropout(F.relu_(self.linear1(x))))
        return residual + x



class Attention(nn.Module):
    # calculate multi-head attention
    def __init__(self, d_query, d_key, n_head, d_hidden, d_out, p_drop=0.1):
        super().__init__()
        self.h = n_head
        self.dim = d_hidden
        #
        self.to_q = nn.Linear(d_query, n_head*d_hidden, bias=False)
        self.to_k = nn.Linear(d_key, n_head*d_hidden, bias=False)
        self.to_v = nn.Linear(d_key, n_head*d_hidden, bias=False)
        #
        self.to_out = nn.Linear(n_head*d_hidden, d_out)
        self.scaling = 1/math.sqrt(d_hidden)
        #
        # initialize all parameters properly
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        nn.init.xavier_uniform_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, query, key, value):
        B, Q = query.shape[:2]
        B, K = key.shape[:2]
        #
        query_proj = self.to_q(query).reshape(B, Q, self.h, self.dim)
        key_proj = self.to_k(key).reshape(B, K, self.h, self.dim)
        value_proj = self.to_v(value).reshape(B, K, self.h, self.dim)
        #
        query_proj = query_proj * self.scaling
        attn = einsum('bqhd,bkhd->bhqk', query_proj, key_proj)
        attn = softmax(attn, dim=-1)
        #
        out = einsum('bhqk,bkhd->bqhd', attn, value_proj)
        out = out.reshape(B, Q, self.h*self.dim)
        #
        #out = self.to_out(out)
        #
        # Add residual connection
        residual = query
        out = out + residual
        return out


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        return self.layer_norm(x)


class TransBlock(nn.Module):
    def __init__(self, model_location,d_out, r_ff=4, p_drop=0.1, n_head=20):
        super().__init__()
        self.model_location = model_location 
        if self.model_location == "esm2_t48_15B_UR50D":
            self.embed_dim = 5120
            self.num_layers = 47
        elif self.model_location == "esm2_t36_3B_UR50D":
            self.embed_dim = 2560
            self.num_layers = 35
        elif self.model_location == "esm2_t33_650M_UR50D":
            self.embed_dim = 1280
            self.num_layers =32
        elif self.model_location == "esm2_t30_150M_UR50D":
            self.embed_dim = 640
            self.num_layers = 29
        elif self.model_location == "esm2_t12_35M_UR50D":
            self.embed_dim = 480
            self.num_layers = 11
        elif self.model_location == "esm2_t6_8M_UR50D":
            self.embed_dim = 320
            self.num_layers = 5
        else:
            raise ValueError("Provide an accurate esm_embedder name")

           
        self.n_head = n_head
        self.d_hidden = self.embed_dim // self.n_head
        self.d_out = d_out
        self.r_ff = r_ff
        self.feedforward = FeedForwardLayer(self.embed_dim, self.r_ff)
        self.attention = Attention(self.embed_dim, self.embed_dim, self.n_head, self.d_hidden, self.d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(x,x,x)
        x = self.feedforward(x)
        return x
