"""The Transformer building blocks: positional encoding, normalization, attention, feedforward."""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# Defining helper to clone a module to be used in Transformer
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# Defining Positional Encoding for token embeddings
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Creating constant 'pe' matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Adding position info to input embeddings
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# Creating Layer Normalization
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    # Applying normalization
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# Creating Sublayer connection block with residual + normalization
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    # Applying residual connection and normalization
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


# Creating Scaled Dot-Product Attention
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)# getting dimension of key/query
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)# calculating scaled dot-product
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)# applying mask to ignore padding
    p_attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)# applying dropout to attention weights
    return torch.matmul(p_attn, value), p_attn


# Creating Multi-Head Attention module
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0# making sure model dimension is divisible by number of heads
        self.d_k = d_model // h# calculating dimension per head
        self.h = h# storing number of heads
        # creating linear layers for query, key, value, and output projection
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)# setting dropout

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)# adjusting mask shape for multi-head
        nbatches = query.size(0)# getting batch size
        # projecting query, key, and value for each head and reshaping
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]
        # computing attention and getting weighted values
        x, self.attn = attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)# reshaping back and combining heads
        return self.linears[-1](x)


# Creating Feedforward layer used in Transformer
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)# projecting input to higher dimension
        self.w_2 = nn.Linear(d_ff, d_model)# projecting back to original dimension
        self.dropout = nn.Dropout(dropout)# applying dropout between layers

    # Applying two-layer feedforward with ReLU and dropout
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# Creating token embeddings with positional scaling
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)# looking up embedding table
        self.d_model = d_model

    def forward(self, x):
        # Scaling embeddings by sqrt(d_model) to stabilize gradients
        return self.lut(x) * math.sqrt(self.d_model)
