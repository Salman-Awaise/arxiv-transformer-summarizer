"""The encoder/decoder stacks and the assembled Transformer model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config
from .layers import (Embeddings, LayerNorm, MultiHeadedAttention,
                     PositionalEncoding, PositionwiseFeedForward,
                     SublayerConnection, clones)


# Creating a single Encoder layer self-attention + feedforward
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn# setting self-attention layer
        self.feed_forward = feed_forward# setting feedforward layer
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    # Applying self-attention followed by feedforward, with residual connections
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


# Creating a single Decoder layer: masked self-attn, encoder-decoder attn, feedforward
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn# setting masked self-attention
        self.src_attn = src_attn# setting encoder-decoder attention
        self.feed_forward = feed_forward# setting feedforward layer
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory# storing encoder output as memory
        # Applying masked self-attention, encoder-decoder attention, then feedforward
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


# Creating Encoder stack by stacking multiple Encoder layers
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)# cloning the EncoderLayer N times
        self.norm = LayerNorm(layer.size)# applying final normalization

    def forward(self, x, mask):
        # Passing input through each Encoder layer
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)# returning final normalized output


# Creating Decoder stack by stacking multiple Decoder layers
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        # Passing input through each Decoder layer
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
            # returning final normalized output
        return self.norm(x)


# Creating the full Transformer model: Encoder + Decoder + Generator
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder# setting encoder stack
        self.decoder = decoder# setting decoder stack
        self.src_embed = src_embed# embedding + positional encoding for source
        self.tgt_embed = tgt_embed# embedding + positional encoding for target
        self.generator = generator# projection to vocabulary + softmax

    def forward(self, src, tgt, src_mask, tgt_mask):
        # Encoding source, then decoding target based on encoded memory
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        # Embedding and passing source into encoder
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        # Embedding and decoding target using encoder memory
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


# Creating the Generator to convert decoder output to vocabulary prediction
class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # Applying log softmax to get token probabilities
        return F.log_softmax(self.proj(x), dim=-1)


# Creating a mask to block attention to future positions in the decoder
def subsequent_mask(size):
    # Masking out upper triangular matrix
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


# Building the full Transformer model
def make_model(vocab_size, N=config.N_LAYERS, d_model=config.D_MODEL,
               d_ff=config.D_FF, h=config.N_HEADS, dropout=config.DROPOUT):
    # Creating attention, feedforward, and positional encoding layers
    attn = MultiHeadedAttention(h, d_model)# setting multi-head attention
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)# creating feedforward network
    position = PositionalEncoding(d_model, dropout)# applying positional encoding
    # Constructing the full Encoder-Decoder architecture
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, attn, ff, dropout), N),# stacking N encoder layers
        Decoder(DecoderLayer(d_model, attn, attn, ff, dropout), N),# stacking N decoder layers
        nn.Sequential(Embeddings(d_model, vocab_size), position),# embedding + positional encoding for source
        nn.Sequential(Embeddings(d_model, vocab_size), position),# embedding + positional encoding for target
        Generator(d_model, vocab_size)# output generator projecting to vocab size
    )
    return model
