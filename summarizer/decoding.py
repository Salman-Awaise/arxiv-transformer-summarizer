"""Turning an article into a generated summary, one token at a time."""

import torch

from . import config
from .model import subsequent_mask


# Defining decoding function for summary generation
def greedy_decode(model, src, src_mask, max_len, start_symbol, eos_symbol):
    model.eval()# setting model to eval mode
    memory = model.encode(src, src_mask)

    # Initializing decoder with <s> token
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src).to(src.device)

    for _ in range(max_len - 1):
        tgt_mask = subsequent_mask(ys.size(1)).to(src.device).unsqueeze(1)# creating mask for decoding
        out = model.decode(memory, src_mask, ys, tgt_mask)# decoding step

        # Sampling next token using probabilities instead of argmax
        probs = torch.softmax(model.generator(out[:, -1]), dim=-1)
        next_word = torch.multinomial(probs, num_samples=1)
        # appending predicted token
        ys = torch.cat([ys, next_word], dim=1)

        # stopping if </s> is predicted
        if next_word.item() == eos_symbol:
            break

    return ys


# Function to generate summary from input text using greedy decoding
def generate_summary(model, input_text, tokenizer, max_len=config.MAX_GEN_LEN):
    model.eval()# setting model to evaluation mode

    # Tokenizing the input article
    input_ids = tokenizer.encode(input_text).ids[:config.SRC_LEN]# trimming to max 128 tokens
    input_ids += [tokenizer.token_to_id("<pad>")] * (config.SRC_LEN - len(input_ids))# padding to fixed length
    src = torch.tensor([input_ids]).to(next(model.parameters()).device)# creating batch and moving to device
    src_mask = (src != tokenizer.token_to_id("<pad>")).unsqueeze(-2)# creating source mask

    # Setting start and end token IDs
    start_symbol = tokenizer.token_to_id("<s>")
    eos_symbol = tokenizer.token_to_id("</s>")

    # Decoding output tokens using greedy strategy
    decoded_ids = greedy_decode(model, src, src_mask, max_len, start_symbol, eos_symbol)

    # Removing special tokens and decoding to text
    token_ids = [t for t in decoded_ids[0].tolist() if t not in [
        tokenizer.token_to_id("<pad>"),
        tokenizer.token_to_id("<s>"),
        tokenizer.token_to_id("</s>")]]

    return tokenizer.decode(token_ids)
