"""Training loop for the from-scratch Transformer summarizer.

Run with: python -m summarizer.train
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from . import config
from .data import load_arxiv, make_dataloader
from .decoding import generate_summary
from .metrics import ROUGE_TYPES, evaluate, format_metrics, make_scorer
from .model import make_model, subsequent_mask
from .tokenization import build_tokenizer

HISTORY_KEYS = [*ROUGE_TYPES, "bleu", "bertscore_f1"]


def get_device():
    """Setting the device, GPU if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, train_loader, loss_fn, optimizer, pad_token_id, device):
    model.train()# setting model to training mode
    total_loss = 0
    # Iterating through batches
    for src, tgt in tqdm(train_loader):
        src = src.to(device)
        tgt = tgt.to(device)
        # Preparing decoder input and output
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        # Creating masks for source and target
        src_mask = (src != pad_token_id).unsqueeze(-2)
        tgt_mask = (tgt_input != pad_token_id).unsqueeze(-2) & subsequent_mask(tgt_input.size(-1)).to(device)
        # Running forward pass
        out = model(src, tgt_input, src_mask, tgt_mask)
        logits = model.generator(out)
        # Reshaping tensors for loss calculation
        logits = logits.view(-1, logits.size(-1))
        tgt_output = tgt_output.contiguous().view(-1)
        # Calculating and updating loss
        loss = loss_fn(logits, tgt_output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def train(epochs=config.EPOCHS):
    dataset = load_arxiv()
    tokenizer = build_tokenizer(dataset)
    train_loader = make_dataloader(dataset, tokenizer)

    # Loading original raw dataset for evaluation
    raw_dataset = load_arxiv()

    device = get_device()

    # Initializing model, loss, and optimizer
    vocab_size = tokenizer.get_vocab_size()
    pad_token_id = tokenizer.token_to_id("<pad>")
    model = make_model(vocab_size).to(device)# creating and sending model to device
    loss_fn = nn.NLLLoss(ignore_index=pad_token_id)# ignoring pad token in loss
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)# setting learning rate

    # Initializing lists to track metrics
    loss_history = []
    history = {k: [] for k in HISTORY_KEYS}

    scorer, smoothie = make_scorer()

    # Starting training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        avg_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, pad_token_id, device)
        loss_history.append(avg_loss)
        print(f"Avg Loss: {avg_loss:.4f}")

        # Evaluate on 1 sample
        model.eval()
        sample = raw_dataset[0]["article"]
        reference = raw_dataset[0]["abstract"]
        prediction = generate_summary(model, sample, tokenizer)

        metrics = evaluate(prediction, reference, scorer, smoothie)
        for key in HISTORY_KEYS:
            history[key].append(metrics[key])

        # Printing all evaluation metrics
        print(format_metrics(metrics))

    return model, tokenizer, raw_dataset, loss_history, history


def show_samples(model, tokenizer, raw_dataset, n=5):
    """Running and printing predictions for a few samples."""
    for i in range(n):
        print(f"\nSample {i + 1}")
        article = raw_dataset[i]["article"]# getting source text
        target = raw_dataset[i]["abstract"]# getting reference summary
        predicted = generate_summary(model, article, tokenizer)# generating prediction

        print(" Reference:", target[:200], "...")# printing first 200 characters of reference
        print(" Generated:", predicted)# printing generated summary
        print("-" * 80)


if __name__ == "__main__":
    model, tokenizer, raw_dataset, loss_history, history = train()
    show_samples(model, tokenizer, raw_dataset)
