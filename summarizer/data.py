"""Loading the arXiv dataset, tokenizing it, and serving it as PyTorch batches."""

import re

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from . import config


def load_arxiv(split=config.DATASET_SPLIT):
    """Load the arXiv slice of the scientific_papers dataset."""
    return load_dataset(config.DATASET_NAME, config.DATASET_CONFIG, split=split)


# Defining preprocessing to tokenize and pad inputs/targets
def make_preprocess(tokenizer):
    def preprocess(example):
        input_ids = tokenizer.encode(example["article"]).ids[:config.SRC_LEN]
        target_ids = tokenizer.encode(example["abstract"]).ids[:config.TGT_LEN]

        input_ids += [tokenizer.token_to_id("<pad>")] * (config.SRC_LEN - len(input_ids))
        target_ids = [tokenizer.token_to_id("<s>")] + target_ids + [tokenizer.token_to_id("</s>")]
        target_ids += [tokenizer.token_to_id("<pad>")] * (config.TGT_LEN_PADDED - len(target_ids))

        return {"input_ids": input_ids, "labels": target_ids}
    return preprocess


# Wrapping tokenized HuggingFace dataset into PyTorch Dataset
class ArxivDataset(Dataset):
    def __init__(self, data):
        self.data = data# storing preprocessed dataset

    def __len__(self):
        return len(self.data)# returning dataset length

    def __getitem__(self, idx):
        # Getting tokenized input-output pair
        item = self.data[idx]
        src = torch.tensor(item["input_ids"])# getting source tokens
        tgt = torch.tensor(item["labels"])# getting target summary tokens
        return src, tgt


def make_dataloader(dataset, tokenizer, batch_size=config.BATCH_SIZE):
    """Tokenize the dataset and wrap it in a shuffling DataLoader."""
    tokenized = dataset.map(make_preprocess(tokenizer))
    return DataLoader(ArxivDataset(tokenized), batch_size=batch_size, shuffle=True)


# Creating a cleaning function to preprocess raw data
def clean_and_filter(example):
    # Replacing @xmath variables with [MATH] placeholder
    article = re.sub(r"@xmath\d+", "[MATH]", example["article"])
    abstract = re.sub(r"@xmath\d+", "[MATH]", example["abstract"])

    # Filtering articles with acceptable character length
    if 50 <= len(article) <= 1000:
        return {
            "article": article,
            "abstract": abstract,
            "section_names": example.get("section_names", "")
        }
    else:
        return None  # dropping article if it's too short/long
