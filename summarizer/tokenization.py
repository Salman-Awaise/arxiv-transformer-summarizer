"""Training and loading the Byte-Level BPE tokenizer used for both articles and abstracts."""

import os

from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

from . import config


def write_corpus(dataset, path=config.CORPUS_PATH):
    """Write every article and abstract as one line each, for tokenizer training."""
    with open(path, "w", encoding="utf-8") as f:
        for sample in dataset:
            f.write(sample['article'].replace("\n", " ") + "\n")   # input
            f.write(sample['abstract'].replace("\n", " ") + "\n")  # target/summary
    return path


def train_tokenizer(corpus_path=config.CORPUS_PATH, out_dir=config.TOKENIZER_DIR):
    """Train a Byte-Level BPE tokenizer from scratch and save it to out_dir."""
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=corpus_path,
        vocab_size=config.VOCAB_SIZE,
        min_frequency=config.MIN_FREQUENCY,
        special_tokens=config.SPECIAL_TOKENS
    )
    os.makedirs(out_dir, exist_ok=True)
    tokenizer.save_model(out_dir)
    return out_dir


def load_tokenizer(out_dir=config.TOKENIZER_DIR, max_length=config.SRC_LEN):
    """Load the saved tokenizer with the <s>/</s> post-processor and truncation enabled."""
    tokenizer = ByteLevelBPETokenizer(
        f"{out_dir}/vocab.json",
        f"{out_dir}/merges.txt"
    )
    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>"))
    )
    tokenizer.enable_truncation(max_length=max_length)
    return tokenizer


def build_tokenizer(dataset):
    """Write the corpus, train the tokenizer, and return it ready for use."""
    write_corpus(dataset)
    train_tokenizer()
    return load_tokenizer()
