"""Hyperparameters, dataset settings and artifact paths for the summarizer."""

DATASET_NAME = "scientific_papers"
DATASET_CONFIG = "arxiv"
DATASET_SPLIT = "train[:1%]"

CORPUS_PATH = "corpus.txt"
TOKENIZER_DIR = "custom_tokenizer"
SPECIAL_TOKENS = ["<s>", "</s>", "<pad>", "<unk>"]
VOCAB_SIZE = 30522
MIN_FREQUENCY = 2

SRC_LEN = 128
TGT_LEN = 32
TGT_LEN_PADDED = 34   # 32 target tokens plus <s> and </s>
MAX_GEN_LEN = 64

N_LAYERS = 6
D_MODEL = 768
D_FF = 3072
N_HEADS = 8
DROPOUT = 0.2

BATCH_SIZE = 8
EPOCHS = 250
LEARNING_RATE = 1e-4
