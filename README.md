# Transformer-Based Abstractive Summarizer

A from-scratch PyTorch implementation of the Transformer encoder–decoder architecture (Vaswani et al., 2017) trained for abstractive summarization on scientific papers from the arXiv dataset.

The Transformer is implemented manually — positional encoding, multi-head attention, layer normalization, encoder/decoder stacks, and the output generator are all written from the ground up rather than imported from `torch.nn.Transformer` or HuggingFace `transformers`. The goal of the project is pedagogical: to understand the architecture by building it.

---

## Overview

Given a scientific paper's body text (`article`), the model is trained to generate its abstract (`abstract`). The full pipeline includes:

1. Loading the `scientific_papers` (arXiv) dataset from HuggingFace.
2. Training a **Byte-Level BPE tokenizer from scratch** on the corpus.
3. Preprocessing articles to a fixed input length of 128 tokens and summaries to 32 tokens (plus start/end tokens).
4. Building the full Transformer architecture in PyTorch.
5. Training the model with the Adam optimizer and NLLLoss.
6. Evaluating each epoch with **ROUGE-1, ROUGE-2, ROUGE-L, BLEU, and BERTScore F1**.
7. Visualizing loss and metric trajectories across training.
8. Generating predicted summaries on held-out samples.

---

## Architecture

The implementation closely follows the original "Attention is All You Need" paper and the Harvard NLP "Annotated Transformer" reference.

| Component | Description |
|---|---|
| `PositionalEncoding` | Sinusoidal positional encodings added to embeddings. |
| `LayerNorm` | Custom layer normalization with learnable scale and shift. |
| `SublayerConnection` | Residual connection with pre-norm and dropout. |
| `attention` | Scaled dot-product attention. |
| `MultiHeadedAttention` | Multi-head attention via parallel projected heads. |
| `PositionwiseFeedForward` | Two-layer feedforward (ReLU + dropout). |
| `EncoderLayer` / `DecoderLayer` | Single layer of encoder / decoder. |
| `Encoder` / `Decoder` | Stacks of `N` layers with final layer norm. |
| `Embeddings` | Token embeddings scaled by `√d_model`. |
| `EncoderDecoder` | Wrapper that ties the encoder, decoder, embeddings, and generator together. |
| `Generator` | Final linear projection to vocab + log-softmax. |
| `subsequent_mask` | Causal mask blocking attention to future positions in the decoder. |

### Default hyperparameters

| Parameter | Value |
|---|---|
| Layers (`N`) | 6 (encoder) + 6 (decoder) |
| Model dimension (`d_model`) | 768 |
| Feedforward dimension (`d_ff`) | 3072 |
| Attention heads (`h`) | 8 |
| Dropout | 0.2 |
| Vocabulary size | 30,522 |
| Source length | 128 tokens |
| Target length | 34 tokens (32 + `<s>` + `</s>`) |
| Batch size | 8 |
| Optimizer | Adam (lr = 1e-4) |
| Loss | NLLLoss (ignoring `<pad>`) |
| Epochs | 250 |

---

## Dataset

- **Source**: [`scientific_papers`](https://huggingface.co/datasets/scientific_papers) (arXiv subset) from HuggingFace Datasets.
- **Slice used**: `train[:1%]` — roughly 1% of the training split, kept small for fast iteration on limited compute.
- **Fields used**: `article` (full paper text) and `abstract` (target summary).

The notebook also contains a `clean_and_filter` helper that replaces `@xmathN` LaTeX-style math placeholders with `[MATH]` tokens and filters by character length, intended as an optional preprocessing pass on the raw data.

---

## Tokenization

A Byte-Level BPE tokenizer is trained from scratch on the concatenated articles and abstracts using HuggingFace `tokenizers`. The tokenizer:

- Uses vocab size 30,522 with minimum frequency 2.
- Adds special tokens: `<s>`, `</s>`, `<pad>`, `<unk>`.
- Applies a `BertProcessing` post-processor to add start/end tokens.
- Truncates sequences at 128 tokens.

Trained tokenizer files are saved under `custom_tokenizer/` (`vocab.json` and `merges.txt`).

---

## Evaluation Metrics

After each epoch, the model is evaluated on a held-out sample using:

- **ROUGE-1, ROUGE-2, ROUGE-L** (`rouge_score`) — unigram, bigram, and longest-common-subsequence overlap with the reference.
- **BLEU** (`nltk.translate.bleu_score`) — with method-4 smoothing for short sequences.
- **BERTScore F1** (`bert_score`) — semantic similarity using contextual embeddings.

All metric trajectories are plotted alongside the training loss.

---

## Decoding

Summary generation uses the `greedy_decode` function, which:

1. Encodes the source with the encoder stack.
2. Initializes the target sequence with `<s>`.
3. Repeatedly samples the next token from the model's softmax distribution (via `torch.multinomial`) until `</s>` is reached or `max_len` is hit.

> Note: The function is named `greedy_decode`, but uses **stochastic sampling** rather than argmax. To get pure greedy decoding, replace the sampling step with `torch.argmax`.

---

## Requirements

```bash
pip install torch
pip install datasets
pip install tokenizers
pip install rouge_score
pip install bert_score
pip install nltk
pip install matplotlib
pip install tqdm
pip install kagglehub
```

Python 3.8+ is recommended. A CUDA-capable GPU is strongly recommended given the model size and 250-epoch training schedule.

NLTK additionally requires:
```python
import nltk
nltk.download('punkt_tab')
```

---

## How to Run

1. Clone this repository.
2. Open the notebook in Jupyter, Colab, or Kaggle.
3. Run the cells in order:
   - Install dependencies.
   - Load the dataset.
   - Train the BPE tokenizer (saves to `custom_tokenizer/`).
   - Preprocess the dataset.
   - Build the model with `make_model(vocab_size)`.
   - Run the training loop.
   - Plot metrics.
   - Generate sample summaries.

The notebook is self-contained — no external scripts or config files are needed.

---

## Project Structure

```
.
├── Group_31_Transformer_based_summarizer.ipynb   # Main notebook
├── custom_tokenizer/                              # Generated at runtime
│   ├── vocab.json
│   └── merges.txt
├── corpus.txt                                     # Generated at runtime; tokenizer training corpus
└── README.md
```

---

## Known Limitations and Notes

A few honest caveats for anyone using or extending this work:

- **Per-epoch evaluation is on a single sample** (`raw_dataset[0]`). This is useful for watching learning progress but is not a statistically meaningful evaluation. For a proper test, evaluate on a held-out validation split.
- **The dataset slice is small** (1% of arXiv train). The model is unlikely to generalize well to arbitrary scientific abstracts with this much data.
- **Model size is large relative to the data**. `d_model=768, d_ff=3072, N=6` is roughly GPT-2-base scale. With more data or a smaller model, training would likely be more stable.
- **Input length of 128 tokens is short** for full scientific papers. Articles are heavily truncated; consider longer context (or a long-context Transformer variant) for better fidelity.
- **The `clean_and_filter` function is defined but not applied** in the main pipeline. If you want to use it, apply it via `dataset.map(clean_and_filter, ...)` before tokenization.
- **The decoder uses sampling, not argmax**, despite being named `greedy_decode`. See the Decoding section above.

---

## Suggested Extensions

If you want to build on this project:

- Replace the per-epoch single-sample evaluation with a proper validation loop over a held-out split.
- Add beam search to the decoder for better generation quality.
- Train on a larger slice of the arXiv data (e.g. `train[:25%]` or the full split).
- Add label smoothing (as in the original Transformer paper) and warmup-then-decay learning rate scheduling — both meaningfully improve convergence.
- Compare against a pre-trained baseline (`facebook/bart-large-cnn` or `google/pegasus-arxiv`) to contextualize the from-scratch results.
- Use gradient accumulation to enable larger effective batch sizes on limited GPU memory.

---

## References

- Vaswani et al., *Attention is All You Need*, NeurIPS 2017. [[paper]](https://arxiv.org/abs/1706.03762)
- Rush et al., *The Annotated Transformer*. [[blog]](http://nlp.seas.harvard.edu/annotated-transformer/)
- Cohan et al., *A Discourse-Aware Attention Model for Abstractive Summarization of Long Documents*, NAACL 2018 (arXiv dataset). [[paper]](https://arxiv.org/abs/1804.05685)
- Lin, *ROUGE: A Package for Automatic Evaluation of Summaries*, ACL Workshop 2004. [[paper]](https://aclanthology.org/W04-1013/)
- Zhang et al., *BERTScore: Evaluating Text Generation with BERT*, ICLR 2020. [[paper]](https://arxiv.org/abs/1904.09675)

---

## Authors

Group 31

*(Add team member names and affiliation here before publishing.)*

---

## License

Add a license here before making the repository public. The MIT License is a reasonable default for educational projects; if any external dataset or code requires attribution, document that here.
