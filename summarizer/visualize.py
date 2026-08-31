"""Plots of token-length distributions and of the training metric trajectories."""

import matplotlib.pyplot as plt


# Plotting token length distribution of articles and summaries
def plot_token_lengths(dataset, tokenizer):
    article_lengths = [len(tokenizer.encode(sample["article"]).ids) for sample in dataset]
    summary_lengths = [len(tokenizer.encode(sample["abstract"]).ids) for sample in dataset]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(article_lengths, bins=30, color="skyblue")
    plt.title("Article Token Length Distribution")
    plt.xlabel("Tokens")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(summary_lengths, bins=30, color="lightgreen")
    plt.title("Summary Token Length Distribution")
    plt.xlabel("Tokens")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


# Plotting training metrics over epochs
def plot_training_history(loss_history, history):
    plt.figure(figsize=(14, 8))

    # Plotting loss per epoch
    plt.subplot(2, 3, 1)
    plt.plot(loss_history, marker='o')
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    # Plotting ROUGE scores per epoch
    plt.subplot(2, 3, 2)
    plt.plot(history["rouge1"], label="ROUGE-1", marker='o')
    plt.plot(history["rouge2"], label="ROUGE-2", marker='s')
    plt.plot(history["rougeL"], label="ROUGE-L", marker='^')
    plt.title("ROUGE Scores")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.legend()
    plt.grid(True)
    # Plotting BLEU score per epoch
    plt.subplot(2, 3, 3)
    plt.plot(history["bleu"], marker='*', color='orange')
    plt.title("BLEU Score")
    plt.xlabel("Epoch")
    plt.ylabel("BLEU")
    plt.grid(True)
    # Plotting BERTScore F1 per epoch
    plt.subplot(2, 3, 4)
    plt.plot(history["bertscore_f1"], marker='d', color='purple')
    plt.title("BERTScore F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
