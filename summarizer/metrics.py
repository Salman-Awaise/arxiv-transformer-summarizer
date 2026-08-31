"""ROUGE, BLEU and BERTScore evaluation of a generated summary against its reference."""

from bert_score import score as bertscore
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]


def make_scorer():
    """Setting up metric helpers."""
    scorer = rouge_scorer.RougeScorer(ROUGE_TYPES, use_stemmer=True)
    smoothie = SmoothingFunction().method4# smoothing for BLEU
    return scorer, smoothie


def evaluate(prediction, reference, scorer, smoothie):
    """Score one prediction against its reference on all five metrics."""
    scores = scorer.score(reference, prediction)

    ref_tokens = word_tokenize(reference)
    pred_tokens = word_tokenize(prediction)
    bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)

    _, _, f1 = bertscore([prediction], [reference], lang="en", verbose=False)

    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
        "bleu": bleu,
        "bertscore_f1": f1[0].item(),
    }


def format_metrics(m):
    return (f"ROUGE-1: {m['rouge1']:.4f} | ROUGE-2: {m['rouge2']:.4f} | "
            f"ROUGE-L: {m['rougeL']:.4f}\n"
            f" BLEU: {m['bleu']:.4f} |  BERTScore F1: {m['bertscore_f1']:.4f}")
