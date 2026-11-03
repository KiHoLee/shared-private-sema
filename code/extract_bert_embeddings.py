"""Extract the BERT sentence-embedding pool data/bert_agnews_8000.pt.

Every script that uses real embeddings reads the shipped pool
data/bert_agnews_8000.pt. This script regenerates it from the corpus:

  corpus     AG-News training split, in its stored order
  sentences  the text before the first period of each item, kept when it
             has 5 to 30 words, until 8,000 are collected, then shuffled
             with Python's random module at seed 0
  encoder    bert-base-uncased, last hidden layer, tokenizer truncation at
             64 tokens, mean pooling over the non-padding tokens
  output     raw pooled vectors (8000 x 768), not centered or normalized;
             semantic_mac.EmbeddingPool centers and normalizes them

The pool was produced with the dataset id "ag_news". Recent releases of
the datasets library reject that id, so pass a mirror of the same split
with --dataset. With --dataset wangrongsheng/ag_news (transformers 5.22,
datasets 5.1, RTX PRO 5000) the output matches the shipped pool to a
maximum absolute difference of 2.2e-5 per coordinate (data/bert_pool_regen.txt).
The default output name keeps the shipped pool intact.

Usage:
    python3 code/extract_bert_embeddings.py --dataset wangrongsheng/ag_news
Requires the transformers and datasets packages and network access.
"""
import argparse
import os
import random

import torch
from transformers import BertModel, BertTokenizer


def load_agnews_sentences(dataset, max_sentences, min_len=5, max_len=30):
    from datasets import load_dataset
    sentences = []
    for example in load_dataset(dataset, split="train"):
        first = example["text"].split(".")[0].strip()
        if min_len <= len(first.split()) <= max_len:
            sentences.append(first)
        if len(sentences) >= max_sentences:
            break
    if len(sentences) < max_sentences:
        raise RuntimeError(f"only {len(sentences)} sentences found")
    random.shuffle(sentences)
    return sentences


@torch.no_grad()
def encode_batch(model, tokenizer, texts, device, max_length=64):
    """Mean-pool last-layer token vectors over non-padding positions."""
    inputs = tokenizer(texts, padding=True, truncation=True,
                       max_length=max_length,
                       return_tensors="pt").to(device)
    hidden = model(**inputs).last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    return ((hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)).cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="ag_news")
    p.add_argument("--model", default="bert-base-uncased")
    p.add_argument("--out", default="data/bert_agnews_8000_regen.pt")
    p.add_argument("--max-sentences", type=int, default=8000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(args.model)
    model = BertModel.from_pretrained(args.model).to(device).eval()

    sents = load_agnews_sentences(args.dataset, args.max_sentences)
    emb = torch.cat([encode_batch(model, tokenizer,
                                  sents[i:i + args.batch_size], device)
                     for i in range(0, len(sents), args.batch_size)])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(emb, args.out)
    print(f"saved {tuple(emb.shape)} to {args.out}")


if __name__ == "__main__":
    main()
