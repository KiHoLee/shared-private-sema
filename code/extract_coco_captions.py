"""Extract BERT embeddings of MS-COCO captions for E24 (Reviewer 4, Weak
Aspect 1 and Recommended Change 1).

Reviewer 4 asked for validation on real multi-device data in which the
shared/private split is not imposed. In MS-COCO every image carries five
(occasionally six or seven) captions written independently by different
annotators. Each image therefore stands for one scene and its captions for
independent observations of that scene, so their embeddings share content
naturally and nothing is constructed. This script produces the embedding
pools that code/exp24_coco_captions.py reads.

Source and license
  MS-COCO 2017 caption annotations, captions_train2017.json and
  captions_val2017.json (T.-Y. Lin et al., "Microsoft COCO: Common Objects
  in Context," ECCV 2014; https://cocodataset.org). The annotations are
  distributed under the Creative Commons Attribution 4.0 License. Only the
  caption text is used, no image is downloaded.

Selection
  val2017    all 5,000 images and all of their captions
  train2017  the image ids sorted ascending, permuted with
             numpy.random.default_rng(SEED_IMG), and the first 21,600 kept,
             in that permuted order. exp24_coco_captions.py uses the first
             1,600 of them for calibration and the remaining 20,000 as a
             second, larger evaluation pool. All captions of a kept image
             are embedded, in annotation-id order.

Encoder, identical to code/extract_bert_embeddings.py
  bert-base-uncased (local Hugging Face cache), last hidden layer,
  tokenizer truncation at 64 tokens, mean pooling over the non-padding
  tokens, torch.manual_seed(0). Caption text is stripped of surrounding
  white space and otherwise left unchanged.

Output (raw pooled vectors, float32, not centered or normalized, as for
the AG-News pool; exp24 centers with the mean of the calibration captions
and normalizes every vector to unit norm, the EmbeddingPool convention)
  <out>/coco_bert_val2017.pt     dict(emb, image_id, caption_id, text)
  <out>/coco_bert_train21600.pt  same keys, plus image_order (the 21,600
                                 kept image ids in permuted order)
  data/coco_pool_notes.txt       counts, seeds and dimension

Large .pt files stay on the compute server under ~/tmc_rev/data_ext/coco/.

Usage (on u24):
    python code/extract_coco_captions.py \
        --ann ~/tmc_rev/data_ext/coco/annotations \
        --out ~/tmc_rev/data_ext/coco
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
from transformers import BertModel, BertTokenizer

SEED = 0            # torch seed, as in extract_bert_embeddings.py
SEED_IMG = 2024     # permutation of the train2017 image ids
N_TRAIN_IMG = 21600  # 1,600 calibration + 20,000 evaluation images

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


@torch.no_grad()
def encode_batch(model, tokenizer, texts, device, max_length=64):
    """Mean-pool last-layer token vectors over non-padding positions."""
    inputs = tokenizer(texts, padding=True, truncation=True,
                       max_length=max_length,
                       return_tensors="pt").to(device)
    hidden = model(**inputs).last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    return ((hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)).cpu()


def load_captions(path):
    """Captions grouped by image id, each group in annotation-id order."""
    with open(path) as f:
        js = json.load(f)
    by_img = defaultdict(list)
    for a in js["annotations"]:
        by_img[a["image_id"]].append((a["id"], a["caption"].strip()))
    for k in by_img:
        by_img[k].sort()
    n_img = len(js["images"])
    return by_img, n_img, len(js["annotations"])


def flatten(by_img, image_ids):
    img, cid, txt = [], [], []
    for i in image_ids:
        for c, t in by_img[i]:
            img.append(i)
            cid.append(c)
            txt.append(t)
    return np.array(img), np.array(cid), txt


def embed(model, tokenizer, texts, device, bs):
    out = []
    for s in range(0, len(texts), bs):
        out.append(encode_batch(model, tokenizer, texts[s:s + bs], device))
    return torch.cat(out).float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ann", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="bert-base-uncased")
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(args.model)
    model = BertModel.from_pretrained(args.model).to(device).eval()
    os.makedirs(args.out, exist_ok=True)
    notes = ["MS-COCO 2017 captions (CC BY 4.0), bert-base-uncased, last "
             "layer, mean pooling, truncation 64, torch seed 0",
             f"device {device}"]

    # ---- val2017: every image, every caption -------------------------
    by_v, n_img_v, n_cap_v = load_captions(
        os.path.join(args.ann, "captions_val2017.json"))
    ids_v = sorted(by_v)
    img, cid, txt = flatten(by_v, ids_v)
    emb = embed(model, tokenizer, txt, device, args.batch_size)
    torch.save(dict(emb=emb, image_id=img, caption_id=cid, text=txt),
               os.path.join(args.out, "coco_bert_val2017.pt"))
    cnt = np.bincount([len(by_v[i]) for i in ids_v])
    notes.append(f"val2017: {len(ids_v)} images with captions (json lists "
                 f"{n_img_v}), {len(txt)} captions embedded (json lists "
                 f"{n_cap_v}), captions per image histogram "
                 + ", ".join(f"{k}:{v}" for k, v in enumerate(cnt) if v)
                 + f", emb {tuple(emb.shape)}")
    print(notes[-1], flush=True)

    # ---- train2017: seeded subset of 21,600 images -------------------
    by_t, n_img_t, n_cap_t = load_captions(
        os.path.join(args.ann, "captions_train2017.json"))
    ids_all = np.array(sorted(by_t))
    perm = np.random.default_rng(SEED_IMG).permutation(len(ids_all))
    order = ids_all[perm[:N_TRAIN_IMG]]
    img, cid, txt = flatten(by_t, order)
    emb = embed(model, tokenizer, txt, device, args.batch_size)
    torch.save(dict(emb=emb, image_id=img, caption_id=cid, text=txt,
                    image_order=order),
               os.path.join(args.out, f"coco_bert_train{N_TRAIN_IMG}.pt"))
    cnt = np.bincount([len(by_t[i]) for i in order])
    notes.append(f"train2017: {len(ids_all)} images with captions (json "
                 f"lists {n_img_t}, {n_cap_t} captions), {N_TRAIN_IMG} kept "
                 f"by default_rng({SEED_IMG}).permutation of the sorted ids "
                 f"(first 1600 calibration, next 20000 evaluation), "
                 f"{len(txt)} captions embedded, captions per image "
                 "histogram " + ", ".join(f"{k}:{v}" for k, v in
                                          enumerate(cnt) if v)
                 + f", emb {tuple(emb.shape)}")
    print(notes[-1], flush=True)
    notes.append("dimension 768; raw pooled vectors saved as float32; "
                 "exp24 centers with the mean of the calibration captions "
                 "and normalizes to unit norm")
    notes.append("files on u24: ~/tmc_rev/data_ext/coco/coco_bert_val2017.pt, "
                 f"~/tmc_rev/data_ext/coco/coco_bert_train{N_TRAIN_IMG}.pt")
    with open(os.path.join(DATA, "coco_pool_notes.txt"), "w") as f:
        f.write("\n".join(notes) + "\n")
    print("done.")


if __name__ == "__main__":
    main()
