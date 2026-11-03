"""E21: stronger conventional references on BERT contents (Reviewer 3.4).

OMA in the paper confines each user to a fixed block of d/U coordinates,
and the genie NOMA-SIC reference estimates each user from an
interference-free look without a content prior. On anisotropic BERT
contents both can be strengthened with knowledge of the marginal content
covariance, which a transmitter or a per-user decoder can learn from its own
unpaired history. This script measures

  OMA-PCA-power   each user sends its top d/U principal coordinates with
                  the per-dimension power of the paper's OMA, Wiener
                  reception per coordinate
  OMA-PCA-energy  the same with the full per-user energy on those
                  coordinates (equal energy to the superposed schemes)
  NOMA-SIC-cov    the genie NOMA-SIC look followed by a Wiener filter in the
                  eigenbasis of the marginal content covariance

beside the paper's OMA, NOMA-SIC, LMMSE and DR-spectral, on the E9
construction (exp9_retrieval.py seeds 7 and 909, beta = 0.5, 4,000
held-out tuples, Rayleigh fading) at 10, 16 and 20 dB. The marginal
covariance comes from 3,000 tuples of the training partition (seed 11).

It also reports, at 20 dB, how close the private retrieval candidates are:
the nearest-neighbour cosine within the held-out private bank, retrieval
accuracy by quartile of that cosine, and the cosine between a wrong pick
and the true sentence.

Outputs: data/e21_conventional_references.csv, data/e21_notes.txt
"""
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp9_retrieval as E
from semantic_mac import (EmbeddingPool, affinity_matrix, matched_filter,
                          demux_lmmse, demux_dr, random_orthogonal,
                          learn_structure_spectral, metrics, oma_observe,
                          demux_noma_genie)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
U, D, DC, BETA = E.U, E.D, E.DC, E.BETA
SNRS = [10, 16, 20]


def main():
    rng = np.random.default_rng(7)
    obj = torch.load(E.POOL_PT, map_location="cpu")
    pool = EmbeddingPool(obj.numpy() if torch.is_tensor(obj)
                         else np.asarray(obj))
    a = math.sqrt(BETA) * np.ones(U)
    B = affinity_matrix(a)
    R = random_orthogonal(D, rng)
    z_cal, _ = E.sample_with_index(pool, E.N_PAIR, a, rng, E.IDX_TRAIN)
    V_spec, _ = learn_structure_spectral(z_cal @ R.T, DC)

    zt, _ = E.sample_with_index(pool, 3000, a, np.random.default_rng(11),
                                E.IDX_TRAIN)
    xt = (zt @ R.T).reshape(-1, D)
    w, V = np.linalg.eigh(xt.T @ xt / len(xt))
    w, V = w[::-1], V[:, ::-1]
    k = D // U
    Pk, lam = V[:, :k], w[:k]
    notes = [f"top-{k} principal directions hold {lam.sum() / w.sum():.4f} "
             "of the content energy"]

    bank_s = pool.pca_coords(E.IDX_EVAL, DC)
    bank_p = pool.pca_coords(E.IDX_EVAL, D - DC)
    pos = {g: i for i, g in enumerate(E.IDX_EVAL)}
    rows = []
    for snr in SNRS:
        rho = 10 ** (snr / 10)
        s2 = 1.0 / rho
        rng_e = np.random.default_rng(909)
        z, idx = E.sample_with_index(pool, E.BATCH_EVAL, a, rng_e, E.IDX_EVAL)
        x = z @ R.T
        tilde, h = matched_filter(x, B, rho, rng_e)
        rng_c = np.random.default_rng(60001)
        est = {"OMA": oma_observe(x, h, rho, rng_c)}
        y_noma = demux_noma_genie(x, h, rho, rng_c)
        est["NOMA-SIC"] = y_noma
        est["LMMSE"] = demux_lmmse(tilde, B, h, rho)[0]
        est["DR-spectral"] = demux_dr(tilde, B, h, rho, a, V_spec)
        rr = np.random.default_rng(70001 + snr)
        s = x @ Pk
        n = rr.standard_normal(s.shape) * math.sqrt(s2) / h[:, :, None]
        for lab, energy in (("OMA-PCA-power", k / D), ("OMA-PCA-energy", 1.0)):
            g = math.sqrt(energy / lam.sum())
            wien = (g * lam)[None, None] / ((g * g * lam)[None, None]
                                            + s2 / h[:, :, None] ** 2)
            est[lab] = (wien * (g * s + n)) @ Pk.T
        wz = w[None, None] / (w[None, None] + s2 / h[:, :, None] ** 2)
        est["NOMA-SIC-cov"] = (wz * (y_noma @ V)) @ V.T
        ts = np.array([pos[q] for q in idx[:, 0]])
        for name, e in est.items():
            cos, nmse, ser = metrics(e, x)
            zh = e @ R
            hs, hp = [], []
            for u in range(U):
                tp = np.array([pos[q] for q in idx[:, u + 1]])
                hs.append(E.top1(zh[:, u, :DC], bank_s, ts))
                hp.append(E.top1(zh[:, u, DC:], bank_p, tp))
            rec = dict(snr=snr, scheme=name, cos=cos, nmse=nmse, ser=ser,
                       shared_top1=float(np.mean(np.concatenate(hs))),
                       private_top1=float(np.mean(np.concatenate(hp))))
            rows.append(rec)
            notes.append(f"snr={snr:2d} {name:15s} cos={cos:.4f} "
                         f"ser={ser:.4f} shared_top1={rec['shared_top1']:.4f} "
                         f"private_top1={rec['private_top1']:.4f}")
            print(notes[-1], flush=True)

        if snr == 20:
            bp = bank_p / np.linalg.norm(bank_p, axis=1, keepdims=True)
            bs = bank_s / np.linalg.norm(bank_s, axis=1, keepdims=True)
            G = bp @ bp.T
            np.fill_diagonal(G, -1)
            Gs = bs @ bs.T
            np.fill_diagonal(Gs, -1)
            nnc = G.max(1)
            notes.append(f"nearest-neighbour cosine, median: private bank "
                         f"{np.median(nnc):.3f}, shared bank "
                         f"{np.median(Gs.max(1)):.3f}; fraction above 0.9: "
                         f"private {(nnc > 0.9).mean():.3f}, shared "
                         f"{(Gs.max(1) > 0.9).mean():.3f}")
            qs = np.quantile(nnc, [0, .25, .5, .75, 1])
            for name in ("DR-spectral", "LMMSE"):
                zh = est[name] @ R
                hit, nn_t, wrong = [], [], []
                for u in range(U):
                    tp = np.array([pos[q] for q in idx[:, u + 1]])
                    qv = zh[:, u, DC:]
                    qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True)
                               + 1e-12)
                    pick = (qv @ bp.T).argmax(1)
                    hit.append(pick == tp)
                    nn_t.append(nnc[tp])
                    wrong.append(G[tp, pick])
                hit, nn_t, wrong = map(np.concatenate, (hit, nn_t, wrong))
                quart = [hit[(nn_t >= lo) & (nn_t <= hi)].mean()
                         for lo, hi in zip(qs[:-1], qs[1:])]
                notes.append(f"{name}: private top-1 by quartile of the true "
                             "sentence's nearest-neighbour cosine "
                             + " / ".join(f"{q:.3f}" for q in quart)
                             + f"; wrong pick vs truth cosine median "
                             f"{np.median(wrong[~hit]):.3f}, above 0.9 "
                             f"{(wrong[~hit] > 0.9).mean():.3f}")
                print(notes[-1], flush=True)

    with open(os.path.join(DATA, "e21_conventional_references.csv"), "w") as f:
        f.write("snr,scheme,cos,nmse,ser,shared_top1,private_top1\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in
                             ("snr", "scheme", "cos", "nmse", "ser",
                              "shared_top1", "private_top1")) + "\n")
    with open(os.path.join(DATA, "e21_notes.txt"), "w") as f:
        f.write("\n".join(notes) + "\n")
    print("E21 done.")


if __name__ == "__main__":
    main()
