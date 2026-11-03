"""Shared-only check: how much of the SER gain is shared recovery.

A shared estimate alone reaches a cosine of a_u with the transmitted
embedding, so at a lenient threshold the SER cannot tell whether private
content was recovered. This script runs a receiver that discards the
private output (the DR fed the V_c-projected matched-filter output, so its
output is a_u V_c c_hat exactly) beside the DR and the unstructured
receivers.

  synthetic : E10 setting (d=64, d_c=16, U=4, 10 dB), beta = 0.3 to 0.7,
              ten chunks of 8,000 tuples per point, every threshold of E10
  real      : E2/E9 construction on BERT embeddings (beta = 0.5, N = 200,
              20 dB), five random rotations (seeds 7 to 11), 4,000
              held-out tuples each, spectral, adapter and true bases

Outputs: data/e10_shared_only.csv, data/e10_shared_only_real.csv
Run with the argument "syn" or "real" to run one part only.
"""
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from semantic_mac import (affinity_matrix, matched_filter, demux_sr,
                          demux_sc, demux_lmmse, demux_dr,
                          sample_latents_isotropic, U_MAIN, D_SYN, DC_SYN,
                          D_REAL, DC_REAL)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
TAUS = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def unit(e):
    return e / (np.linalg.norm(e, axis=-1, keepdims=True) + 1e-12)


def shared_only(tilde, B, h, rho, a, Vc):
    """The DR with its private output discarded: output a_u V_c c_hat."""
    return demux_dr((tilde @ Vc) @ Vc.T, B, h, rho, a, Vc)


def block_cos(e, x, V):
    P = lambda y: (y @ V) @ V.T
    es, xs = P(e), P(x)
    ep, xp = e - es, x - xs
    return ((unit(es) * unit(xs)).sum(-1).mean(),
            (unit(ep) * unit(xp)).sum(-1).mean())


def summarize(e, x, V):
    c = (unit(e) * x).sum(-1)
    cs, cp = block_cos(e, x, V)
    r = dict(cos=c.mean(), nmse=((e - x) ** 2).sum(-1).mean(),
             cos_shared=cs, cos_private=cp)
    r.update({f"ser{t:.2f}": (c < t).mean() for t in TAUS})
    return r


def synthetic():
    U, D, DC = U_MAIN, D_SYN, DC_SYN
    Vc = np.eye(D)[:, :DC]
    rows = []
    for snr, n_ch in ((10, 10), (20, 4)):
        rho = 10 ** (snr / 10)
        for beta in ((0.3, 0.4, 0.5, 0.6, 0.7) if snr == 10 else (0.5,)):
            a = math.sqrt(beta) * np.ones(U)
            B = affinity_matrix(a)
            acc = {}
            for ch in range(n_ch):
                rng = np.random.default_rng(
                    1000000 + snr * 1000 + int(beta * 100) * 10 + ch)
                z = sample_latents_isotropic(8000, U, D, DC, a, rng)
                tilde, h = matched_filter(z, B, rho, rng)
                est = dict(SR=demux_sr(tilde, B, h), SC=demux_sc(tilde, h),
                           LMMSE=demux_lmmse(tilde, B, h, rho)[0],
                           DR=demux_dr(tilde, B, h, rho, a, Vc),
                           SO=shared_only(tilde, B, h, rho, a, Vc))
                for k, e in est.items():
                    acc.setdefault(k, []).append(summarize(e, z, Vc))
            for k, lst in acc.items():
                rec = dict(snr=snr, beta=beta, scheme=k, tuples=8000 * n_ch)
                for q in lst[0]:
                    v = np.array([r[q] for r in lst])
                    rec[q] = float(v.mean())
                    rec[q + "_se"] = float(v.std(ddof=1) / math.sqrt(n_ch))
                rows.append(rec)
                print(f"snr={snr} beta={beta} {k:5s} cos={rec['cos']:.4f} "
                      f"ser.45={rec['ser0.45']:.4f} ser.60={rec['ser0.60']:.4f}",
                      flush=True)
    with open(os.path.join(DATA, "e10_shared_only.csv"), "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def real():
    import torch
    from semantic_mac import (EmbeddingPool, random_orthogonal,
                              learn_structure_spectral, subspace_error)
    from exp2_structure import train_adapter, gen_clean
    U, D, DC = U_MAIN, D_REAL, DC_REAL
    obj = torch.load(os.path.join(DATA, "bert_agnews_8000.pt"),
                     map_location="cpu")
    pool = EmbeddingPool(obj.numpy() if torch.is_tensor(obj)
                         else np.asarray(obj))
    TR, EV = np.arange(7000), np.arange(7000, 8000)
    beta = 0.5
    a = math.sqrt(beta) * np.ones(U)
    B = affinity_matrix(a)
    rows = []
    for seed in (7, 8, 9, 10, 11):
        rng = np.random.default_rng(seed)
        R = random_orthogonal(D, rng)
        Vt = R[:, :DC]
        _, xc = gen_clean(pool, 200, a, R, rng, idx_pool=TR)
        Vs, _ = learn_structure_spectral(xc, DC)
        Q, _ = np.linalg.qr(np.hstack([Vs, rng.standard_normal((D, D - DC))]))
        Va = train_adapter(xc, a, Q, rng).T[:, :DC]
        print(f"seed={seed} subspace error spectral "
              f"{subspace_error(Vs, Vt):.4f} adapter "
              f"{subspace_error(Va, Vt):.4f}", flush=True)
        for snr in (16, 20, 30):
            rho = 10 ** (snr / 10)
            re = np.random.default_rng(90000 + seed * 100 + snr)
            _, x = gen_clean(pool, 4000, a, R, re, idx_pool=EV)
            tilde, h = matched_filter(x, B, rho, re)
            est = dict(LMMSE=demux_lmmse(tilde, B, h, rho)[0])
            for tag, V in (("true", Vt), ("spec", Vs), ("adapt", Va)):
                est["DR-" + tag] = demux_dr(tilde, B, h, rho, a, V)
                est["SO-" + tag] = shared_only(tilde, B, h, rho, a, V)
            for k, e in est.items():
                rec = dict(seed=seed, snr=snr, scheme=k)
                rec.update(summarize(e, x, Vt))
                rows.append(rec)
    with open(os.path.join(DATA, "e10_shared_only_real.csv"), "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for snr in (16, 20, 30):
        for k in sorted({r["scheme"] for r in rows}):
            v = [r for r in rows if r["snr"] == snr and r["scheme"] == k]
            print(f"snr={snr} {k:9s} cos={np.mean([r['cos'] for r in v]):.4f} "
                  f"ser.45={np.mean([r['ser0.45'] for r in v]):.4f}",
                  flush=True)


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    if part in ("syn", "all"):
        synthetic()
    if part in ("real", "all"):
        real()
    print("shared-only check done.")
