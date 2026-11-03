"""Calibration over the air: the SNR the calibration upload needs.

Every E2 result estimates the shared basis from N = 200 clean calibration
tuples. This script sends the calibration tuples through a noisy link
instead and reports the spectral-basis subspace error and the DR SER at the
E2 operating point (beta = 0.5, 20 dB, held-out tuples), so the overhead
analysis can state what link quality, or how many tuples, the calibration
needs. Noise has variance 1/rho per coordinate, the convention of the
paper. "fadew" adds Rayleigh fading per tuple and user and weights the
cross-covariance by the inverse noise level h_u^2 h_v^2.

Output: data/e2_calibration_snr.csv
"""
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from semantic_mac import (EmbeddingPool, affinity_matrix, matched_filter,
                          demux_lmmse, demux_dr, sample_latents_pool,
                          random_orthogonal, learn_structure_spectral,
                          subspace_error, metrics, H_MIN, U_MAIN as U,
                          D_REAL as D, DC_REAL as DC)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
BETA, TEST_DB, DRAWS, BATCH = 0.5, 20, 12, 2000
IDX_TRAIN, IDX_EVAL = np.arange(7000), np.arange(7000, 8000)
# (link, N, calibration SNR in dB)
CASES = [("clean", 200, None), ("awgn", 200, 40), ("awgn", 200, 30),
         ("awgn", 200, 20), ("awgn", 3200, 20), ("awgn", 6400, 20),
         ("fadew", 6400, 20)]


def weighted_basis(x, w):
    N, Uu, d = x.shape
    M = np.zeros((d, d))
    for u in range(Uu):
        for v in range(u + 1, Uu):
            ww = w[:, u] * w[:, v]
            C = (x[:, u, :] * ww[:, None]).T @ x[:, v, :] / ww.sum()
            M += C + C.T
    ev, V = np.linalg.eigh(M)
    return V[:, np.argsort(ev)[::-1][:DC]]


def calibrate(pool, a, kind, N, snr_db, R, rng):
    x = sample_latents_pool(pool, N, U, D, DC, a, rng,
                            idx_pool=IDX_TRAIN) @ R.T
    if kind == "clean":
        return learn_structure_spectral(x, DC)[0]
    s = math.sqrt(10 ** (-snr_db / 10))
    if kind == "awgn":
        return learn_structure_spectral(
            x + rng.standard_normal(x.shape) * s, DC)[0]
    hc = (rng.standard_normal((N, U)) +
          1j * rng.standard_normal((N, U))) / math.sqrt(2)
    h = np.clip(np.abs(hc), H_MIN, None)
    return weighted_basis(x + rng.standard_normal(x.shape) * s
                          / h[:, :, None], h ** 2)


def main():
    obj = torch.load(os.path.join(DATA, "bert_agnews_8000.pt"),
                     map_location="cpu")
    pool = EmbeddingPool(obj.numpy() if torch.is_tensor(obj)
                         else np.asarray(obj))
    a = math.sqrt(BETA) * np.ones(U)
    B = affinity_matrix(a)
    rho = 10 ** (TEST_DB / 10)
    rows = []
    for ci, (kind, N, snr_db) in enumerate(CASES):
        err, dr, lm = [], [], []
        for m in range(DRAWS):
            rng = np.random.default_rng([4400, ci, m])
            R = random_orthogonal(D, rng)
            V = calibrate(pool, a, kind, N, snr_db, R, rng)
            err.append(subspace_error(V, R[:, :DC]))
            x = sample_latents_pool(pool, BATCH, U, D, DC, a, rng,
                                    idx_pool=IDX_EVAL) @ R.T
            t, h = matched_filter(x, B, rho, rng)
            dr.append(metrics(demux_dr(t, B, h, rho, a, V), x)[2])
            lm.append(metrics(demux_lmmse(t, B, h, rho)[0], x)[2])
        se = lambda v: float(np.std(v, ddof=1) / math.sqrt(len(v)))
        rows.append((kind, N, snr_db, float(np.mean(err)),
                     float(np.mean(dr)), se(dr), float(np.mean(lm))))
        print(f"{kind:6s} N={N:5d} cal_snr={snr_db} suberr={rows[-1][3]:.4f} "
              f"DR_ser={rows[-1][4]:.4f}+-{rows[-1][5]:.4f} "
              f"LMMSE_ser={rows[-1][6]:.4f}", flush=True)
    with open(os.path.join(DATA, "e2_calibration_snr.csv"), "w") as f:
        f.write("link,N,cal_snr_db,subspace_error,dr_ser,dr_ser_se,"
                "lmmse_ser\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")
    print("calibration check done.")


if __name__ == "__main__":
    main()
