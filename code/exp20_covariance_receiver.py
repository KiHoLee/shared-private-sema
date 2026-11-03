"""E20: a covariance-aware member of the decomposition receiver (Reviewer 1.1).

DR-MMSE (exp6_sideinfo.demux_block_lmmse) is the linear MMSE receiver of the
structured class when the contents are isotropic within each block. BERT
contents are not. The calibration pairs that DR-spectral and DR-adapter
already use to estimate the basis also give the covariance within each
block, so the class contains a receiver with no further side information
that uses those covariances:

  DR-cov  rotate the front-end output into the eigenvectors of the sample
          covariance within the shared block and within its complement,
          both estimated from the same N = 200 calibration tuples, and
          apply in every eigen-coordinate k the U x U linear MMSE filter
          with prior a a^T lambda_k (shared) or diag(1 - a_u^2) lambda_k
          (private). The cost is one U x U solve per coordinate and sample,
          O(d U^3) per embedding tuple, against O(U^3) for the DR.

The run also contains an unstructured reference outside the class:

  LMMSE-fullcov  the linear MMSE receiver of the stacked U d model with
          the own covariance and the raw sample cross-covariance of the
          same N = 200 calibration tuples, with no basis and no split
          (prior kron(I, S0) + kron(11^T - I, Sc), clipped to be positive
          semidefinite). It needs one U d x U d solve per sample, computed
          in float64 on a GPU when one is present.

Setting: the E9 construction exactly (exp9_retrieval.py seeds 7 and 909,
beta = 0.5, 4,000 held-out tuples, Rayleigh fading), at 10, 16, 20 and
30 dB, for the true, spectral and adapter bases. With the true basis the
covariances come from the same 200 tuples in the true frame. The E9 rows
(DR, DR-MMSE, DR-spectral, DR-adapter, LMMSE) are recomputed in the same
run. Run on u24 (Python 3.12, NumPy 2.5), the 20 dB rows reproduce
data/e9_retrieval.csv, which was produced with NumPy 2.2, to within 0.002.

Outputs: data/e20_covariance_receiver.csv, data/e20_notes.txt
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
                          learn_structure_spectral, metrics)
from exp6_sideinfo import demux_block_lmmse
from exp2_structure import train_adapter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
U, D, DC, BETA = E.U, E.D, E.DC, E.BETA
SNRS = [10, 16, 20, 30]


def block_eig(x_cal, Vs, Vp, a):
    """Eigenbasis and variances of the shared and private blocks."""
    am = a[None, :, None]
    xs = ((x_cal @ Vs) / am).reshape(-1, Vs.shape[1])
    xp = ((x_cal @ Vp) / np.sqrt(1 - am ** 2)).reshape(-1, Vp.shape[1])
    ls, Es = np.linalg.eigh(xs.T @ xs / len(xs))
    lp, Ep = np.linalg.eigh(xp.T @ xp / len(xp))
    return (np.hstack([Vs @ Es, Vp @ Ep]), np.maximum(ls, 1e-6),
            np.maximum(lp, 1e-6))


def demux_cov(tilde, B, h, rho, a, Rb, var_s, var_p, chunk=200):
    """Per-eigen-coordinate U x U linear MMSE filter (DR-cov)."""
    s2 = 1.0 / rho
    t = tilde @ Rb
    out = np.empty_like(t)
    dc = len(var_s)
    blocks = [(np.outer(a, a), var_s, slice(0, dc)),
              (np.diag(1 - a ** 2), var_p, slice(dc, t.shape[2]))]
    for s in range(0, t.shape[0], chunk):
        hb = h[s:s + chunk]
        G = (1 / hb)[:, :, None] * B[None] * hb[:, None, :]
        Cn = s2 * (1 / hb)[:, :, None] * B[None] * (1 / hb)[:, None, :]
        for P, var, sl in blocks:
            Cx = P[None, None] * var[None, :, None, None]
            Gx = G[:, None]
            S = Gx @ Cx @ np.swapaxes(Gx, -1, -2) + Cn[:, None]
            A = Cx @ np.swapaxes(Gx, -1, -2)
            y = np.swapaxes(t[s:s + chunk, :, sl], 1, 2)[..., None]
            r = A @ np.linalg.solve(S, y)
            out[s:s + chunk, :, sl] = np.swapaxes(r[..., 0], 1, 2)
    return out @ Rb.T


def fullcov_prior(x_cal):
    """Stacked U d prior from the calibration tuples (no basis, no split)."""
    n, u_, d = x_cal.shape
    x0 = x_cal.reshape(-1, d)
    s0 = x0.T @ x0 / len(x0)
    tot = x_cal.sum(axis=1)
    sc = (tot.T @ tot - x0.T @ x0) / (u_ * (u_ - 1) * n)
    sc = (sc + sc.T) / 2
    sig = (np.kron(np.eye(u_), s0) +
           np.kron(np.ones((u_, u_)) - np.eye(u_), sc))
    w, q = np.linalg.eigh(sig)
    return (q * np.clip(w, 0, None)) @ q.T


def demux_fullcov(tilde, B, h, rho, sig, chunk=16):
    """Linear MMSE of the stacked model (LMMSE-fullcov).

    With Gamma = H^-1 B H, y = Gamma^-1 tilde = x + n' and
    Cov(n') = sigma^2 H^-1 B^-1 H^-1 (x) I_d, so the estimate is
    Sig (Sig + Cov(n'))^-1 y, one U d x U d Cholesky solve per sample."""
    batch, u_, d = tilde.shape
    s2 = 1.0 / rho
    binv = np.linalg.inv(B)
    ginv = (1 / h)[:, :, None] * binv[None] * h[:, None, :]
    y = np.einsum('buv,bvd->bud', ginv, tilde).reshape(batch, -1)
    cb = s2 * (1 / h)[:, :, None] * binv[None] * (1 / h)[:, None, :]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S = torch.as_tensor(sig, dtype=torch.float64, device=dev)
    eye = torch.eye(d, dtype=torch.float64, device=dev)
    out = np.empty((batch, u_ * d))
    for s in range(0, batch, chunk):
        c = torch.as_tensor(cb[s:s + chunk], dtype=torch.float64, device=dev)
        n_ = (c[:, :, None, :, None] * eye[None, None, :, None, :])
        M = S[None] + n_.reshape(-1, u_ * d, u_ * d)
        L = torch.linalg.cholesky(M)
        yb = torch.as_tensor(y[s:s + chunk], dtype=torch.float64,
                             device=dev)[..., None]
        z = torch.cholesky_solve(yb, L)
        out[s:s + chunk] = (S[None] @ z)[..., 0].cpu().numpy()
    return out.reshape(batch, u_, d)


def main():
    rng = np.random.default_rng(7)
    obj = torch.load(E.POOL_PT, map_location="cpu")
    pool = EmbeddingPool(obj.numpy() if torch.is_tensor(obj)
                         else np.asarray(obj))
    a = math.sqrt(BETA) * np.ones(U)
    B = affinity_matrix(a)
    R = random_orthogonal(D, rng)
    z_cal, _ = E.sample_with_index(pool, E.N_PAIR, a, rng, E.IDX_TRAIN)
    x_cal = z_cal @ R.T
    V_spec, _ = learn_structure_spectral(x_cal, DC)
    Q, _ = np.linalg.qr(np.hstack([V_spec, rng.standard_normal((D, D - DC))]))
    W_ad = train_adapter(x_cal, a, Q, rng)
    bases = {
        "true": (R[:, :DC], R[:, DC:]),
        "spec": (V_spec, Q[:, DC:]),
        "adapt": (W_ad.T[:, :DC], W_ad.T[:, DC:]),
    }
    eig = {k: block_eig(x_cal, *v, a) for k, v in bases.items()}
    sig_fc = fullcov_prior(x_cal)

    bank_s = pool.pca_coords(E.IDX_EVAL, DC)
    bank_p = pool.pca_coords(E.IDX_EVAL, D - DC)
    pos = {g: i for i, g in enumerate(E.IDX_EVAL)}
    rows, notes = [], []
    for snr in SNRS:
        rho = 10 ** (snr / 10)
        rng_e = np.random.default_rng(909)
        z, idx = E.sample_with_index(pool, E.BATCH_EVAL, a, rng_e, E.IDX_EVAL)
        x = z @ R.T
        tilde, h = matched_filter(x, B, rho, rng_e)
        est = {"LMMSE": demux_lmmse(tilde, B, h, rho)[0]}
        for k, (Vs, _) in bases.items():
            est[f"DR-{k}"] = demux_dr(tilde, B, h, rho, a, Vs)
            est[f"DR-cov-{k}"] = demux_cov(tilde, B, h, rho, a, *eig[k])
        est["DR-MMSE-true"] = demux_block_lmmse(tilde, B, h, rho, a, R[:, :DC])
        est["LMMSE-fullcov"] = demux_fullcov(tilde, B, h, rho, sig_fc)
        ts = np.array([pos[g] for g in idx[:, 0]])
        for name, e in est.items():
            cos, nmse, ser = metrics(e, x)
            zh = e @ R
            hs, hp = [], []
            for u in range(U):
                tp = np.array([pos[g] for g in idx[:, u + 1]])
                hs.append(E.top1(zh[:, u, :DC], bank_s, ts))
                hp.append(E.top1(zh[:, u, DC:], bank_p, tp))
            rec = dict(snr=snr, scheme=name, cos=cos, nmse=nmse, ser=ser,
                       shared_top1=float(np.mean(np.concatenate(hs))),
                       private_top1=float(np.mean(np.concatenate(hp))))
            rows.append(rec)
            notes.append(f"snr={snr:2d} {name:13s} cos={cos:.4f} "
                         f"nmse={nmse:.4f} ser={ser:.4f} "
                         f"shared_top1={rec['shared_top1']:.4f} "
                         f"private_top1={rec['private_top1']:.4f}")
            print(notes[-1], flush=True)
    with open(os.path.join(DATA, "e20_covariance_receiver.csv"), "w") as f:
        f.write("snr,scheme,cos,nmse,ser,shared_top1,private_top1\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in
                             ("snr", "scheme", "cos", "nmse", "ser",
                              "shared_top1", "private_top1")) + "\n")
    with open(os.path.join(DATA, "e20_notes.txt"), "w") as f:
        f.write("\n".join(notes) + "\n")
    print("E20 done.")


if __name__ == "__main__":
    main()
