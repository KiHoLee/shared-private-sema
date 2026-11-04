"""E20b: stronger full-covariance LMMSE references, calibration size and
receiver cost (Reviewer 1.1 and the AE summary).

E20 (exp20_covariance_receiver.py) found that the unstructured
full-covariance LMMSE receiver (LMMSE-fullcov, raw sample covariances of the
same N = 200 calibration tuples) beats DR-spectral, loses to DR-cov-adapter
at 10, 16 and 20 dB, and leaves an SER floor of 0.013 at 30 dB. A reviewer
can object that the raw sample covariance handicaps it. This script asks
whether a better estimated full-covariance prior closes the gap, whether
the estimated shared subspace helps it, how the ranking moves with the
number of calibration tuples, and what each receiver costs.

  (a) Shrinkage.  The stacked prior of E20 is
          Sig = kron(I_U, S0) + kron(11^T - I_U, Sc),
      the permutation-symmetrized sample covariance of the stacked U d
      vectors (uncentered, as in E20). It is shrunk towards mu I with
      mu = tr(Sig)/(U d) = 1/d,
          Sig_delta = (1 - delta) Sig + delta mu I,
      and then clipped to be positive semidefinite as in E20. The intensity
      delta is the Ledoit-Wolf value (LMMSE-fullcov-LW) or the OAS value
      (LMMSE-fullcov-OAS), both computed by sklearn.covariance with
      assume_centered=True from the N stacked calibration vectors of
      dimension U d. A grid delta in {0.01, 0.03, 0.1, 0.3, 0.6}
      (LMMSE-fullcov-shr<delta>) shows the best intensity in hindsight. It
      is chosen on the test tuples and is therefore optimistic for the
      full-covariance receiver.
  (b) Projection.  Sc is replaced by P Sc P with P the projector onto the
      estimated shared subspace (LMMSE-fullcov-proj-spec with the spectral
      basis, -proj-adapt with the adapter basis, -proj-true with the true
      basis as an oracle), and LMMSE-fullcov-LW-proj-adapt combines (a)
      and (b).
  (c) Calibration size.  At 20 dB, N in {100, 200, 400, 800, 1600}
      calibration tuples drawn exactly as E20 draws its 200 (rng seed 7,
      the rotation R first, then the N tuples, the QR completion and the
      adapter), so the N = 200 point is the E20 calibration and the first
      200 tuples of every larger N are the E20 tuples. DR-cov-adapter is
      run at every N (the adapter takes a few seconds), together with
      DR-spectral, DR-adapter, DR-cov-spectral, DR-cov-true and the
      full-covariance variants, all built from the same N tuples.
  (d) Cost.  Analytic multiply counts per coherence block (one fading
      realization) and per sample for DR, DR-cov and LMMSE-fullcov at
      d = 768, U = 4, d_c = 128, and measured wall clock per 1,000 samples
      of batched float64 torch ports of the same receivers on one CPU
      setting (4 threads) and on one GPU, for a block length of 1 (every
      sample has its own fading, as in all simulations) and 1,000.

Every full-covariance variant is solved exactly with a reduction that the
permutation symmetry of Sig allows (demux_fc_struct). With J = 11^T,
Sig = I (x) Bbar + (J/U) (x) (Abar - Bbar), Abar = clip(S0 + (U-1) Sc) and
Bbar = clip(S0 - Sc). In the eigenbasis of Bbar and of the U x U noise
covariance, the U d x U d solve of E20 becomes one d x d solve per fading
realization (Woodbury identity). The script checks it against the
U d x U d Cholesky of E20 (exp20_covariance_receiver.demux_fullcov) on 16
tuples at each SNR and against the E20 LMMSE-fullcov rows.

Setting: the E9 and E20 construction exactly (exp9_retrieval.py seeds 7 and
909, beta = 0.5, 4,000 held-out tuples, Rayleigh fading, SNR 10, 16, 20 and
30 dB, metrics and top-one retrieval as in E20). Timing seeds: 2024 for the
synthetic timing inputs. Run on u24 (Python 3.12, NumPy 2.5, torch 2.11).

Outputs: data/e20b_fullcov_variants.csv, data/e20b_calib_sweep.csv,
data/e20b_cost.csv, data/e20b_notes.txt
Usage: python exp20b_fullcov_variants.py --part main|sweep|cost|notes
"""
import argparse
import math
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np
import torch
from sklearn.covariance import ledoit_wolf_shrinkage, oas

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp9_retrieval as E
import exp20_covariance_receiver as E20
from semantic_mac import (EmbeddingPool, affinity_matrix, matched_filter,
                          demux_lmmse, demux_dr, random_orthogonal,
                          learn_structure_spectral, subspace_error, metrics)
from exp2_structure import train_adapter

torch.set_num_threads(4)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
LOGS = os.path.join(HERE, "..", "rev5_logs")
os.makedirs(LOGS, exist_ok=True)
U, D, DC, BETA = E.U, E.D, E.DC, E.BETA
SNRS = [10, 16, 20, 30]
N_SWEEP = [100, 200, 400, 800, 1600]
SHR_GRID = [0.01, 0.03, 0.1, 0.3, 0.6]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COLS = ("cos", "nmse", "ser", "ser_se", "shared_top1", "private_top1")


def _solve(A, b):
    """torch.linalg.solve, through NumPy LAPACK on the CPU: the batched
    CPU LU of the torch 2.11 build on u24 fails (MKL DLASWP error) for
    matrices above about 100 x 100, while the CUDA path is unaffected."""
    if A.device.type == "cpu":
        return torch.from_numpy(np.linalg.solve(A.numpy(), b.numpy()))
    return torch.linalg.solve(A, b)


def _inv(A):
    if A.device.type == "cpu":
        return torch.from_numpy(np.linalg.inv(A.numpy()))
    return torch.linalg.inv(A)


# ----------------------------------------------------------------------
#  Calibration exactly as in E20 main()
# ----------------------------------------------------------------------
def load_pool():
    obj = torch.load(E.POOL_PT, map_location="cpu")
    return EmbeddingPool(obj.numpy() if torch.is_tensor(obj)
                         else np.asarray(obj))


def calibrate(pool, a, n_pair):
    """Same draws as E20 (rng 7: R, the n_pair tuples, QR, adapter)."""
    rng = np.random.default_rng(7)
    R = random_orthogonal(D, rng)
    z_cal, _ = E.sample_with_index(pool, n_pair, a, rng, E.IDX_TRAIN)
    x_cal = z_cal @ R.T
    V_spec, _ = learn_structure_spectral(x_cal, DC)
    Q, _ = np.linalg.qr(np.hstack([V_spec, rng.standard_normal((D, D - DC))]))
    W_ad = train_adapter(x_cal, a, Q, rng)
    bases = {
        "true": (R[:, :DC], R[:, DC:]),
        "spec": (V_spec, Q[:, DC:]),
        "adapt": (W_ad.T[:, :DC], W_ad.T[:, DC:]),
    }
    eig = {k: E20.block_eig(x_cal, *v, a) for k, v in bases.items()}
    err = {k: subspace_error(bases[k][0], R[:, :DC]) for k in bases}
    return R, x_cal, bases, eig, err


# ----------------------------------------------------------------------
#  Full-covariance priors in the permutation-symmetric form
# ----------------------------------------------------------------------
def stacked_moments(x_cal):
    """S0 and Sc exactly as in E20.fullcov_prior."""
    n, u_, d = x_cal.shape
    x0 = x_cal.reshape(-1, d)
    s0 = x0.T @ x0 / len(x0)
    tot = x_cal.sum(axis=1)
    sc = (tot.T @ tot - x0.T @ x0) / (u_ * (u_ - 1) * n)
    return s0, (sc + sc.T) / 2


def shrinkage_intensities(x_cal):
    xs = x_cal.reshape(len(x_cal), -1)
    lw = float(ledoit_wolf_shrinkage(xs, assume_centered=True))
    _, oa = oas(xs, assume_centered=True)
    return lw, float(oa)


def fc_blocks(s0, sc, delta=0.0, proj=None):
    """(mu, E, Delta, n_clipped) of the clipped, optionally shrunk and
    projected prior Sig = I (x) Bbar + (J/U) (x) (Abar - Bbar).

    Shrinking the stacked prior to (1 - delta) Sig + delta mu I maps
    S0 -> (1 - delta) S0 + delta mu I and Sc -> (1 - delta) Sc. The PSD
    clip of E20 acts on the eigenvalues of Sig, which are those of
    S0 + (U-1) Sc (on the all-ones user direction) and of S0 - Sc
    (multiplicity U - 1), so clipping the two d x d blocks is the same."""
    d = s0.shape[0]
    if proj is not None:
        P = proj @ proj.T
        sc = P @ sc @ P
    mu0 = np.trace(s0) / d
    s0 = (1 - delta) * s0 + delta * mu0 * np.eye(d)
    sc = (1 - delta) * sc
    wa, qa = np.linalg.eigh(s0 + (U - 1) * sc)
    wb, qb = np.linalg.eigh(s0 - sc)
    n_clip = int((wa < 0).sum() + (U - 1) * (wb < 0).sum())
    abar = (qa * np.clip(wa, 0, None)) @ qa.T
    mu = np.clip(wb, 0, None)
    Delta = qb.T @ abar @ qb - np.diag(mu)
    return mu, qb, (Delta + Delta.T) / 2, n_clip


def fc_dense(blocks):
    """The U d x U d prior of fc_blocks, for the check against E20."""
    mu, Eb, Delta, _ = blocks
    bbar = (Eb * mu) @ Eb.T
    dlt = Eb @ Delta @ Eb.T
    return (np.kron(np.eye(U), bbar) +
            np.kron(np.ones((U, U)) / U, dlt))


def fc_struct_t(tl, h, B, rho, blocks, dev):
    """Exact LMMSE of the stacked model with a symmetric prior (torch).

    tl: (nb, L, U, d) front-end outputs of nb fading realizations with L
    samples each; h: (nb, U). Same estimate as E20.demux_fullcov, with one
    d x d LU per realization in place of a U d x U d Cholesky.

    In the eigenbasis E of Bbar, Sig = I (x) diag(mu) + (w w^T) (x) Delta
    with w = 1/sqrt(U). The noise of y = Gamma^-1 tilde is C (x) I with
    C = Q diag(lam) Q^T, so K = I (x) diag(mu) + C (x) I is a U x U matrix
    per coordinate and (K + W Delta W^T)^-1 follows from Woodbury with
    W = w (x) I and G = W^T K^-1 W diagonal."""
    mu, Eb, Delta = (torch.as_tensor(v, dtype=torch.float64, device=dev)
                     for v in blocks[:3])
    nb, L, u_, d = tl.shape
    s2 = 1.0 / rho
    Bt = torch.as_tensor(B, dtype=torch.float64, device=dev)
    binv = _inv(Bt)
    ginv = (1 / h)[:, :, None] * binv[None] * h[:, None, :]
    cb = s2 * (1 / h)[:, :, None] * binv[None] * (1 / h)[:, None, :]
    y = torch.einsum('nuv,nlvd->nlud', ginv, tl) @ Eb
    lam, Q = torch.linalg.eigh(cb)
    inv = 1.0 / (mu[None, None, :] + lam[:, :, None])          # (nb, U, d)
    Qt = Q.transpose(-1, -2)

    def kinv(v):
        return Q[:, None] @ ((Qt[:, None] @ v) * inv[:, None])

    ky = kinv(y)
    sw = math.sqrt(u_)
    s = ky.sum(2) / sw                                          # (nb, L, d)
    qw = Qt.sum(-1) / sw
    g = ((qw ** 2)[:, :, None] * inv).sum(1)                    # (nb, d)
    sg = torch.sqrt(g)
    Mi = (torch.eye(d, dtype=torch.float64, device=dev)[None] +
          sg[:, :, None] * Delta[None] * sg[:, None, :])
    rhs = (sg[:, None, :] * (s @ Delta)).transpose(1, 2)        # (nb, d, L)
    t = _solve(Mi, rhs).transpose(1, 2) / sg[:, None, :]
    z = ky - kinv(t[:, :, None, :].expand(nb, L, u_, d) / sw)
    xh = mu * z + ((z.sum(2) @ Delta) / u_)[:, :, None, :]
    return xh @ Eb.T


def demux_fc_struct(tilde, B, h, rho, blocks, chunk=250, dev=DEV):
    out = np.empty_like(tilde)
    for s in range(0, len(tilde), chunk):
        tl = torch.as_tensor(tilde[s:s + chunk, None], dtype=torch.float64,
                             device=dev)
        hb = torch.as_tensor(h[s:s + chunk], dtype=torch.float64, device=dev)
        out[s:s + chunk] = fc_struct_t(tl, hb, B, rho, blocks,
                                       dev)[:, 0].cpu().numpy()
    return out


# ----------------------------------------------------------------------
#  Scoring as in E20
# ----------------------------------------------------------------------
class Scorer:
    def __init__(self, pool, R, idx):
        self.R = R
        self.bank_s = pool.pca_coords(E.IDX_EVAL, DC)
        self.bank_p = pool.pca_coords(E.IDX_EVAL, D - DC)
        self.pos = {g: i for i, g in enumerate(E.IDX_EVAL)}
        self.idx = idx
        self.ts = np.array([self.pos[g] for g in idx[:, 0]])

    def __call__(self, e, x):
        cos, nmse, ser = metrics(e, x)
        zh = e @ self.R
        hs, hp = [], []
        for u in range(U):
            tp = np.array([self.pos[g] for g in self.idx[:, u + 1]])
            hs.append(E.top1(zh[:, u, :DC], self.bank_s, self.ts))
            hp.append(E.top1(zh[:, u, DC:], self.bank_p, tp))
        n = x.shape[0] * x.shape[1]
        return dict(cos=cos, nmse=nmse, ser=ser,
                    ser_se=math.sqrt(ser * (1 - ser) / n),
                    shared_top1=float(np.mean(np.concatenate(hs))),
                    private_top1=float(np.mean(np.concatenate(hp))))


def eval_tuples(pool, a, B, snr, R):
    rho = 10 ** (snr / 10)
    rng_e = np.random.default_rng(909)
    z, idx = E.sample_with_index(pool, E.BATCH_EVAL, a, rng_e, E.IDX_EVAL)
    x = z @ R.T
    tilde, h = matched_filter(x, B, rho, rng_e)
    return rho, x, tilde, h, idx


def fmt(tag, name, r, extra=""):
    return (f"{tag} {name:28s} cos={r['cos']:.4f} nmse={r['nmse']:.4f} "
            f"ser={r['ser']:.4f}+-{r['ser_se']:.4f} "
            f"shared_top1={r['shared_top1']:.4f} "
            f"private_top1={r['private_top1']:.4f}{extra}")


def write_csv(path, key_cols, rows, cols):
    with open(path, "w") as f:
        f.write(",".join(key_cols + cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in key_cols + cols) + "\n")


def fc_variants(x_cal, bases, lw, oa, grid=True):
    s0, sc = stacked_moments(x_cal)
    v = {"LMMSE-fullcov": (fc_blocks(s0, sc), 0.0),
         "LMMSE-fullcov-LW": (fc_blocks(s0, sc, lw), lw),
         "LMMSE-fullcov-OAS": (fc_blocks(s0, sc, oa), oa)}
    for k in ("spec", "adapt", "true"):
        v[f"LMMSE-fullcov-proj-{k}"] = (
            fc_blocks(s0, sc, proj=bases[k][0]), 0.0)
    v["LMMSE-fullcov-LW-proj-adapt"] = (
        fc_blocks(s0, sc, lw, proj=bases["adapt"][0]), lw)
    if grid:
        for dl in SHR_GRID:
            v[f"LMMSE-fullcov-shr{dl:.2f}"] = (fc_blocks(s0, sc, dl), dl)
    return v


# ----------------------------------------------------------------------
#  (a), (b): variants at 10, 16, 20, 30 dB on the E20 calibration
# ----------------------------------------------------------------------
def part_main():
    pool = load_pool()
    a = math.sqrt(BETA) * np.ones(U)
    B = affinity_matrix(a)
    R, x_cal, bases, eig, err = calibrate(pool, a, E.N_PAIR)
    lw, oa = shrinkage_intensities(x_cal)
    fcv = fc_variants(x_cal, bases, lw, oa)
    sig_e20 = E20.fullcov_prior(x_cal)
    notes = [f"[main] N={E.N_PAIR} subspace error spec={err['spec']:.4f} "
             f"adapt={err['adapt']:.4f}; Ledoit-Wolf delta={lw:.4f}, "
             f"OAS delta={oa:.4f} (stacked U d={U*D}, n={E.N_PAIR})",
             f"[main] PSD-clipped eigenvalues of the raw stacked prior: "
             f"{fcv['LMMSE-fullcov'][0][3]} of {U*D}; after LW: "
             f"{fcv['LMMSE-fullcov-LW'][0][3]}; proj-spec: "
             f"{fcv['LMMSE-fullcov-proj-spec'][0][3]}; proj-adapt: "
             f"{fcv['LMMSE-fullcov-proj-adapt'][0][3]}",
             "[main] check of the symmetric prior against "
             "E20.fullcov_prior: max abs diff "
             f"{np.abs(fc_dense(fcv['LMMSE-fullcov'][0]) - sig_e20).max():.2e}"]
    print("\n".join(notes), flush=True)
    rows = []
    for snr in SNRS:
        t0 = time.time()
        rho, x, tilde, h, idx = eval_tuples(pool, a, B, snr, R)
        sc_ = Scorer(pool, R, idx)
        est = {"LMMSE": demux_lmmse(tilde, B, h, rho)[0]}
        for k, (Vs, _) in bases.items():
            est[f"DR-{k}"] = demux_dr(tilde, B, h, rho, a, Vs)
            est[f"DR-cov-{k}"] = E20.demux_cov(tilde, B, h, rho, a, *eig[k])
        shr = {}
        for name, (blk, dl) in fcv.items():
            est[name] = demux_fc_struct(tilde, B, h, rho, blk)
            shr[name] = dl
        ref = E20.demux_fullcov(tilde[:16], B, h[:16], rho, sig_e20)
        dmax = np.abs(ref - est["LMMSE-fullcov"][:16]).max()
        notes.append(f"[main] snr={snr} structured vs E20 Ud x Ud Cholesky "
                     f"on 16 tuples: max abs diff {dmax:.2e} "
                     f"(max |x_hat| {np.abs(ref).max():.3f})")
        print(notes[-1], flush=True)
        for name, e in est.items():
            r = sc_(e, x)
            r.update(snr=snr, scheme=name, shrinkage=shr.get(name, ""))
            rows.append(r)
            notes.append(fmt(f"snr={snr:2d}", name, r))
            print(notes[-1], flush=True)
        print(f"  snr {snr} took {time.time()-t0:.1f} s", flush=True)
    write_csv(os.path.join(DATA, "e20b_fullcov_variants.csv"),
              ["snr", "scheme"], rows, list(COLS) + ["shrinkage"])
    with open(os.path.join(LOGS, "e20b_notes_main.txt"), "w") as f:
        f.write("\n".join(notes) + "\n")


# ----------------------------------------------------------------------
#  (c): calibration-size sweep at 20 dB
# ----------------------------------------------------------------------
def part_sweep(snr=20):
    pool = load_pool()
    a = math.sqrt(BETA) * np.ones(U)
    B = affinity_matrix(a)
    rows, notes = [], []
    R0 = None
    for n_pair in N_SWEEP:
        t0 = time.time()
        R, x_cal, bases, eig, err = calibrate(pool, a, n_pair)
        if R0 is None:
            R0 = R
            rho, x, tilde, h, idx = eval_tuples(pool, a, B, snr, R)
            sc_ = Scorer(pool, R, idx)
            r_lm = sc_(demux_lmmse(tilde, B, h, rho)[0], x)
            notes.append(fmt("[sweep] N=  any", "LMMSE", r_lm))
        assert np.array_equal(R, R0)
        lw, oa = shrinkage_intensities(x_cal)
        notes.append(f"[sweep] N={n_pair} subspace error spec="
                     f"{err['spec']:.4f} adapt={err['adapt']:.4f} "
                     f"LW delta={lw:.4f} OAS delta={oa:.4f}")
        print(notes[-1], flush=True)
        est, shr = {}, {}
        for k in ("spec", "adapt"):
            est[f"DR-{k}"] = demux_dr(tilde, B, h, rho, a, bases[k][0])
        for k in ("spec", "adapt", "true"):
            est[f"DR-cov-{k}"] = E20.demux_cov(tilde, B, h, rho, a, *eig[k])
        for name, (blk, dl) in fc_variants(x_cal, bases, lw, oa,
                                           grid=False).items():
            est[name] = demux_fc_struct(tilde, B, h, rho, blk)
            shr[name] = dl
        for name, e in est.items():
            r = sc_(e, x)
            k = name.split("-")[-1]
            r.update(n_pair=n_pair, scheme=name, shrinkage=shr.get(name, ""),
                     subspace_err=(err[k] if name.startswith("DR-") and
                                   k in ("spec", "adapt") else ""))
            rows.append(r)
            notes.append(fmt(f"[sweep] N={n_pair:5d}", name, r))
            print(notes[-1], flush=True)
        print(f"  N {n_pair} took {time.time()-t0:.1f} s", flush=True)
    rows.insert(0, dict(n_pair="any", scheme="LMMSE", shrinkage="",
                        subspace_err="", **r_lm))
    write_csv(os.path.join(DATA, "e20b_calib_sweep.csv"),
              ["n_pair", "scheme"], rows,
              list(COLS) + ["shrinkage", "subspace_err"])
    with open(os.path.join(LOGS, "e20b_notes_sweep.txt"), "w") as f:
        f.write("\n".join(notes) + "\n")


# ----------------------------------------------------------------------
#  (d): cost
# ----------------------------------------------------------------------
def dr_t(tl, h, B, rho, a, Vc, dev):
    """Torch port of semantic_mac.demux_dr (same estimate)."""
    nb, L, u_, d = tl.shape
    dc = Vc.shape[1]
    s2 = 1.0 / rho
    Bt = torch.as_tensor(B, dtype=torch.float64, device=dev)
    at = torch.as_tensor(a, dtype=torch.float64, device=dev)
    Vt = torch.as_tensor(Vc, dtype=torch.float64, device=dev)
    gam = (1 / h)[:, :, None] * Bt[None] * h[:, None, :]
    Cn = s2 * (1 / h)[:, :, None] * Bt[None] * (1 / h)[:, None, :]
    gvec = gam @ at
    Ci = _inv(Cn)
    cg = (Ci @ gvec[..., None])[..., 0]
    den = (gvec * cg).sum(-1)
    wv = cg / den[:, None] * ((1 / dc) / (1 / dc + 1 / den))[:, None]
    Zs = tl @ Vt
    c_hat = torch.einsum('nu,nluk->nlk', wv, Zs)
    Zp = tl - Zs @ Vt.T
    Gp = _inv(gam)[:, None] @ Zp
    sig_p = 1 - at ** 2
    err_p = (d - dc) * s2 * _inv(Bt).diagonal()[None] / h ** 2
    shp = sig_p[None] / (sig_p[None] + err_p)
    return (at[None, None, :, None] * (c_hat @ Vt.T)[:, :, None, :] +
            shp[:, None, :, None] * Gp)


def drcov_t(tl, h, B, rho, a, Rb, var_s, var_p, dev):
    """Torch port of exp20_covariance_receiver.demux_cov (same estimate)."""
    s2 = 1.0 / rho
    f64 = dict(dtype=torch.float64, device=dev)
    Bt = torch.as_tensor(B, **f64)
    at = torch.as_tensor(a, **f64)
    Rt = torch.as_tensor(Rb, **f64)
    t = tl @ Rt
    G = (1 / h)[:, :, None] * Bt[None] * h[:, None, :]
    Cn = s2 * (1 / h)[:, :, None] * Bt[None] * (1 / h)[:, None, :]
    Gt = G.transpose(-1, -2)[:, None]
    dc = len(var_s)
    out = torch.empty_like(t)
    for P, var, sl in ((torch.outer(at, at), var_s, slice(0, dc)),
                       (torch.diag(1 - at ** 2), var_p, slice(dc, None))):
        Cx = P[None, None] * torch.as_tensor(var, **f64)[None, :, None, None]
        S = G[:, None] @ Cx @ Gt + Cn[:, None]
        F = _solve(S.transpose(-1, -2),
                               (Cx @ Gt).transpose(-1, -2)).transpose(-1, -2)
        out[..., sl] = torch.einsum('nkuv,nlvk->nluk', F, t[..., sl])
    return out @ Rt.T


def fc_naive_t(tl, h, B, rho, sig, dev):
    """The U d x U d Cholesky of E20.demux_fullcov with a device argument
    and block length L (one factorization per realization)."""
    nb, L, u_, d = tl.shape
    s2 = 1.0 / rho
    Bt = torch.as_tensor(B, dtype=torch.float64, device=dev)
    binv = _inv(Bt)
    ginv = (1 / h)[:, :, None] * binv[None] * h[:, None, :]
    cb = s2 * (1 / h)[:, :, None] * binv[None] * (1 / h)[:, None, :]
    y = torch.einsum('nuv,nlvd->nlud', ginv, tl).reshape(nb, L, -1)
    eye = torch.eye(d, dtype=torch.float64, device=dev)
    out = []
    for i in range(nb):
        M = sig + (cb[i][:, None, :, None] * eye[None, :, None, :]
                   ).reshape(u_ * d, u_ * d)
        Lc = torch.linalg.cholesky(M)
        z = torch.cholesky_solve(y[i].T, Lc)
        out.append((sig @ z).T)
    return torch.stack(out).reshape(nb, L, u_, d)


def mult_counts(d=D, u_=U, dc=DC):
    """Leading multiply counts (real multiplications) of the
    implementations above, excluding the matched-filter front end that all
    receivers share. Per block: the work that depends on the fading
    realization only. Per sample: the work for each embedding tuple."""
    lu = 2 * u_ ** 3 / 3                    # U x U LU
    return {
        "DR": (3 * u_ ** 3 + 2 * u_ ** 2,
               2 * u_ * d * dc + d * dc + u_ ** 2 * d + u_ * d),
        "DR-cov": (d * (3 * u_ ** 3 + lu + u_ ** 3),
                   2 * u_ * d * d + u_ ** 2 * d),
        "LMMSE-fullcov (U d x U d Cholesky)":
            ((u_ * d) ** 3 / 3 + u_ ** 2 * d * u_,
             2 * (u_ * d) ** 2 + (u_ * d) ** 2 + u_ ** 2 * d),
        "LMMSE-fullcov (symmetric reduction)":
            (2 * d ** 3 / 3 + d ** 2 + 2 * u_ ** 2 * d + u_ ** 3,
             2 * u_ * d * d + 3 * d ** 2 + 4 * u_ ** 2 * d),
    }


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def part_cost():
    pool = load_pool()
    a = math.sqrt(BETA) * np.ones(U)
    B = affinity_matrix(a)
    R, x_cal, bases, eig, _ = calibrate(pool, a, E.N_PAIR)
    blk = fc_blocks(*stacked_moments(x_cal))
    sig = E20.fullcov_prior(x_cal)
    Vs = bases["spec"][0]
    Rb, vs, vp = eig["spec"]
    rng = np.random.default_rng(2024)
    rho = 10 ** (20 / 10)
    z, _ = E.sample_with_index(pool, 1000, a, rng, E.IDX_EVAL)
    x = z @ R.T
    notes, rows = [], []

    # equivalence of the torch ports on 50 tuples with their own fading
    tilde, h = matched_filter(x[:50], B, rho, rng)
    tl = torch.as_tensor(tilde[:, None], dtype=torch.float64)
    ht = torch.as_tensor(h, dtype=torch.float64)
    cpu = torch.device("cpu")
    chk = {
        "DR": (dr_t(tl, ht, B, rho, a, Vs, cpu)[:, 0].numpy(),
               demux_dr(tilde, B, h, rho, a, Vs)),
        "DR-cov": (drcov_t(tl, ht, B, rho, a, Rb, vs, vp, cpu)[:, 0].numpy(),
                   E20.demux_cov(tilde, B, h, rho, a, Rb, vs, vp)),
        "LMMSE-fullcov sym": (fc_struct_t(tl, ht, B, rho, blk, cpu)[:, 0]
                              .numpy(),
                              E20.demux_fullcov(tilde[:4], B, h[:4], rho,
                                                sig)),
    }
    for k, (p, q) in chk.items():
        n = len(q)
        notes.append(f"[cost] torch port check {k}: max abs diff "
                     f"{np.abs(p[:n] - q).max():.2e} on {n} tuples")
        print(notes[-1], flush=True)

    mc = mult_counts()
    impl = {
        "DR": lambda tl, h, dev: dr_t(tl, h, B, rho, a, Vs, dev),
        "DR-cov": lambda tl, h, dev: drcov_t(tl, h, B, rho, a, Rb, vs, vp,
                                             dev),
        "LMMSE-fullcov (U d x U d Cholesky)":
            lambda tl, h, dev: fc_naive_t(tl, h, B, rho, sig_t[dev.type],
                                          dev),
        "LMMSE-fullcov (symmetric reduction)":
            lambda tl, h, dev: fc_struct_t(tl, h, B, rho, blk, dev),
    }
    devs = [cpu] + ([torch.device("cuda")] if torch.cuda.is_available()
                    else [])
    sig_t = {dv.type: torch.as_tensor(sig, dtype=torch.float64, device=dv)
             for dv in devs}
    hc = (rng.standard_normal((1000, U)) +
          1j * rng.standard_normal((1000, U))) / math.sqrt(2)
    hh = np.clip(np.abs(hc), 0.2, None)
    for name, fn in impl.items():
        pb, ps = mc[name]
        for dev in devs:
            for L in (1, 1000):
                nb_full = 1000 // L
                # samples actually timed (scaled to 1,000 when fewer)
                if name.startswith("LMMSE-fullcov (U d") and L == 1:
                    n_meas = 20 if dev.type == "cpu" else 100
                elif name.startswith("LMMSE-fullcov (sym") and L == 1:
                    n_meas = 200 if dev.type == "cpu" else 1000
                else:
                    n_meas = 1000
                nb = max(1, n_meas // L)
                tl = torch.as_tensor(
                    x[:nb * L].reshape(nb, L, U, D), dtype=torch.float64,
                    device=dev)
                ht = torch.as_tensor(hh[:nb], dtype=torch.float64,
                                     device=dev)
                chunk = 50 if (dev.type == "cpu" and L == 1) else 250
                if name.startswith("LMMSE-fullcov (U d"):
                    chunk = nb

                def run():
                    for s in range(0, nb, chunk):
                        fn(tl[s:s + chunk], ht[s:s + chunk], dev)
                    _sync(dev)
                run()                                   # warm-up
                ts = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    run()
                    ts.append(time.perf_counter() - t0)
                sec = float(np.median(ts)) * 1000 / (nb * L)
                tot = nb_full * pb + 1000 * ps
                rows.append(dict(
                    scheme=name, mults_per_block=f"{pb:.4g}",
                    mults_per_sample=f"{ps:.4g}",
                    mults_per_1000=f"{tot:.4g}",
                    device=("cpu-4threads" if dev.type == "cpu"
                            else torch.cuda.get_device_name(0)),
                    block_len=L, n_measured=nb * L,
                    seconds_per_1000=f"{sec:.4g}"))
                notes.append(f"[cost] {name:36s} {rows[-1]['device']:30s} "
                             f"L={L:4d} mults/block={pb:.3g} "
                             f"mults/sample={ps:.3g} "
                             f"mults/1000={tot:.3g} "
                             f"s/1000={sec:.4g} (timed {nb*L})")
                print(notes[-1], flush=True)
    write_csv(os.path.join(DATA, "e20b_cost.csv"), ["scheme", "device"],
              rows, ["block_len", "mults_per_block", "mults_per_sample",
                     "mults_per_1000", "n_measured", "seconds_per_1000"])
    with open(os.path.join(LOGS, "e20b_notes_cost.txt"), "w") as f:
        f.write("\n".join(notes) + "\n")


def part_notes():
    head = ["E20b notes (exp20b_fullcov_variants.py). Columns as in E20; "
            "ser +- is the binomial standard error over 16,000 user "
            "samples. Shrinkage intensities: Ledoit-Wolf and OAS from "
            "sklearn.covariance (assume_centered=True) on the N stacked "
            "U d vectors, applied as (1-delta) Sig + delta (1/d) I. "
            "shr<delta> rows use a fixed delta and the best of them is "
            "chosen in hindsight on the test tuples."]
    import csv
    rd = lambda fn: list(csv.DictReader(open(os.path.join(DATA, fn))))
    e20 = {(r["snr"], r["scheme"]): r for r in rd("e20_covariance_receiver.csv")}
    var = {(r["snr"], r["scheme"]): r for r in rd("e20b_fullcov_variants.csv")}
    sw = {(r["n_pair"], r["scheme"]): r for r in rd("e20b_calib_sweep.csv")}
    m = ("cos", "nmse", "ser", "shared_top1", "private_top1")
    d1 = max(abs(float(e20[k][c]) - float(var[k][c]))
             for k in e20 if k in var for c in m)
    n1 = sum(k in var for k in e20)
    d2 = max(abs(float(var[("20", k[1])][c]) - float(sw[k][c]))
             for k in sw if k[0] == "200" for c in m)
    head.append(f"[check] {n1} E20 rows (4 SNRs x 8 receivers, every E20 "
                f"receiver except DR-MMSE-true, which E20b does not rerun) "
                f"reproduced, max abs diff {d1:.1e} over {m}; sweep N=200 "
                f"vs the 20 dB rows of the variants run: max abs diff {d2:.1e}")
    for snr in ("10", "16", "20", "30"):
        g = min((float(var[(snr, f"LMMSE-fullcov-shr{x:.2f}")]["ser"]), x)
                for x in SHR_GRID)
        head.append(f"[summary] snr={snr} SER DR-cov-adapt="
                    f"{float(var[(snr, 'DR-cov-adapt')]['ser']):.4f} "
                    f"fullcov raw={float(var[(snr, 'LMMSE-fullcov')]['ser']):.4f} "
                    f"LW={float(var[(snr, 'LMMSE-fullcov-LW')]['ser']):.4f} "
                    f"best grid={g[0]:.4f} (delta={g[1]}) "
                    f"proj-adapt={float(var[(snr, 'LMMSE-fullcov-proj-adapt')]['ser']):.4f} "
                    f"LW-proj-adapt="
                    f"{float(var[(snr, 'LMMSE-fullcov-LW-proj-adapt')]['ser']):.4f}")
    for p in ("main", "sweep", "cost"):
        fp = os.path.join(LOGS, f"e20b_notes_{p}.txt")
        if os.path.exists(fp):
            with open(fp) as f:
                head.append(f.read().rstrip())
    with open(os.path.join(DATA, "e20b_notes.txt"), "w") as f:
        f.write("\n".join(head) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="main",
                    choices=["main", "sweep", "cost", "notes"])
    p = ap.parse_args().part
    {"main": part_main, "sweep": part_sweep, "cost": part_cost,
     "notes": part_notes}[p]()
