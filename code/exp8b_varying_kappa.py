"""E8b: partial sharing that varies from sample to sample and is unknown
to the receiver (Reviewer 1.3, follow-up to E8).

Reviewer 1 observed that a shared subspace fixed at calibration does not
guarantee that its content is common on every online sample. E8
(exp8_partial_sharing.py) answers this with one contamination level kappa
for the whole batch and gives the masks and the receivers the exact
reduced affinity a_u a_v (1-kappa). This script removes both
simplifications. The shared block of user u on sample b is

    a_u ( sqrt(1-kappa_b) c_b + sqrt(kappa_b) r_{u,b} ),

as in E8, but kappa_b is now drawn per sample from a law with mean
E[kappa], and the transmitter masks and every receiver are built from one
fixed affinity that ignores kappa_b. Two such choices are tested.

  setting "mean"   masks and receivers at a^2 (1 - E[kappa]), the value a
                   long-run affinity estimate would converge to;
  setting "calib"  masks and receivers at the calibration value a^2, that
                   is, kappa assumed zero (fully mismatched).

Because a setting fixes both the mask Gram (a transmitter choice) and the
receiver prior, two cross settings separate the two effects:
  setting "mMrC"   masks at the mean value, receivers at the calibration value;
  setting "mCrM"   masks at the calibration value, receivers at the mean value.
In every setting the receivers know the mask Gram exactly (it is their own
signature design); only the content prior a_u can be wrong. In the cross
settings the LMMSE prior Cx = B(a_rx)/d differs from the mask Gram, so it
uses the local function demux_lmmse_prior (same algebra as
semantic_mac.lmmse_matrices with a separate prior).

Kappa laws (per sample, shared by the U users of that sample)
  fixed0    kappa = 0                        (no contamination, context)
  fixed01   kappa = 0.1                      (matched reference for mix01)
  fixed03   kappa = 0.3                      (E8 point, reference)
  unif06    kappa ~ Uniform[0, 0.6]          (mean 0.3)
  mix03     kappa = 1 w.p. 0.3, else 0       (mean 0.3)
  mix01     kappa = 1 w.p. 0.1, else 0       (mean 0.1, extra law)
The law fixed03 under setting "mean" is exactly the matched E8 point.

Setting: U=4, d=64, d_c=16 (Table IV), rho = 10 dB, Rayleigh fading with
the H_MIN clip, beta = a^2 in {0.5, 0.8}, batch 4000, 5 seeds. Receivers:
OMA and genie NOMA-SIC (unaffected by kappa), SR, SC, LMMSE (unstructured),
DR-MMSE (demux_block_lmmse of exp6, Remark 1) and DR. "Best unstructured"
is the smallest SER among SR, SC and LMMSE, as in E8.

Seeds. Seed s in 0..4 and beta index i give the latent stream
82000+100s+10i, the channel stream 83000+100s+10i, the kappa stream
84000+100s+10i and the OMA/NOMA stream 85000+100s+10i. Every law and
setting of one (s, beta) reuses the same latent, channel and noise draws
(common random numbers), and the mixture and uniform laws share one
uniform draw per sample, so the comparisons are paired.

Reproduction check. Before the new runs the script replays the E8 loop
(rng seed 11, the E8 kappa grid, both betas) through its own per-sample
code path with a constant kappa and compares every SER, cosine and NMSE
against data/e8_partial_sharing.csv.

Outputs: data/e8b_varying_kappa.csv (one row per seed, beta, law,
setting), data/e8b_varying_kappa_summary.csv (means, standard errors over
seeds, ratios), data/e8b_notes.txt (reproduction check and key lines).
"""
import csv
import math
import os
import sys
from multiprocessing import Pool

import numpy as np

from semantic_mac import (affinity_matrix, matched_filter, demux_sr,
                          demux_sc, demux_lmmse, demux_dr, metrics,
                          oma_observe, demux_noma_genie, TAU)
from exp6_sideinfo import demux_block_lmmse
from semantic_mac import U_MAIN as U, D_SYN as D, DC_SYN as DC  # Table IV

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
os.makedirs(DATA, exist_ok=True)

BATCH = 4000                      # as in E8
RHO_DB = 10                       # E8 operating point
BETAS = (0.5, 0.8)
SEEDS = range(5)
NAMES = ["OMA", "NOMA", "SR", "SC", "LMMSE", "BLMMSE", "DR"]
SUBSET_NAMES = ["LMMSE", "BLMMSE", "DR"]
LAWS = [("fixed0", 0.0), ("fixed01", 0.1), ("fixed03", 0.3),
        ("unif06", 0.3), ("mix03", 0.3), ("mix01", 0.1)]
SETTINGS = ("mean", "calib", "mMrC", "mCrM")
# (mask uses mean value?, receiver uses mean value?)
SETTING_SCALES = {"mean": (True, True), "calib": (False, False),
                  "mMrC": (True, False), "mCrM": (False, True)}
E8_KAPPAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]   # E8 grid, in order


def draw_kappa(law, u):
    """Per-sample kappa from one uniform draw u (shape (batch,))."""
    if law == "fixed0":
        return np.zeros_like(u)
    if law == "fixed01":
        return np.full_like(u, 0.1)
    if law == "fixed03":
        return np.full_like(u, 0.3)
    if law == "unif06":
        return 0.6 * u
    if law == "mix03":
        return (u < 0.3).astype(float)
    if law == "mix01":
        return (u < 0.1).astype(float)
    raise ValueError(law)


def sample_latents_varying(batch, U, d, d_c, a, kappa, rng):
    """E8 generator with a per-sample kappa (shape (batch,)).

    Same draw order as exp8_partial_sharing.sample_latents_partial
    (c, then r, then p), so a constant kappa reproduces it bit for bit."""
    a = np.broadcast_to(np.asarray(a, float), (U,))
    kappa = np.asarray(kappa, float).reshape(batch, 1, 1)
    z = np.zeros((batch, U, d))
    c = rng.standard_normal((batch, d_c))
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    r = rng.standard_normal((batch, U, d_c))
    r /= np.linalg.norm(r, axis=2, keepdims=True)
    p = rng.standard_normal((batch, U, d - d_c))
    p /= np.linalg.norm(p, axis=2, keepdims=True)
    s = np.sqrt(1.0 - kappa) * c[:, None, :] + np.sqrt(kappa) * r
    s /= np.linalg.norm(s, axis=2, keepdims=True)
    z[:, :, :d_c] = a[None, :, None] * s
    z[:, :, d_c:] = np.sqrt(1.0 - a ** 2)[None, :, None] * p
    return z


def per_sample_error(e_hat, e_true, tau=TAU):
    """Boolean (batch, U) semantic error indicator, as in metrics()."""
    e_n = e_hat / (np.linalg.norm(e_hat, axis=2, keepdims=True) + 1e-12)
    return (e_n * e_true).sum(-1) < tau


def demux_lmmse_prior(tilde, B, h, rho, Cprior):
    """LMMSE of the form (F (x) I_d) with channel Gram B and a separate
    content prior Cprior/d (semantic_mac.lmmse_matrices uses Cprior = B)."""
    batch, U_, d = tilde.shape
    sigma2 = 1.0 / rho
    out = np.empty_like(tilde)
    Cx = Cprior / d
    for b in range(batch):
        H = np.diag(h[b])
        Hi = np.diag(1.0 / h[b])
        Gamma = Hi @ B @ H
        Cn = sigma2 * (Hi @ B @ Hi)
        S = Gamma @ Cx @ Gamma.T + Cn
        W = np.linalg.solve(S.T, (Cx @ Gamma.T).T).T
        out[b] = W @ tilde[b]
    return out


def run_config(beta, kappa, a_rx_scale, rho, rng_lat, rng_ch, rng_conv,
               mask_scale=None):
    """One batch. kappa: per-sample (batch,). The masks (signature Gram)
    use a_mask = sqrt(beta) * mask_scale (default a_rx_scale) and every
    receiver uses the content prior a_rx = sqrt(beta) * a_rx_scale for all
    samples. Returns ({name: (cos, nmse, ser)}, {name: err (batch, U)})."""
    if mask_scale is None:
        mask_scale = a_rx_scale
    a = math.sqrt(beta) * np.ones(U)
    z = sample_latents_varying(BATCH, U, D, DC, a, kappa, rng_lat)
    a_rx = a * a_rx_scale
    B = affinity_matrix(a * mask_scale)
    tilde, h = matched_filter(z, B, rho, rng_ch)
    Vc = np.eye(D)[:, :DC]
    est = {}
    est["SR"] = demux_sr(tilde, B, h)
    est["SC"] = demux_sc(tilde, h)
    if mask_scale == a_rx_scale:
        est["LMMSE"], _ = demux_lmmse(tilde, B, h, rho)
    else:
        est["LMMSE"] = demux_lmmse_prior(tilde, B, h, rho, affinity_matrix(a_rx))
    est["BLMMSE"] = demux_block_lmmse(tilde, B, h, rho, a_rx, Vc)
    est["DR"] = demux_dr(tilde, B, h, rho, a_rx, Vc)
    est["OMA"] = oma_observe(z, h, rho, rng_conv)
    est["NOMA"] = demux_noma_genie(z, h, rho, rng_conv)
    res = {n: metrics(est[n], z) for n in NAMES}
    err = {n: per_sample_error(est[n], z) for n in SUBSET_NAMES}
    return res, err


def reproduce_e8():
    """Replay E8 (rng 11, same order) through run_config, constant kappa,
    receivers at the exact reduced affinity. Returns rows in E8 order."""
    rng = np.random.default_rng(11)
    rho = 10 ** (10 / 10)
    rows = []
    for beta in (0.5, 0.8):
        for k in E8_KAPPAS:
            conv = np.random.default_rng(80000 + int(1000 * beta + 100 * k))
            res, _ = run_config(beta, np.full(BATCH, k), math.sqrt(1.0 - k),
                                rho, rng, rng, conv)
            rows.append((beta, k, res))
    return rows


def run_seed_beta(args):
    s, ib = args
    beta = BETAS[ib]
    rho = 10 ** (RHO_DB / 10)
    tag = 100 * s + 10 * ib
    u = np.random.default_rng(84000 + tag).random(BATCH)
    out = []
    for law, mk in LAWS:
        kappa = draw_kappa(law, u)
        for setting in SETTINGS:
            m_mean, r_mean = SETTING_SCALES[setting]
            scale = math.sqrt(1.0 - mk) if r_mean else 1.0
            mscale = math.sqrt(1.0 - mk) if m_mean else 1.0
            res, err = run_config(beta, kappa, scale, rho,
                                  np.random.default_rng(82000 + tag),
                                  np.random.default_rng(83000 + tag),
                                  np.random.default_rng(85000 + tag),
                                  mask_scale=mscale)
            lo = kappa <= mk + 1e-12           # mixture: the kappa=0 samples
            hi = ~lo                           # mixture: the kappa=1 samples
            row = dict(seed=s, beta=beta, law=law, mean_kappa=mk,
                       emp_mean_kappa=float(kappa.mean()),
                       frac_hi=float(hi.mean()), setting=setting,
                       a2_rx=beta * scale ** 2, a2_mask=beta * mscale ** 2)
            for n in NAMES:
                row[f"{n}_cos"], row[f"{n}_nmse"], row[f"{n}_ser"] = res[n]
            for n in SUBSET_NAMES:
                row[f"{n}_ser_lo"] = float(err[n][lo].mean()) if lo.any() else float("nan")
                row[f"{n}_ser_hi"] = float(err[n][hi].mean()) if hi.any() else float("nan")
            row["BestUn_ser"] = min(row["SR_ser"], row["SC_ser"], row["LMMSE_ser"])
            row["BestUn_name"] = min(("SR", "SC", "LMMSE"),
                                     key=lambda n: row[f"{n}_ser"])
            out.append(row)
            print(f"seed={s} beta={beta} {law:8s} {setting:5s} "
                  f"DR={row['DR_ser']:.4f} DRMMSE={row['BLMMSE_ser']:.4f} "
                  f"LMMSE={row['LMMSE_ser']:.4f}", flush=True)
    return out


def task(args):
    if args == "repro":
        return ("repro", reproduce_e8())
    return ("run", run_seed_beta(args))


def mean_se(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))


def main():
    jobs = ["repro"] + [(s, ib) for s in SEEDS for ib in range(len(BETAS))]
    nproc = int(os.environ.get("E8B_PROCS", "4"))
    with Pool(nproc) as pool:
        results = pool.map(task, jobs, chunksize=1)
    repro = [r for t, r in results if t == "repro"][0]
    rows = [row for t, r in results if t == "run" for row in r]

    notes = []
    # ---------------- reproduction check against the published E8 CSV
    with open(os.path.join(DATA, "e8_partial_sharing.csv")) as f:
        pub = list(csv.DictReader(f))
    maxdiff = 0.0
    for (beta, k, res), p in zip(repro, pub):
        assert abs(float(p["beta_nominal"]) - beta) < 1e-12
        assert abs(float(p["kappa"]) - k) < 1e-12
        for n in NAMES:
            for j, m in enumerate(("cos", "nmse", "ser")):
                maxdiff = max(maxdiff, abs(res[n][j] - float(p[f"{n}_{m}"])))
    notes.append("Reproduction check: E8 replayed (rng 11, 16 points) through "
                 "the per-sample code path with constant kappa; max |diff| "
                 f"over all cos/nmse/ser entries vs data/e8_partial_sharing.csv"
                 f" = {maxdiff:.3e}")
    for beta, k, res in repro:
        if abs(k - 0.3) < 1e-12 or k == 0.0:
            notes.append(f"  E8 beta={beta} kappa={k}: DR {res['DR'][2]:.4f}, "
                         f"DR-MMSE {res['BLMMSE'][2]:.4f}, LMMSE "
                         f"{res['LMMSE'][2]:.4f}")

    # ---------------- per-seed CSV
    keys = list(rows[0].keys())
    rows.sort(key=lambda r: (r["beta"], [l for l, _ in LAWS].index(r["law"]),
                             SETTINGS.index(r["setting"]), r["seed"]))
    with open(os.path.join(DATA, "e8b_varying_kappa.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # ---------------- summary over seeds
    summ = []
    for beta in BETAS:
        ref = {}  # matched fixed reference per mean kappa: DR per seed
        for law, mk in LAWS:
            for setting in SETTINGS:
                sel = [r for r in rows if r["beta"] == beta and r["law"] == law
                       and r["setting"] == setting]
                sel.sort(key=lambda r: r["seed"])
                d = dict(beta=beta, law=law, mean_kappa=mk, setting=setting,
                         n_seeds=len(sel))
                for n in ["DR", "BLMMSE", "LMMSE", "SR", "SC", "NOMA",
                          "OMA", "BestUn"]:
                    d[f"{n}_ser_mean"], d[f"{n}_ser_se"] = mean_se(
                        [r[f"{n}_ser"] for r in sel])
                for n in SUBSET_NAMES:
                    for sub in ("lo", "hi"):
                        v = [r[f"{n}_ser_{sub}"] for r in sel]
                        d[f"{n}_ser_{sub}_mean"] = (float(np.mean(v)) if
                                                    not np.isnan(v).any()
                                                    else float("nan"))
                d["BestUn_names"] = "/".join(sorted({r["BestUn_name"] for r in sel}))
                rat = [r["BestUn_ser"] / r["DR_ser"] for r in sel]
                d["ratio_BestUn_over_DR_mean"], d["ratio_BestUn_over_DR_se"] = mean_se(rat)
                d["ratio_of_means_BestUn_over_DR"] = d["BestUn_ser_mean"] / d["DR_ser_mean"]
                rat = [r["BestUn_ser"] / r["BLMMSE_ser"] for r in sel]
                d["ratio_BestUn_over_DRMMSE_mean"], d["ratio_BestUn_over_DRMMSE_se"] = mean_se(rat)
                d["ratio_of_means_BestUn_over_DRMMSE"] = d["BestUn_ser_mean"] / d["BLMMSE_ser_mean"]
                d["DR_wins_all_seeds"] = int(all(r["DR_ser"] < r["BestUn_ser"] for r in sel))
                if law.startswith("fixed") and setting == "mean":
                    ref[mk] = [r["DR_ser"] for r in sel]
                summ.append(d)
        # DR SER relative to the matched fixed reference with the same mean
        for d in summ:
            if d["beta"] != beta:
                continue
            sel = sorted([r for r in rows if r["beta"] == beta and
                          r["law"] == d["law"] and r["setting"] == d["setting"]],
                         key=lambda r: r["seed"])
            rr = [r["DR_ser"] / q for r, q in zip(sel, ref[d["mean_kappa"]])]
            d["DR_over_matched_ref_mean"], d["DR_over_matched_ref_se"] = mean_se(rr)
    skeys = list(summ[0].keys())
    with open(os.path.join(DATA, "e8b_varying_kappa_summary.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=skeys)
        w.writeheader()
        w.writerows(summ)

    # ---------------- notes
    notes.append("")
    notes.append(f"E8b: rho={RHO_DB} dB, U={U}, d={D}, d_c={DC}, batch {BATCH}, "
                 f"{len(SEEDS)} seeds; SER mean +- SE over seeds.")
    notes.append("Columns: law, setting (masks and receivers at a^2(1-E[kappa]) "
                 "= mean, or at a^2 = calib; mMrC masks mean, receivers calib; "
                 "mCrM masks calib, receivers mean), DR, DR-MMSE, LMMSE, best "
                 "unstructured, ratio best-unstructured/DR (mean +- SE of "
                 "per-seed ratios), DR/matched-fixed reference, DR wins on all seeds")
    for d in summ:
        notes.append(
            f"  beta={d['beta']} {d['law']:8s} {d['setting']:5s} "
            f"DR {d['DR_ser_mean']:.4f}+-{d['DR_ser_se']:.4f}  "
            f"DR-MMSE {d['BLMMSE_ser_mean']:.4f}+-{d['BLMMSE_ser_se']:.4f}  "
            f"LMMSE {d['LMMSE_ser_mean']:.4f}+-{d['LMMSE_ser_se']:.4f}  "
            f"bestUn({d['BestUn_names']}) {d['BestUn_ser_mean']:.4f}  "
            f"NOMA {d['NOMA_ser_mean']:.4f}  "
            f"ratio {d['ratio_BestUn_over_DR_mean']:.2f}+-{d['ratio_BestUn_over_DR_se']:.2f}  "
            f"DR/ref {d['DR_over_matched_ref_mean']:.2f}+-{d['DR_over_matched_ref_se']:.2f}  "
            f"wins={d['DR_wins_all_seeds']}")
    notes.append("Mixture subsets (SER on kappa=0 samples | kappa=1 samples):")
    for d in summ:
        if d["law"].startswith("mix"):
            notes.append(
                f"  beta={d['beta']} {d['law']} {d['setting']:5s} "
                f"DR {d['DR_ser_lo_mean']:.4f} | {d['DR_ser_hi_mean']:.4f}   "
                f"DR-MMSE {d['BLMMSE_ser_lo_mean']:.4f} | {d['BLMMSE_ser_hi_mean']:.4f}   "
                f"LMMSE {d['LMMSE_ser_lo_mean']:.4f} | {d['LMMSE_ser_hi_mean']:.4f}")
    notes.append("Uniform law subsets (SER on kappa<=0.3 | kappa>0.3):")
    for d in summ:
        if d["law"] == "unif06":
            notes.append(
                f"  beta={d['beta']} unif06 {d['setting']:5s} "
                f"DR {d['DR_ser_lo_mean']:.4f} | {d['DR_ser_hi_mean']:.4f}   "
                f"LMMSE {d['LMMSE_ser_lo_mean']:.4f} | {d['LMMSE_ser_hi_mean']:.4f}")
    with open(os.path.join(DATA, "e8b_notes.txt"), "w") as f:
        f.write("\n".join(notes) + "\n")
    print("\n".join(notes))
    print("E8b done.")


if __name__ == "__main__":
    sys.exit(main())
