"""E2c: why the adapter gain depends on SNR (Reviewer 2.3).

Reviewer 2 asked why the benefit of the learned adapter becomes smaller at
moderate SNR. In E2 the adapter lowers the SER of the spectral basis at
16 and 20 dB but raises it at 8 to 12 dB, and its NMSE is slightly above
that of the spectral basis at every SNR below 20 dB. This script tests
whether that inversion survives other calibration draws, whether it
comes from the training objective (training SNR drawn uniformly from 0 to
20 dB, alignment penalty lambda_a = 0.2), and in which latent block the
SER change arises.

Setting. The E2 configuration of exp2_structure.py: real BERT AG-News
contents, U = 4, d = 768, d_c = 128, beta = 0.5, N = 200 calibration
pairs from the training partition, mixing R of E2 (seed 7), Rayleigh
fading clipped at 0.2, 400 adapter steps, batch 24, lr 3e-4,
lambda_o = 1. The adapter is a copy of exp2_structure.train_adapter with
the training SNR range as an argument (with the range 0 to 20 dB it is
the same function and consumes the same draws).

Part A (reproduction and decomposition on E2 itself). The E2 random chain
is replayed with the copied trainer (exp2b_dc_misspec.replay_e2). The
adapter subspace error must reproduce data/e2_caliberr.txt, and the
DR-spec and DR-adapt entries are compared with data/e2_ladder.csv.

Part B (calibration seeds and training ablation). Ten calibration draws,
seeds 2300 + s, s = 0..9, each with its own N = 200 pairs and spectral
basis. For each draw six adapters are trained from the same spectral
initialization and the same training channel draws (seed 2400 + s, torch
seed s): training SNR uniform on [0, 20] dB (paper), fixed 12 dB, fixed
20 dB, each with lambda_a = 0.2 (paper) and lambda_a = 0. All are
evaluated on common held-out tuples, 4,000 per SNR (seed 2500 + SNR) at
0, 4, 8, 12, 16 and 20 dB, so differences are paired across variants and
seeds.

Decomposition. Estimates are mapped to the latent frame, zhat = e R. The
NMSE splits exactly into a shared-block part (first d_c coordinates) and
a private-block part. For the SER, two hybrids are formed in the latent
frame: the adapter's shared block with the spectral private block, and
the spectral shared block with the adapter's private block. Their SER
minus the spectral SER attributes the SER change to each block (the
remainder is the interaction).

Outputs: data/e2c_adapter_snr.csv (one row per seed, variant and SNR),
data/e2c_summary.csv (mean and standard error over the ten seeds),
data/e2c_notes.txt
"""
import math
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp2_structure as e2
from exp2_structure import dr_torch
import exp2b_dc_misspec as e2b
from semantic_mac import (affinity_matrix, matched_filter, demux_dr,
                          learn_structure_spectral, subspace_error, TAU)

DATA = e2.DATA
U, D, DC = e2.U, e2.D, e2.DC
SNRS = [0, 4, 8, 12, 16, 20]
SEEDS = list(range(10))
VARIANTS = [("U0-20", 0, 20, 0.2), ("fix12", 12, 12, 0.2),
            ("fix20", 20, 20, 0.2), ("U0-20", 0, 20, 0.0),
            ("fix12", 12, 12, 0.0), ("fix20", 20, 20, 0.0)]


def train_adapter_snr(x_cal, a, V_init, rng, snr_lo=0.0, snr_hi=20.0,
                      steps=400, batch=24, lr=3e-4, lam_orth=1.0,
                      lam_align=0.2, seed=0):
    """exp2_structure.train_adapter with the training SNR range exposed.
    With snr_lo = 0 and snr_hi = 20 it is that function line by line."""
    torch.manual_seed(seed)
    W = torch.nn.Parameter(torch.tensor(V_init.T, dtype=torch.float32))
    a_t = torch.tensor(a, dtype=torch.float32)
    B_np = affinity_matrix(a)
    B_t = torch.tensor(B_np, dtype=torch.float32)
    opt = torch.optim.Adam([W], lr=lr)
    n_cal = x_cal.shape[0]
    x_cal_t = torch.tensor(x_cal, dtype=torch.float32)
    for it in range(steps):
        idx = torch.randint(0, n_cal, (batch,))
        x = x_cal_t[idx]
        # one uniform draw per step in every case (uniform(lo, lo) returns
        # lo), so the channel draws below are identical across variants
        snr_db = float(rng.uniform(snr_lo, snr_hi))
        rho = 10 ** (snr_db / 10)
        tilde_np, h_np = matched_filter(x.numpy().astype(np.float64),
                                        B_np, rho, rng)
        tilde = torch.tensor(tilde_np, dtype=torch.float32)
        h = torch.tensor(h_np, dtype=torch.float32)
        x_hat = dr_torch(tilde, B_t, h, rho, a_t, W, DC)
        cosd = 1.0 - torch.nn.functional.cosine_similarity(
            x_hat, x, dim=2).mean()
        y = x @ W.T
        s = torch.nn.functional.normalize(y[:, :, :DC], dim=2)
        align = (s.unsqueeze(1) - s.unsqueeze(2)).pow(2).sum(-1).mean()
        orth = (W @ W.T - torch.eye(D)).pow(2).mean()
        loss = cosd + lam_align * align + lam_orth * orth
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        Uo, _, Vo = torch.linalg.svd(W)
        Wo = Uo @ Vo
    return Wo.numpy().astype(np.float64)


def cos_inst(zhat, z):
    zn = zhat / (np.linalg.norm(zhat, axis=2, keepdims=True) + 1e-12)
    return (zn * z).sum(-1)       # z rows are unit norm


def compare(out_s, out_a, z, R):
    """Spectral vs adapter on the same tuples, with block decomposition."""
    zs, za = out_s @ R, out_a @ R
    r = {}
    for tag, zh in (("spec", zs), ("adapt", za)):
        c = cos_inst(zh, z)
        r[f"{tag}_cos"] = float(c.mean())
        r[f"{tag}_ser"] = float((c < TAU).mean())
        r[f"{tag}_nmse"] = float(((zh - z) ** 2).sum(-1).mean())
        r[f"{tag}_nmse_sh"] = float(((zh[..., :DC] - z[..., :DC]) ** 2)
                                    .sum(-1).mean())
        r[f"{tag}_nmse_pr"] = float(((zh[..., DC:] - z[..., DC:]) ** 2)
                                    .sum(-1).mean())
        r[f"{tag}_err"] = (c < TAU).astype(float).ravel()
    h_sh = np.concatenate([za[..., :DC], zs[..., DC:]], axis=-1)
    h_pr = np.concatenate([zs[..., :DC], za[..., DC:]], axis=-1)
    r["ser_hyb_adapt_shared"] = float((cos_inst(h_sh, z) < TAU).mean())
    r["ser_hyb_adapt_private"] = float((cos_inst(h_pr, z) < TAU).mean())
    d_i = r["spec_err"] - r["adapt_err"]
    r["ser_gain"] = float(d_i.mean())            # spectral minus adapter
    r["ser_gain_se"] = float(d_i.std(ddof=1) / math.sqrt(d_i.size))
    del r["spec_err"], r["adapt_err"]
    return r


COLS = ["spec_cos", "adapt_cos", "spec_nmse", "adapt_nmse", "spec_nmse_sh",
        "adapt_nmse_sh", "spec_nmse_pr", "adapt_nmse_pr", "spec_ser",
        "adapt_ser", "ser_gain", "ser_gain_se", "ser_hyb_adapt_shared",
        "ser_hyb_adapt_private"]


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    pool = e2b.load_pool()
    notes = []
    rows = []

    # ---- Part A: E2 chain replayed with the copied trainer ----------------
    ch = e2b.replay_e2(pool, SNRS, adapter_fn=lambda xc, a, Q, rng:
                       train_adapter_snr(xc, a, Q, rng))
    a, B, R = ch["a"], ch["B"], ch["R"]
    V_true = ch["V_true"]
    err_ad = subspace_error(ch["W_ad"].T[:, :DC], V_true)
    notes.append(f"check A: adapter subspace error {err_ad!r} "
                 f"(e2_caliberr.txt adapter_err=0.47115266508068226), "
                 f"spectral {subspace_error(ch['V_spec'], V_true)!r}")
    lad = e2b.read_ladder()
    for s in SNRS:
        rho = 10 ** (s / 10)
        z, x, tilde, h, _ = ch["evals"][s]
        out_s = demux_dr(tilde, B, h, rho, a, ch["V_spec"])
        out_a = demux_dr(tilde, B, h, rho, a, ch["W_ad"].T[:, :DC])
        r = compare(out_s, out_a, z, R)
        rows.append(("E2chain", "U0-20", 0.2, s, err_ad, r))
        notes.append(f"A {s:2d} dB spec ser {r['spec_ser']:.5f} nmse "
                     f"{r['spec_nmse']:.5f} | adapt ser {r['adapt_ser']:.5f} "
                     f"nmse {r['adapt_nmse']:.5f} | e2_ladder spec "
                     f"{lad[s]['DR-spec_ser']:.5f}/{lad[s]['DR-spec_nmse']:.5f}"
                     f" adapt {lad[s]['DR-adapt_ser']:.5f}/"
                     f"{lad[s]['DR-adapt_nmse']:.5f}")
        print(notes[-1], flush=True)
    del ch

    # ---- Part B: ten calibration seeds, six training variants -------------
    evals = {}
    for s in SNRS:
        rng_e = np.random.default_rng(2500 + s)
        z, x = e2.gen_clean(pool, e2.BATCH_EVAL, a, R, rng_e,
                            idx_pool=e2.IDX_EVAL)
        tilde, h = matched_filter(x, B, 10 ** (s / 10), rng_e)
        evals[s] = (z, x, tilde, h)
    for seed in SEEDS:
        rng_s = np.random.default_rng(2300 + seed)
        _, x_cal = e2.gen_clean(pool, e2.N_PAIR, a, R, rng_s,
                                idx_pool=e2.IDX_TRAIN)
        V_spec, _ = learn_structure_spectral(x_cal, DC)
        Q, _ = np.linalg.qr(np.hstack([V_spec,
                                       rng_s.standard_normal((D, D - DC))]))
        out_spec = {s: demux_dr(evals[s][2], B, evals[s][3], 10 ** (s / 10),
                                a, V_spec) for s in SNRS}
        for (vname, lo, hi, lam) in VARIANTS:
            rng_t = np.random.default_rng(2400 + seed)
            W = train_adapter_snr(x_cal, a, Q, rng_t, snr_lo=lo, snr_hi=hi,
                                  lam_align=lam, seed=seed)
            Va = W.T[:, :DC]
            err = subspace_error(Va, V_true)
            for s in SNRS:
                z, x, tilde, h = evals[s]
                out_a = demux_dr(tilde, B, h, 10 ** (s / 10), a, Va)
                r = compare(out_spec[s], out_a, z, R)
                r["spec_subspace"] = subspace_error(V_spec, V_true)
                rows.append((f"seed{seed}", vname, lam, s, err, r))
            print(f"seed {seed} {vname} lam {lam}: subspace err {err:.4f} "
                  + " ".join(f"{s}dB:{rows[-6 + i][5]['ser_gain']:+.4f}"
                             for i, s in enumerate(SNRS)), flush=True)

    with open(os.path.join(DATA, "e2c_adapter_snr.csv"), "w") as f:
        f.write("run,train_snr,lam_align,snr,adapter_subspace_err,"
                + ",".join(COLS) + "\n")
        for (run, v, lam, s, err, r) in rows:
            f.write(f"{run},{v},{lam},{s},{err},"
                    + ",".join(str(r[c]) for c in COLS) + "\n")

    # ---- summary over seeds -----------------------------------------------
    summ = []
    derived = {
        "d_nmse": lambda r: r["adapt_nmse"] - r["spec_nmse"],
        "d_nmse_sh": lambda r: r["adapt_nmse_sh"] - r["spec_nmse_sh"],
        "d_nmse_pr": lambda r: r["adapt_nmse_pr"] - r["spec_nmse_pr"],
        "d_cos": lambda r: r["adapt_cos"] - r["spec_cos"],
        "ser_gain": lambda r: r["ser_gain"],
        "ser_gain_shared": lambda r: r["spec_ser"] - r["ser_hyb_adapt_shared"],
        "ser_gain_private": lambda r: r["spec_ser"]
        - r["ser_hyb_adapt_private"],
        "spec_ser": lambda r: r["spec_ser"],
        "adapt_ser": lambda r: r["adapt_ser"],
    }
    for (vname, lo, hi, lam) in VARIANTS:
        for s in SNRS:
            sel = [r for (run, v, l_, ss, err, r) in rows
                   if run.startswith("seed") and v == vname and l_ == lam
                   and ss == s]
            errs = [err for (run, v, l_, ss, err, r) in rows
                    if run.startswith("seed") and v == vname and l_ == lam
                    and ss == s]
            line = {"train_snr": vname, "lam_align": lam, "snr": s,
                    "n_seeds": len(sel),
                    "adapter_subspace_err": float(np.mean(errs)),
                    "spec_subspace_err": float(np.mean(
                        [r["spec_subspace"] for r in sel]))}
            for k, fn in derived.items():
                v = np.array([fn(r) for r in sel])
                line[k] = float(v.mean())
                line[k + "_se"] = float(v.std(ddof=1) / math.sqrt(len(v)))
            line["n_seeds_adapter_better_ser"] = int(
                sum(r["ser_gain"] > 0 for r in sel))
            summ.append(line)
    keys = list(summ[0].keys())
    with open(os.path.join(DATA, "e2c_summary.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for line in summ:
            f.write(",".join(str(line[k]) for k in keys) + "\n")
    notes.append("Part B, mean (SE) over 10 calibration seeds; ser_gain = "
                 "SER(spectral) - SER(adapter), >0 means the adapter helps; "
                 "d_nmse = NMSE(adapter) - NMSE(spectral), <0 helps")
    for line in summ:
        notes.append(
            f"B {line['train_snr']:5s} lam={line['lam_align']} "
            f"{line['snr']:2d} dB: ser_gain {line['ser_gain']:+.4f} "
            f"({line['ser_gain_se']:.4f}) [shared {line['ser_gain_shared']:+.4f}"
            f", private {line['ser_gain_private']:+.4f}] "
            f"seeds better {line['n_seeds_adapter_better_ser']}/10 | "
            f"d_nmse {line['d_nmse']:+.5f} ({line['d_nmse_se']:.5f}) "
            f"[sh {line['d_nmse_sh']:+.5f}, pr {line['d_nmse_pr']:+.5f}] | "
            f"d_cos {line['d_cos']:+.4f} | spec ser {line['spec_ser']:.4f} | "
            f"sub err {line['spec_subspace_err']:.4f}->"
            f"{line['adapter_subspace_err']:.4f}")
    # Part A decomposition lines
    for (run, v, lam, s, err, r) in rows:
        if run == "E2chain":
            notes.append(
                f"A-decomp {s:2d} dB: ser_gain {r['ser_gain']:+.5f} "
                f"(se {r['ser_gain_se']:.5f}) shared "
                f"{r['spec_ser'] - r['ser_hyb_adapt_shared']:+.5f} private "
                f"{r['spec_ser'] - r['ser_hyb_adapt_private']:+.5f} | "
                f"d_nmse_sh {r['adapt_nmse_sh'] - r['spec_nmse_sh']:+.5f} "
                f"d_nmse_pr {r['adapt_nmse_pr'] - r['spec_nmse_pr']:+.5f}")
    with open(os.path.join(DATA, "e2c_notes.txt"), "w") as f:
        f.write("\n".join(notes) + "\n")
    print("\n".join(notes))
    print("E2c done.")


if __name__ == "__main__":
    main()
