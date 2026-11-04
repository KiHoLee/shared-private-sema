"""E11b: synchronization impairments simulated at the symbol level under
common random numbers (Reviewer 4, Recommended Change 2).

Reviewer 4 asked for robustness under synchronization errors. E11
(code/exp11_impairments.py) tested only a constant residual phase per user,
and each column of Table VI used independent draws, so the SER changes it
reported (0.038 to 0.045 for the DR) are inside the draw-to-draw spread
measured by code/exp11_seedvar.py. This script simulates the impairments the
letter only argued about, on the E11 configuration, and pairs every level
with the zero-impairment point of the same seed.

Setting (E11, imported from exp11_impairments.py): U=4 users, d=64,
d_c=16, beta=0.5, rho=10 dB, Rayleigh gains clipped at 0.2, tau=0.45,
4,000 embeddings per seed, oracle shared basis V_c for the DR. Each real
coordinate of the masked embedding s_v = M_v x_v is one in-phase symbol of
a single-carrier stream (Section III-A), and the embeddings of a user follow
each other in the stream. The receiver keeps its nominal model in every
case: nominal gains, nominal sigma^2, no phase, frequency or timing
compensation.

Impairments (4 levels each plus the shared zero level):
  cfo     residual carrier frequency offset eps (cycles per embedding of
          d=64 symbols). The phase is aligned at the first coordinate of
          every embedding (a tracker that re-aligns once per embedding) and
          ramps as phi_i = 2 pi eps i / d; the in-phase projection keeps
          cos(phi_i) of symbol i, the quadrature part is lost. One sign per
          user (immaterial, since cos is even). eps in {0.01,0.05,0.1,0.25}.
  timing  residual symbol timing offset delta (fraction of a symbol). The
          overall pulse is raised cosine, roll-off 0.25, truncated to +-4
          taps. User v's stream arrives delayed by delta_v = +-delta (one
          sign per user) and is sampled at the nominal instants, so sample i
          holds sum_k p(k - delta_v) s_{v,i-k}, including leakage from the
          neighbouring embeddings. The gain of the current embedding
          multiplies its samples. delta in {0.05,0.1,0.2,0.3}.
  pn      Wiener phase noise with linewidth-times-symbol-time beta_pn: the
          phase starts at zero at the first coordinate of every embedding
          and accumulates increments N(0, 2 pi beta_pn) per symbol,
          independently per user and embedding. beta_pn in
          {1e-4, 1e-3, 3e-3, 1e-2}.
  static  per-user constant residual phase uniform within +-phi_max (deg),
          applied with the E11 construction (matched_filter with h cos phi,
          receivers built from h). phi_max in {10, 30, 60, 90}. The E11
          construction divides the matched-filter output by h cos(phi), so
          its NMSE explodes when cos(phi) nears zero (phi_max = 90). The RM
          front end has no expected-Gram matched filter, so there "static"
          is applied at the symbol level and equals "static_sym".
  static_sym  the same constant phase applied at the symbol level like the
          other impairments (consistency check of the two constructions).
  gain    multiplicative gain-estimation error h_hat = h (1 + sigma_e eps),
          as in E11, rerun under common random numbers.
          sigma_e in {0.05, 0.1, 0.2, 0.3}.

Two front ends.
  EG (primary): the expected-Gram front end of every experiment in the
      paper (semantic_mac.matched_filter with the true gains) plus the exact
      impairment term (1/h_u) M_u^T sum_v h_v (G_v s_v - s_v) evaluated with
      realized Haar masks. At zero impairment it is exactly the E11 front end.
  RM (check): realized Haar masks throughout, r = sum_v h_v G_v s_v + n and
      tilde_u = M_u^T r / h_u, receivers with the nominal sigma^2, as in
      code/check_mask_realization.py.

Common random numbers, seed s in 0..15:
  default_rng(900000+s) draws z, h and the front-end noise in the order of
      exp11_seedvar.py, so the EG zero level reproduces it seed by seed;
  default_rng(910000+s) draws the Haar masks;
  default_rng(920000+s) draws every impairment variable at unit scale
      (uniform phases, Gaussian gain errors, phase-noise increments, signs),
      which each level rescales;
  default_rng(930000+s) draws the ambient noise of the RM front end.
All levels of all impairments of one seed share these draws.

Reference check (printed and written to the notes): the zero level of EG
must reproduce data/e11_seedvar.txt (LMMSE SER mean 0.2772, DR 0.0407), and
the E11 construction at seed 101 must reproduce the phi=0 column of
data/e11_phase.csv (LMMSE 0.280625, DR 0.038375).

Outputs: data/e11b_sync.csv (mean and SE over seeds per front end,
impairment and level), data/e11b_sync_perseed.csv, data/e11b_notes.txt
"""
import csv
import math
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from semantic_mac import (affinity_matrix, matched_filter, demux_lmmse,
                          demux_dr, metrics, sample_latents_isotropic)
from exp11_impairments import rayleigh, U, D, DC, BATCH, BETA, SNR_DB
from check_mask_realization import haar_masks

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

SEEDS = list(range(16))
ROLLOFF, NTAP = 0.25, 4
RS, FC = 10e6, 5.9e9          # stated symbol rate and carrier for the mapping
LEVELS = {
    "cfo": [0.01, 0.05, 0.1, 0.25],
    "timing": [0.05, 0.1, 0.2, 0.3],
    "pn": [1e-4, 1e-3, 3e-3, 1e-2],
    "static": [10, 30, 60, 90],
    "static_sym": [10, 30, 60, 90],
    "gain": [0.05, 0.1, 0.2, 0.3],
}
UNITS = {"cfo": "cycles_per_embedding", "timing": "symbol_fraction",
         "pn": "linewidth_x_Ts", "static": "phi_max_deg",
         "static_sym": "phi_max_deg", "gain": "sigma_e"}

RHO = 10 ** (SNR_DB / 10)
A_VEC = math.sqrt(BETA) * np.ones(U)
B_MAT = affinity_matrix(A_VEC)
VC = np.eye(D)[:, :DC]


def physical(imp, lev):
    """Physical reading of a level at RS symbols/s and carrier FC."""
    if imp == "cfo":
        df = lev * RS / D
        return f"{df/1e3:.2f} kHz = {df/FC*1e6:.2f} ppm at 5.9 GHz"
    if imp == "timing":
        return f"{lev/RS*1e9:.0f} ns at 10 Msym/s"
    if imp == "pn":
        return (f"linewidth {lev*RS/1e3:.0f} kHz, end-of-embedding phase sd "
                f"{math.degrees(math.sqrt(2*math.pi*lev*D)):.0f} deg")
    if imp in ("static", "static_sym"):
        return f"uniform within +-{lev} deg"
    return f"sigma_e={lev}"


def rc_pulse(t, beta=ROLLOFF):
    t = np.asarray(t, float)
    den = 1.0 - (2.0 * beta * t) ** 2
    out = np.sinc(t) * np.cos(np.pi * beta * t) / np.where(
        np.abs(den) < 1e-12, 1.0, den)
    sing = np.abs(den) < 1e-12
    out[sing] = (np.pi / 4) * np.sinc(1.0 / (2 * beta))
    return out


def impair(s, imp, lev, draws):
    """Apply impairment to the masked streams s (batch, U, d): returns G s."""
    batch = s.shape[0]
    i = np.arange(D)
    if imp == "cfo":
        phi = 2 * np.pi * lev * draws["sign"][None, :, None] * i[None, None, :] / D
        return np.cos(phi) * s
    if imp == "pn":
        inc = draws["pn"] * math.sqrt(2 * np.pi * lev)      # (batch, U, d)
        inc[:, :, 0] = 0.0                                   # aligned at i=0
        return np.cos(np.cumsum(inc, axis=2)) * s
    if imp == "static_sym":
        phi = np.radians(lev) * draws["uphi"]                # (batch, U)
        return np.cos(phi)[:, :, None] * s
    if imp == "timing":
        out = np.empty_like(s)
        k = np.arange(-NTAP, NTAP + 1)
        for v in range(U):
            g = rc_pulse(k - lev * draws["sign"][v])         # taps g_k, k=-4..4
            S = s[:, v, :].reshape(-1)
            c = np.convolve(S, g)                            # c_n = sum_m g[m] S[n-m]
            out[:, v, :] = c[NTAP:NTAP + S.size].reshape(batch, D)
        return out
    raise ValueError(imp)


def eval_rx(tilde, h_rx, z):
    lm, _ = demux_lmmse(tilde, B_MAT, h_rx, RHO)
    dr = demux_dr(tilde, B_MAT, h_rx, RHO, A_VEC, VC)
    cl, nl, sl = metrics(lm, z)
    cd, nd, sd = metrics(dr, z)
    return dict(LMMSE_cos=cl, LMMSE_nmse=nl, LMMSE_ser=sl,
                DR_cos=cd, DR_nmse=nd, DR_ser=sd)


def distortion_stats(s, gs):
    """Coherent gain g and signal-to-distortion ratio of G s relative to s."""
    g = float((s * gs).sum() / (s * s).sum())
    dist = float(((gs - g * s) ** 2).sum() / (s * s).sum())
    sdr = 10 * math.log10(g * g / dist) if dist > 0 else float("inf")
    return g, sdr


def run_seed(seed):
    rows = []
    rng = np.random.default_rng(900000 + seed)
    z = sample_latents_isotropic(BATCH, U, D, DC, A_VEC, rng)
    h = rayleigh(BATCH, rng)
    st = rng.bit_generator.state

    def mf(h_true):
        r = np.random.default_rng()
        r.bit_generator.state = st
        t, _ = matched_filter(z, B_MAT, RHO, r, h_fixed=h_true)
        return t

    tilde0 = mf(h)                                   # exp11_seedvar draw
    Ms = haar_masks(B_MAT, D, np.random.default_rng(910000 + seed))
    Mst = np.stack(Ms)                               # (U, d, d)
    ri = np.random.default_rng(920000 + seed)
    draws = dict(uphi=ri.uniform(-1, 1, (BATCH, U)),
                 eps=ri.standard_normal((BATCH, U)),
                 pn=ri.standard_normal((BATCH, U, D)),
                 sign=ri.choice([-1.0, 1.0], size=U))
    n_amb = (np.random.default_rng(930000 + seed).standard_normal((BATCH, D))
             * math.sqrt(1.0 / RHO))
    s = np.einsum('vde,bve->bvd', Mst, z)            # s_v = M_v x_v
    r0 = np.einsum('bv,bvd->bd', h, s)

    def despread(r):
        return np.stack([(r @ Ms[u]) / h[:, u][:, None] for u in range(U)],
                        axis=1)

    tildeR0 = despread(r0 + n_amb)
    base = dict(seed=seed)
    for fe, t0 in (("EG", tilde0), ("RM", tildeR0)):
        rows.append(dict(base, frontend=fe, impairment="none", level=0.0,
                         coh_gain=1.0, sdr_db=float("inf"),
                         **eval_rx(t0, h, z)))
    for imp, levs in LEVELS.items():
        for lev in levs:
            if imp == "static":
                hc = h * np.cos(np.radians(lev) * draws["uphi"])
                out_eg = eval_rx(mf(hc), h, z)
                gs = np.cos(np.radians(lev) * draws["uphi"])[:, :, None] * s
                g, sdr = distortion_stats(s, gs)
                out_rm = eval_rx(despread(np.einsum('bv,bvd->bd', h, gs)
                                          + n_amb), h, z)
            elif imp == "gain":
                hh = np.clip(h * (1.0 + lev * draws["eps"]), 0.05, None)
                out_eg = eval_rx(tilde0, hh, z)
                out_rm = eval_rx(tildeR0, hh, z)
                g, sdr = 1.0, float("inf")
            else:
                gs = impair(s, imp, lev, draws)
                g, sdr = distortion_stats(s, gs)
                rdev = np.einsum('bv,bvd->bd', h, gs - s)
                out_eg = eval_rx(tilde0 + despread(rdev), h, z)
                out_rm = eval_rx(despread(r0 + rdev + n_amb), h, z)
            for fe, out in (("EG", out_eg), ("RM", out_rm)):
                rows.append(dict(base, frontend=fe, impairment=imp,
                                 level=lev, coh_gain=g, sdr_db=sdr, **out))
    print(f"seed {seed} done: EG zero LMMSE {rows[0]['LMMSE_ser']:.4f} "
          f"DR {rows[0]['DR_ser']:.4f}", flush=True)
    return rows


def reference_e11_phase0():
    """E11 construction at seed 101 (phi=0 column of data/e11_phase.csv)."""
    rng = np.random.default_rng(101)
    z = sample_latents_isotropic(BATCH, U, D, DC, A_VEC, rng)
    h = rayleigh(BATCH, rng)
    phi = rng.uniform(0.0, 0.0, size=(BATCH, U))
    out = eval_rx(matched_filter(z, B_MAT, RHO, rng,
                                 h_fixed=h * np.cos(phi))[0], h, z)
    return out["LMMSE_ser"], out["DR_ser"]


def se(x):
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / math.sqrt(len(x)))


def main():
    with Pool(4) as p:
        per = p.map(run_seed, SEEDS)
    allrows = [r for rs in per for r in rs]
    keys = ["seed", "frontend", "impairment", "level", "coh_gain", "sdr_db",
            "LMMSE_cos", "LMMSE_nmse", "LMMSE_ser", "DR_cos", "DR_nmse",
            "DR_ser"]
    with open(os.path.join(DATA, "e11b_sync_perseed.csv"), "w") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in allrows:
            w.writerow({k: r[k] for k in keys})

    def sel(fe, imp, lev):
        rs = [r for r in allrows if r["frontend"] == fe and
              r["impairment"] == imp and r["level"] == lev]
        rs.sort(key=lambda r: r["seed"])
        assert len(rs) == len(SEEDS)
        return rs

    notes = []
    zero_eg = sel("EG", "none", 0.0)
    lm0 = np.array([r["LMMSE_ser"] for r in zero_eg])
    dr0 = np.array([r["DR_ser"] for r in zero_eg])
    ref_lm, ref_dr = reference_e11_phase0()
    notes.append("Reference check")
    notes.append(f"  EG zero level, 16 seeds (900000+s): LMMSE SER mean "
                 f"{lm0.mean():.4f} sd {lm0.std(ddof=1):.4f}, DR mean "
                 f"{dr0.mean():.4f} sd {dr0.std(ddof=1):.4f} "
                 "(data/e11_seedvar.txt: 0.2772 sd 0.0053, 0.0407 sd 0.0019)")
    notes.append(f"  E11 construction seed 101, phi=0: LMMSE {ref_lm:.6f}, DR "
                 f"{ref_dr:.6f} (data/e11_phase.csv: 0.280625, 0.038375)")
    notes.append(f"Setting: U={U}, d={D}, d_c={DC}, beta={BETA}, SNR={SNR_DB} "
                 f"dB, batch={BATCH}, seeds 0..15, RC roll-off {ROLLOFF}, "
                 f"+-{NTAP} taps, mapping at {RS/1e6:.0f} Msym/s and "
                 f"{FC/1e9:.1f} GHz")

    out = []
    for fe in ("EG", "RM"):
        z0 = sel(fe, "none", 0.0)
        notes.append(f"\nFront end {fe}: mean +- SE over 16 seeds; dSER = "
                     "paired increase over the zero level of the same seed")
        for imp in ["none"] + list(LEVELS):
            levs = [0.0] if imp == "none" else LEVELS[imp]
            for lev in levs:
                rs = sel(fe, imp, lev)
                rec = dict(frontend=fe, impairment=imp, level=lev,
                           unit=UNITS.get(imp, ""),
                           physical="" if imp == "none" else physical(imp, lev),
                           coh_gain=float(np.mean([r["coh_gain"] for r in rs])),
                           sdr_db=float(np.mean([r["sdr_db"] for r in rs])))
                for m in ("DR", "LMMSE"):
                    for q in ("ser", "nmse", "cos"):
                        x = [r[f"{m}_{q}"] for r in rs]
                        rec[f"{m}_{q}_mean"] = float(np.mean(x))
                        rec[f"{m}_{q}_se"] = se(x)
                    d = [r[f"{m}_ser"] - q0[f"{m}_ser"] for r, q0 in zip(rs, z0)]
                    rec[f"{m}_dser_mean"] = float(np.mean(d))
                    rec[f"{m}_dser_se"] = se(d)
                gap = [r["LMMSE_ser"] - r["DR_ser"] for r in rs]
                rec["gap_mean"] = float(np.mean(gap))
                rec["gap_se"] = se(gap)
                rec["ratio"] = rec["LMMSE_ser_mean"] / rec["DR_ser_mean"]
                rec["dr_wins"] = int(sum(g > 0 for g in gap))
                out.append(rec)
                notes.append(
                    f"  {imp:10s} {lev:<7g} ({rec['physical']}) g={rec['coh_gain']:.3f}"
                    f" SDR={rec['sdr_db']:.1f} dB | DR SER {rec['DR_ser_mean']:.4f}"
                    f"+-{rec['DR_ser_se']:.4f} dSER {rec['DR_dser_mean']:+.4f}"
                    f"+-{rec['DR_dser_se']:.4f} NMSE {rec['DR_nmse_mean']:.3f}"
                    f" | LMMSE SER {rec['LMMSE_ser_mean']:.4f}+-{rec['LMMSE_ser_se']:.4f}"
                    f" dSER {rec['LMMSE_dser_mean']:+.4f}+-{rec['LMMSE_dser_se']:.4f}"
                    f" NMSE {rec['LMMSE_nmse_mean']:.3f} | ratio {rec['ratio']:.2f}"
                    f" DR better in {rec['dr_wins']}/16")
    cols = list(out[0].keys())
    with open(os.path.join(DATA, "e11b_sync.csv"), "w") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in out:
            w.writerow(r)
    with open(os.path.join(DATA, "e11b_notes.txt"), "w") as f:
        f.write("\n".join(notes) + "\n")
    print("\n".join(notes))
    print("E11b done.")


if __name__ == "__main__":
    main()
