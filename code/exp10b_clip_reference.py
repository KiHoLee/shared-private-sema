"""E10b: unclipped reference for the deep-fade clip study (Reviewer 2.5).

The E10 clip sweep of exp10_eval_choices.py covers h_min in {0.05, 0.1,
0.2, 0.3} only, so it has no unclipped reference point, and it uses one
configuration (synthetic d = 64 contents at 10 dB). This script adds
h_min = 0 (no clip) and h_min = 0.01 to that sweep for every receiver,
and repeats the clip comparison at the E2 BERT operating point.

Part 1, synthetic. The E10 construction unchanged: U = 4, d = 64,
d_c = 16, 10 dB, beta in {0.5, 0.7}, 4,000 draws, a fresh generator
seeded 77 for every clip value (so every clip value sees the same
complex gains and noise, which makes the comparison paired), OMA and
NOMA-SIC seeded 95000 + int(1000 h_min + beta), SER at tau = 0.45. The
published clip values are rerun and must reproduce data/e10_clip_sweep.csv
exactly. E[1/h^2] is reported as the sample mean used by exp10 and in
closed form, (1 - exp(-x))/x + E1(x) with x = h_min^2 for |h|^2 ~ Exp(1),
which is infinite at h_min = 0.

Part 2, BERT. The E2 configuration (exp2_structure.py, real BERT contents,
d = 768, d_c = 128, beta = 0.5, N = 200 spectral basis) at 20 dB. The E2
random chain is replayed (exp2b_dc_misspec.replay_e2), and the 20 dB
matched filter is redrawn from the generator state of E2 for each
h_min in {0, 0.01, 0.05, 0.1, 0.2, 0.3}, so the gains and noise are those
of E2 and only the clip differs. The h_min = 0.2 row is compared with the
20 dB LMMSE and DR-spec entries of data/e2_ladder.csv. That file was
produced on the Mac, and on u24 the BERT-pool draws differ at the Monte
Carlo level (see exp9b_threshold_by_scheme.py), so the comparison is
within Monte Carlo error, not to the digit.

Outputs: data/e10b_clip_reference.csv, data/e10b_notes.txt
"""
import csv
import math
import os
import sys
import numpy as np
from scipy.special import exp1

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp10_eval_choices as e10
from exp10_eval_choices import cosines, NAMES, UNSTRUCTURED
from semantic_mac import (matched_filter, demux_lmmse, demux_dr, metrics,
                          TAU, U_MAIN as U)
import exp2b_dc_misspec as e2b

DATA = e10.DATA
CLIPS = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3]
PUBLISHED = (0.05, 0.1, 0.2, 0.3)


def einv_closed(hm):
    if hm == 0:
        return float("inf")
    x = hm ** 2
    return float((1 - math.exp(-x)) / x + exp1(x))


def main():
    notes = []
    rows = []
    rho = 10 ** (10 / 10)
    with open(os.path.join(DATA, "e10_clip_sweep.csv")) as f:
        pub = {(float(r["h_min"]), float(r["beta"])): r
               for r in csv.DictReader(f)}
    exact = True
    einv_dev = 0.0
    for hm in CLIPS:
        rng_h = np.random.default_rng(77)
        for b in (0.5, 0.7):
            c = cosines(b, rho, rng_h, hm, conv_seed=int(1000 * hm + b))
            ser = {n: float((c[n] < 0.45).mean()) for n in NAMES}
            mcos = {n: float(c[n].mean()) for n in NAMES}
            best_un = min(ser[n] for n in UNSTRUCTURED)
            ratio = best_un / ser["DR"] if ser["DR"] > 0 else float("inf")
            _, h = matched_filter(np.zeros((4000, U, 4)), np.eye(U), rho,
                                  np.random.default_rng(5), h_min=hm)
            einv = float(np.mean(1.0 / h ** 2))
            rows.append(("synthetic_d64_10dB", hm, b, ser, mcos, ratio, einv,
                         einv_closed(hm)))
            if hm in PUBLISHED:
                p = pub[(hm, b)]
                ok = all(float(p[f"{n}_ser"]) == ser[n] for n in NAMES) and \
                    float(p["best_unstructured_over_DR"]) == ratio
                exact &= ok
                einv_dev = max(einv_dev, abs(float(p["E_inv_h2"]) - einv))
            line = (f"synthetic h_min={hm} beta={b}: " + " ".join(
                f"{n} {ser[n]:.4f}" for n in NAMES)
                + f" | best unstructured over DR {ratio:.2f}x, "
                f"E[1/h^2] sample {einv:.3f} closed {einv_closed(hm):.3f}")
            notes.append(line)
            print(line, flush=True)
    notes.insert(0, f"check: SER of every receiver and the advantage ratio "
                    f"of the published clip rows of e10_clip_sweep.csv "
                    f"reproduced exactly: {exact}; largest deviation of the "
                    f"sample E[1/h^2] (float summation order): {einv_dev:.1e}")

    # ---- Part 2: E2 BERT point at 20 dB -----------------------------------
    pool = e2b.load_pool()
    ch = e2b.replay_e2(pool, [20])
    a, B = ch["a"], ch["B"]
    z, x, tilde0, h0, state = ch["evals"][20]
    rho20 = 100.0
    ladder = e2b.read_ladder()[20]
    bert = []
    err_ref = None
    for hm in CLIPS:
        rng = np.random.default_rng(0)
        rng.bit_generator.state = state
        tilde, h = matched_filter(x, B, rho20, rng, h_min=hm)
        lm, _ = demux_lmmse(tilde, B, h, rho20)
        dr = demux_dr(tilde, B, h, rho20, a, ch["V_spec"])
        m_lm, m_dr = metrics(lm, x), metrics(dr, x)
        err_dr = e2b.inst_err(dr, x)
        err_lm = e2b.inst_err(lm, x)
        bert.append((hm, m_lm, m_dr, err_lm, err_dr,
                     float(np.mean(1.0 / h ** 2)),
                     float((np.abs(h) <= max(hm, 1e-300)).mean()),
                     float(h.min())))
        if hm == 0.2:
            err_ref = (err_lm, err_dr)
            notes.append(f"check BERT 20 dB h_min=0.2: LMMSE ser {m_lm[2]!r} "
                         f"nmse {m_lm[1]!r} (e2_ladder {ladder['LMMSE_ser']!r},"
                         f" {ladder['LMMSE_nmse']!r}); DR-spec ser {m_dr[2]!r} "
                         f"nmse {m_dr[1]!r} (e2_ladder "
                         f"{ladder['DR-spec_ser']!r}, "
                         f"{ladder['DR-spec_nmse']!r})")
    for i, r in enumerate(bert):
        hm, m_lm, m_dr, err_lm, err_dr, einv, frac, hmin_obs = r
        d_dr = err_dr - err_ref[1]
        d_lm = err_lm - err_ref[0]
        se_dr = float(d_dr.std(ddof=1) / math.sqrt(d_dr.size))
        se_lm = float(d_lm.std(ddof=1) / math.sqrt(d_lm.size))
        ratio = m_lm[2] / m_dr[2] if m_dr[2] > 0 else float("inf")
        bert[i] = (hm, m_lm, m_dr, ratio, float(d_lm.mean()), se_lm,
                   float(d_dr.mean()), se_dr, einv, frac, hmin_obs)
        line = (f"BERT 20 dB h_min={hm}: LMMSE ser {m_lm[2]:.5f} cos "
                f"{m_lm[0]:.4f} nmse {m_lm[1]:.4f} | DR-spec ser "
                f"{m_dr[2]:.5f} cos {m_dr[0]:.4f} nmse {m_dr[1]:.4f} | "
                f"LMMSE/DR {ratio:.2f}x | dSER vs h_min=0.2: LMMSE "
                f"{d_lm.mean():+.5f} (se {se_lm:.5f}), DR {d_dr.mean():+.5f} "
                f"(se {se_dr:.5f}) | clipped fraction {frac:.4f}, "
                f"E[1/h^2] sample {einv:.2f}, min h {hmin_obs:.2e}")
        notes.append(line)
        print(line, flush=True)

    with open(os.path.join(DATA, "e10b_clip_reference.csv"), "w") as f:
        f.write("config,h_min,beta,receiver,ser,mean_cos,nmse,"
                "best_unstructured_over_DR,lmmse_over_DR,"
                "dser_vs_hmin02,se_paired,E_inv_h2_sample,E_inv_h2_closed\n")
        for (cfg, hm, b, ser, mcos, ratio, einv, einvc) in rows:
            for n in NAMES:
                f.write(f"{cfg},{hm},{b},{n},{ser[n]},{mcos[n]},,{ratio},,,,"
                        f"{einv},{einvc}\n")
        for (hm, m_lm, m_dr, ratio, dl, sel, dd, sed, einv, frac,
             hmin_obs) in bert:
            for n, m, dlt, se in (("LMMSE", m_lm, dl, sel),
                                  ("DR-spec", m_dr, dd, sed)):
                f.write(f"bert_d768_20dB,{hm},0.5,{n},{m[2]},{m[0]},{m[1]},,"
                        f"{ratio},{dlt},{se},{einv},{einv_closed(hm)}\n")
    with open(os.path.join(DATA, "e10b_notes.txt"), "w") as f:
        f.write("\n".join(notes) + "\n")
    print("E10b done.")


if __name__ == "__main__":
    main()
