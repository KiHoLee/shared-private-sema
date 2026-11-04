"""Key lines of the four small Reviewer 2 checks (R2.2, R2.3, R2.4, R2.5).

Reads the outputs of exp2b_dc_misspec.py, exp2c_adapter_snr.py,
exp9b_threshold_by_scheme.py and exp10b_clip_reference.py and writes the
lines the response letter quotes, including a few ratios derived from
those CSV files (no new draws). Run after the four scripts.

Output: data/rev5_small_notes.txt
"""
import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def rows(name):
    with open(os.path.join(DATA, name)) as f:
        return list(csv.DictReader(f))


def head(name, n):
    with open(os.path.join(DATA, name)) as f:
        return [ln.rstrip("\n") for ln in f][:n]


def main():
    out = ["Rev5 small Reviewer 2 checks, run on u24 "
           "(python 3.12.3, numpy 2.5.4, torch 2.11.0, CPU).",
           "Reproduction note: E10 synthetic rows reproduce exactly. BERT-pool "
           "experiments (E2, E9) reproduce the u24 run of E21 exactly and the "
           "Mac-produced e2_ladder.csv / e9_retrieval.csv within Monte Carlo "
           "error (spectral subspace error identical to 10 digits).", ""]

    # ---- R2.2 ---------------------------------------------------------------
    out.append("[R2.2] exp2b_dc_misspec.py -> e2b_dc_misspec.csv "
               "(E2 draws replayed, true d_c = 128, N = 200)")
    out += [ln for ln in head("e2b_notes.txt", 7) if ln.startswith("check")]
    e2b = rows("e2b_dc_misspec.csv")
    for s in ("12", "16", "20"):
        lm = [r for r in e2b if r["snr"] == s and r["receiver"] == "LMMSE"][0]
        for basis in ("spectral", "oracle"):
            sel = [r for r in e2b if r["snr"] == s and r["basis"] == basis]
            out.append(f"{s} dB DR-{basis} SER by d_c' " + ", ".join(
                f"{r['dc_rx']}: {float(r['ser']):.4f}"
                f" ({float(r['ser_minus_spec128']):+.4f} se "
                f"{float(r['se_paired']):.4f})" for r in sel)
                + f" | LMMSE {float(lm['ser']):.4f}")
        sel = [r for r in e2b if r["snr"] == s and r["basis"] == "spectral"]
        worst = max(float(r["ser"]) for r in sel)
        out.append(f"{s} dB LMMSE over worst spectral d_c' choice: "
                   f"{float(lm['ser']) / worst:.2f}x")
    sel = [r for r in e2b if r["snr"] == "20" and r["basis"] == "spectral"]
    out.append("captured true-shared fraction by d_c' " + ", ".join(
        f"{r['dc_rx']}: {float(r['captured_shared_frac']):.3f}" for r in sel)
        + "; truly shared fraction of the receiver subspace " + ", ".join(
        f"{r['dc_rx']}: {float(r['rx_subspace_shared_frac']):.3f}"
        for r in sel))
    out += [ln for ln in head("e2b_notes.txt", 200)
            if ln.startswith("calibration eigenvalues")]
    out.append("")

    # ---- R2.3 ---------------------------------------------------------------
    out.append("[R2.3] exp2c_adapter_snr.py -> e2c_adapter_snr.csv, "
               "e2c_summary.csv (10 calibration seeds, 6 training variants)")
    out.append(head("e2c_notes.txt", 1)[0])
    e2c = rows("e2c_adapter_snr.csv")
    for r in e2c:
        if r["run"] == "E2chain" and r["snr"] in ("8", "12", "16", "20"):
            sp, ad = float(r["spec_ser"]), float(r["adapt_ser"])
            g_sh = sp - float(r["ser_hyb_adapt_shared"])
            g_pr = sp - float(r["ser_hyb_adapt_private"])
            out.append(
                f"E2 chain {r['snr']} dB: SER spec {sp:.4f} adapt {ad:.4f} "
                f"gain {float(r['ser_gain']):+.4f} (se "
                f"{float(r['ser_gain_se']):.4f}), relative "
                f"{(sp - ad) / sp:+.1%}; shared-block part {g_sh:+.4f}, "
                f"private-block part {g_pr:+.4f}; NMSE spec "
                f"{float(r['spec_nmse']):.5f} adapt "
                f"{float(r['adapt_nmse']):.5f} "
                f"({(float(r['adapt_nmse']) / float(r['spec_nmse']) - 1):+.3%})"
                f"; shared-block NMSE {float(r['adapt_nmse_sh']) - float(r['spec_nmse_sh']):+.5f}"
                f", private-block NMSE {float(r['adapt_nmse_pr']) - float(r['spec_nmse_pr']):+.5f}")
    summ = rows("e2c_summary.csv")
    for r in summ:
        if r["snr"] in ("8", "12", "16", "20"):
            out.append(
                f"seeds {r['train_snr']:5s} lam {r['lam_align']} "
                f"{r['snr']:>2s} dB: SER gain {float(r['ser_gain']):+.4f} "
                f"(se {float(r['ser_gain_se']):.4f}), better in "
                f"{r['n_seeds_adapter_better_ser']}/10 seeds; shared part "
                f"{float(r['ser_gain_shared']):+.4f}, private part "
                f"{float(r['ser_gain_private']):+.4f}; dNMSE "
                f"{float(r['d_nmse']):+.5f} (sh {float(r['d_nmse_sh']):+.5f},"
                f" pr {float(r['d_nmse_pr']):+.5f}); spectral SER "
                f"{float(r['spec_ser']):.4f}; subspace error "
                f"{float(r['spec_subspace_err']):.4f} -> "
                f"{float(r['adapter_subspace_err']):.4f}")
    # relative gain and NMSE change of the paper adapter over seeds
    for s in ("12", "16", "20"):
        sel = [r for r in e2c if r["run"].startswith("seed")
               and r["train_snr"] == "U0-20" and r["lam_align"] == "0.2"
               and r["snr"] == s]
        rel = np.array([float(r["ser_gain"]) / float(r["spec_ser"])
                        for r in sel])
        nrel = np.array([float(r["adapt_nmse"]) / float(r["spec_nmse"]) - 1
                         for r in sel])
        out.append(f"paper adapter, {s} dB, over 10 seeds: relative SER "
                   f"reduction mean {rel.mean():+.1%} (min {rel.min():+.1%}, "
                   f"max {rel.max():+.1%}); relative NMSE change mean "
                   f"{nrel.mean():+.3%}")
    out.append("")

    # ---- R2.4 ---------------------------------------------------------------
    out.append("[R2.4] exp9b_threshold_by_scheme.py -> "
               "e9b_threshold_by_scheme.csv, e9b_scheme_summary.csv")
    out += [ln for ln in head("e9b_notes.txt", 400)
            if ln.startswith(("largest", "OMA, NOMA-SIC", "per-scheme means",
                              "pooled4_full reproduces"))]
    for r in rows("e9b_scheme_summary.csv"):
        out.append(
            f"{r['scheme']:9s} full cos {float(r['cos']):.4f} SER "
            f"{float(r['ser']):.4f} shared top-1 {float(r['shared_top1']):.4f}"
            f" private top-1 {float(r['private_top1']):.4f} shared-block cos "
            f"{float(r['shared_block_cos']):.4f} private-block cos "
            f"{float(r['private_block_cos']):.4f} private misses to another "
            f"user's sentence {float(r['private_miss_other_user_frac']):.3f}")
    out += [ln for ln in head("e9b_notes.txt", 400)
            if "first >= 50%" in ln or ln.startswith(("pooled8_private:",
                                                      "pooled8_shared:",
                                                      "private-block bin"))]
    b = rows("e9b_threshold_by_scheme.csv")
    for kind, bins in (("per_scheme_full", ("0.30", "0.45", "0.60")),
                       ("per_scheme_private", ("0.15", "0.20", "0.30"))):
        for lo in bins:
            sel = [r for r in b if r["kind"] == kind and r["cos_lo"] == lo
                   and int(r["n"]) >= 30]
            out.append(f"{kind} bin {lo}: private top-1 " + ", ".join(
                f"{r['scheme']} {float(r['private_top1']):.3f}" for r in sel)
                + " | shared top-1 " + ", ".join(
                f"{r['scheme']} {float(r['shared_top1']):.3f}" for r in sel))
    out.append("")

    # ---- R2.5 ---------------------------------------------------------------
    out.append("[R2.5] exp10b_clip_reference.py -> e10b_clip_reference.csv")
    out += [ln for ln in head("e10b_notes.txt", 400)
            if ln.startswith(("check", "synthetic h_min=0.0 ",
                              "synthetic h_min=0.01 ", "synthetic h_min=0.2 ",
                              "BERT"))]
    e10 = rows("e10b_clip_reference.csv")
    s5 = {r["h_min"]: r for r in e10 if r["config"] == "synthetic_d64_10dB"
          and r["beta"] == "0.5" and r["receiver"] == "DR"}
    out.append(f"synthetic beta=0.5 advantage: no clip "
               f"{float(s5['0.0']['best_unstructured_over_DR']):.3f}x, h_min=0.2 "
               f"{float(s5['0.2']['best_unstructured_over_DR']):.3f}x, ratio "
               f"{float(s5['0.2']['best_unstructured_over_DR']) / float(s5['0.0']['best_unstructured_over_DR']):.3f}")
    bt = {r["h_min"]: r for r in e10 if r["config"] == "bert_d768_20dB"
          and r["receiver"] == "DR-spec"}
    out.append(f"BERT 20 dB advantage LMMSE/DR: no clip "
               f"{float(bt['0.0']['lmmse_over_DR']):.3f}x, h_min=0.2 "
               f"{float(bt['0.2']['lmmse_over_DR']):.3f}x (ratio "
               f"{float(bt['0.2']['lmmse_over_DR']) / float(bt['0.0']['lmmse_over_DR']):.3f}), "
               f"h_min=0.3 {float(bt['0.3']['lmmse_over_DR']):.3f}x (ratio "
               f"{float(bt['0.3']['lmmse_over_DR']) / float(bt['0.0']['lmmse_over_DR']):.3f}); "
               f"DR SER change from no clip to h_min=0.2 "
               f"{float(bt['0.2']['ser']) / float(bt['0.0']['ser']) - 1:+.1%}")
    with open(os.path.join(DATA, "rev5_small_notes.txt"), "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
