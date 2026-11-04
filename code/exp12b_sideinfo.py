"""E12b: how much of the E12 corpus result rests on matched side information
(Reviewer 4, Weak Aspect 1 and Recommended Change 1).

E12 (code/exp12_corpus.py) transmits unmodified BERT embeddings of
neighbouring AG-News sentences, so no shared/private split is imposed.
Its receivers, however, are given the share coefficients of a rank-one fit
to the mean clean Gram of the EVALUATION tuples in each quantized affinity
bin, and the same coefficients set the transmit masks B. That fit uses the
evaluation data and is therefore partly oracle. This experiment replaces
it by two calibration-only alternatives and keeps everything else fixed.

Side-information modes, applied identically at transmitter (B) and at
every affinity-aware receiver (SR, SC, LMMSE, BLMMSE/DR-MMSE, DR):
  evalbin  (a) as published: rank-one fit to the mean clean Gram of the
               evaluation tuples of the bin (reproduces E12 and E16).
  calbin   (b) per-bin fit on calibration tuples only: the calibration
               tuples whose quantized affinity lies within +-0.05 of the
               bin value, falling back to the 10 calibration tuples whose
               measured affinity is nearest to the bin value when fewer
               than 10 qualify. The quantized per-tuple affinity is still
               used to select the bin, as in E12.
  static   (c) one rank-one fit to the mean clean Gram of all 200
               calibration tuples, used for every tuple. No per-tuple
               affinity at transmitter or receiver.

Setting. Identical to E12: pool data/bert_agnews_8000.pt, U=4, KNN=8,
200 calibration tuples from the first 7000 sentences, 3000 evaluation
tuples from the last 1000, rng seed 17 for the tuples, spectral basis from
the calibration tuples, d_c in {32, 64, 128, 256}, affinity grid 0.05,
bins with fewer than 20 evaluation tuples skipped (2981 tuples remain).
SNR 20 dB as in E12 and 30 dB as in the E16 follow-up run.

Seeds. Common random numbers across the three modes. At 20 dB the channel
and noise seed of a bin is 4000+int(100 qb) and the OMA/NOMA seed is
97000+int(100 qb), exactly as in E12. At 30 dB the channel seed is
8000+int(1000 qb)+30, exactly as in E16, and the OMA/NOMA seed is
97000+int(100 qb)+30. Mode (a) at 20 dB therefore reproduces
data/e12_corpus.csv and mode (a) at 30 dB reproduces the coarse-grid rows
of data/e16_scene_gap.csv at d_c=32 and 128.

Metrics as in E12: mean cosine, SER (cosine below 0.45), exact top-1 and
top-5 retrieval of the transmitted sentence in the 1000-sentence
evaluation bank, and scene error (top-1 outside the anchor's 8 nearest
neighbours). OMA and NOMA do not use the affinity and are unchanged
across modes.

Platform note. build_tuples draws U sentences from the unordered output of
np.argpartition, whose element order depends on the SIMD sort kernel. On
an x86 host with AVX2/AVX-512 the tuples then differ from the published
E12 tuples (Mac, numpy 2.2.5). The u24 run therefore sets
  NPY_DISABLE_CPU_FEATURES="X86_V3 X86_V4 AVX512_ICL AVX512_SPR"
which selects the generic kernel. The script checks the measured affinity
of every evaluation tuple against data/e12_affinity.csv and records the
result in the notes.

Outputs: data/e12b_sideinfo.csv, data/e12b_notes.txt
"""
import os
import numpy as np
import torch

from semantic_mac import (affinity_matrix, matched_filter, demux_sr,
                          demux_sc, demux_lmmse, demux_dr,
                          learn_structure_spectral, oma_observe,
                          demux_noma_genie, EmbeddingPool)
from exp6_sideinfo import demux_block_lmmse
from exp12_corpus import (build_tuples, rank_one_fit, U, D, KNN, N_CAL,
                          N_EVAL, N_TRAIN_POOL, DC_GRID, POOL_PT)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
os.makedirs(DATA, exist_ok=True)

SNR_GRID = [20, 30]
MODES = ["evalbin", "calbin", "static"]
QSTEP = 0.05
WIN = 0.05          # calibration-bin window of mode (b)
N_FALLBACK = 10     # nearest calibration tuples when the window is thin
MIN_BIN = 20        # E12 skips evaluation bins smaller than this
DC_FREE = ["OMA", "NOMA", "SR", "SC", "LMMSE"]
DC_DEP = ["BLMMSE", "DR"]
NAMES = DC_FREE + DC_DEP


def quantize(beta):
    return np.clip(np.round(beta / QSTEP) * QSTEP, 0.05, 0.95)


def mean_offdiag(x):
    G = np.einsum('nud,nvd->nuv', x, x)
    return G[:, ~np.eye(U, dtype=bool)].reshape(len(x), -1).mean(axis=1)


def seeds(qb, snr_db):
    if snr_db == 20:
        return 4000 + int(qb * 100), 97000 + int(qb * 100)
    return 8000 + int(qb * 1000) + snr_db, 97000 + int(qb * 100) + snr_db


def score(e, xb, bank, truth, scene):
    en = e / (np.linalg.norm(e, axis=2, keepdims=True) + 1e-12)
    cos = (en * xb).sum(-1).ravel()
    sims = en.reshape(-1, D) @ bank.T
    top = sims.argmax(axis=1)
    top5 = np.argpartition(-sims, 5, axis=1)[:, :5]
    return (cos, top == truth, (top5 == truth[:, None]).any(axis=1),
            (top[:, None] == scene).any(axis=1))


def main():
    rng = np.random.default_rng(17)
    obj = torch.load(POOL_PT, map_location="cpu", weights_only=False)
    X = obj.numpy() if torch.is_tensor(obj) else np.asarray(obj)
    Z = EmbeddingPool(X).X
    idx_train = np.arange(N_TRAIN_POOL)
    idx_eval = np.arange(N_TRAIN_POOL, len(Z))
    bank = Z[idx_eval]
    pos_of = {g: i for i, g in enumerate(idx_eval)}

    tup_cal, _ = build_tuples(Z, idx_train, N_CAL, rng)
    tup_ev, nbr_ev = build_tuples(Z, idx_eval, N_EVAL, rng)
    x_cal = Z[tup_cal]
    x_ev = Z[tup_ev]
    beta_ev = mean_offdiag(x_ev)
    beta_cal = mean_offdiag(x_cal)
    ref = np.loadtxt(os.path.join(DATA, "e12_affinity.csv"), delimiter=",",
                     skiprows=1)[:, 1]
    dev = np.abs(beta_ev - ref)
    tuple_check = (f"tuple check vs data/e12_affinity.csv: max |diff| "
                   f"{dev.max():.2e}, mismatched tuples "
                   f"{int((dev > 1e-9).sum())} of {N_EVAL}")
    q_ev = quantize(beta_ev)
    q_cal = quantize(beta_cal)
    bins = [qb for qb in np.unique(q_ev) if (q_ev == qb).sum() >= MIN_BIN]
    n_used = int(sum((q_ev == qb).sum() for qb in bins))

    G_cal_all = np.einsum('nud,nvd->nuv', x_cal, x_cal).mean(axis=0)
    a_static = rank_one_fit(G_cal_all)
    notes = [tuple_check,
             f"NPY_DISABLE_CPU_FEATURES="
             f"{os.environ.get('NPY_DISABLE_CPU_FEATURES', '')!r}, "
             f"numpy {np.__version__}",
             f"pool {Z.shape}, KNN={KNN}, U={U}, N_CAL={N_CAL}, "
             f"N_EVAL={N_EVAL}, evaluated tuples {n_used}",
             f"measured affinity eval mean {beta_ev.mean():.4f} sd "
             f"{beta_ev.std():.4f}; calibration mean {beta_cal.mean():.4f} "
             f"sd {beta_cal.std():.4f}",
             "static rank-one fit on all calibration tuples a = "
             + " ".join(f"{v:.4f}" for v in a_static)
             + f" (mean a_u a_v {np.mean(np.outer(a_static, a_static)[~np.eye(U, dtype=bool)]):.4f})"]

    # share coefficients per bin and mode
    a_of = {}
    for qb in bins:
        m = np.where(q_ev == qb)[0]
        Gb = np.einsum('nud,nvd->nuv', x_ev[m], x_ev[m]).mean(axis=0)
        a_eval = rank_one_fit(Gb)
        sel = np.where(np.abs(q_cal - qb) <= WIN + 1e-9)[0]
        rule = "window"
        if len(sel) < N_FALLBACK:
            sel = np.argsort(np.abs(beta_cal - qb))[:N_FALLBACK]
            rule = "nearest10"
        Gc = np.einsum('nud,nvd->nuv', x_cal[sel], x_cal[sel]).mean(axis=0)
        a_calb = rank_one_fit(Gc)
        a_of[qb] = {"evalbin": a_eval, "calbin": a_calb, "static": a_static}
        off = ~np.eye(U, dtype=bool)
        notes.append(
            f"bin {qb:.2f}: n_eval={len(m):4d} n_cal={len(sel):3d} ({rule}) "
            f"mean a_u a_v evalbin={np.outer(a_eval, a_eval)[off].mean():.3f} "
            f"calbin={np.outer(a_calb, a_calb)[off].mean():.3f} "
            f"static={np.outer(a_static, a_static)[off].mean():.3f}")
    print("\n".join(notes))

    V_of = {d_c: learn_structure_spectral(x_cal, d_c)[0] for d_c in DC_GRID}

    rows = []
    keep = {}          # per-decision outcomes for the paired comparisons
    for snr_db in SNR_GRID:
        rho = 10 ** (snr_db / 10)
        for mode in MODES:
            acc = {(n, d_c): [[], [], [], []] for n in NAMES
                   for d_c in DC_GRID}
            for qb in bins:
                m = np.where(q_ev == qb)[0]
                xb = x_ev[m]
                a_hat = a_of[qb][mode]
                B = affinity_matrix(a_hat)
                sb, sc = seeds(qb, snr_db)
                tilde, h = matched_filter(xb, B, rho,
                                          np.random.default_rng(sb))
                lm, _ = demux_lmmse(tilde, B, h, rho)
                rng_c = np.random.default_rng(sc)
                free = {"OMA": oma_observe(xb, h, rho, rng_c),
                        "NOMA": demux_noma_genie(xb, h, rho, rng_c),
                        "SR": demux_sr(tilde, B, h),
                        "SC": demux_sc(tilde, h),
                        "LMMSE": lm}
                truth = np.array([pos_of[g] for g in tup_ev[m].ravel()])
                scene = np.repeat(np.array([[pos_of[g] for g in row]
                                            for row in nbr_ev[m]]), U, axis=0)
                for n, e in free.items():
                    s = score(e, xb, bank, truth, scene)
                    for d_c in DC_GRID:
                        for k in range(4):
                            acc[(n, d_c)][k].append(s[k])
                for d_c in DC_GRID:
                    V = V_of[d_c]
                    dep = {"BLMMSE": demux_block_lmmse(tilde, B, h, rho,
                                                       a_hat, V),
                           "DR": demux_dr(tilde, B, h, rho, a_hat, V)}
                    for n, e in dep.items():
                        s = score(e, xb, bank, truth, scene)
                        for k in range(4):
                            acc[(n, d_c)][k].append(s[k])
            for d_c in DC_GRID:
                for n in NAMES:
                    cos, t1, t5, sh = (np.concatenate(v)
                                       for v in acc[(n, d_c)])
                    ser = float((cos < 0.45).mean())
                    if n in ("LMMSE", "BLMMSE", "DR"):
                        keep[(snr_db, mode, d_c, n)] = np.stack(
                            [(cos < 0.45), t1, ~sh]).astype(float)
                    rows.append((snr_db, mode, d_c, n, float(cos.mean()),
                                 ser, float(np.sqrt(ser * (1 - ser)
                                                    / len(cos))),
                                 float(t1.mean()), float(t5.mean()),
                                 float(sh.mean()), float(1 - sh.mean()),
                                 len(cos)))
                line = (f"{snr_db}dB {mode:7s} d_c={d_c:3d} " + " ".join(
                    f"{r[3]} ser={r[5]:.3f}/top1={r[7]:.3f}/"
                    f"scene_err={r[10]:.3f}"
                    for r in rows[-len(NAMES):]
                    if r[3] in ("LMMSE", "BLMMSE", "DR", "NOMA")))
                notes.append(line)
                print(line, flush=True)

    # paired differences (common channel and noise draws, same decisions)
    def pd(x, y):
        dlt = x - y
        return dlt.mean(), dlt.std(ddof=1) / np.sqrt(len(dlt))

    notes.append("paired differences, mean +- standard error over "
                 "decisions (positive LMMSE-DR means DR lower), "
                 "rows give SER, top1 and scene_err")
    for snr_db in SNR_GRID:
        for d_c in DC_GRID:
            for mode in MODES:
                parts = []
                for k, lab in enumerate(("ser", "top1", "scene_err")):
                    m1, s1 = pd(keep[(snr_db, mode, d_c, "LMMSE")][k],
                                keep[(snr_db, mode, d_c, "DR")][k])
                    m2, s2 = pd(keep[(snr_db, mode, d_c, "LMMSE")][k],
                                keep[(snr_db, mode, d_c, "BLMMSE")][k])
                    parts.append(f"{lab} LMMSE-DR {m1:+.4f}+-{s1:.4f} "
                                 f"LMMSE-BLMMSE {m2:+.4f}+-{s2:.4f}")
                notes.append(f"{snr_db}dB d_c={d_c:3d} {mode:7s} "
                             + " | ".join(parts))
                if mode == "evalbin":
                    continue
                parts = []
                for n in ("DR", "BLMMSE", "LMMSE"):
                    m1, s1 = pd(keep[(snr_db, mode, d_c, n)][0],
                                keep[(snr_db, "evalbin", d_c, n)][0])
                    parts.append(f"{n} {m1:+.4f}+-{s1:.4f}")
                notes.append(f"{snr_db}dB d_c={d_c:3d} SER {mode} minus "
                             f"evalbin: " + " ".join(parts))
    print("\n".join(notes[-36:]))

    with open(os.path.join(DATA, "e12b_sideinfo.csv"), "w") as f:
        f.write("snr_db,mode,d_c,receiver,cos,ser,ser_se,top1,top5,"
                "scene_hit,scene_err,n_decisions\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")
    with open(os.path.join(DATA, "e12b_notes.txt"), "w") as f:
        f.write("\n".join(notes) + "\n")
    print("E12b done.")


if __name__ == "__main__":
    main()
