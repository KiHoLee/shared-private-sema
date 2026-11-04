"""E9b: meaning of the SER threshold, per scheme and per block
(Reviewer 2.4).

The E9 curve of data/e9_threshold.csv bins private top-1 retrieval against
the full-embedding cosine, pooled over the four Wiener-type receivers
LMMSE, DR-spectral, DR-adapter and block LMMSE only. Unshrunk schemes
(SimR, genie NOMA-SIC) retrieve the private sentence often at a low full
cosine, so this script asks whether the binned meaning of tau holds per
scheme, and whether a block cosine predicts retrieval across all schemes.

Setting. Exactly the E9 construction of exp9_retrieval.py: seed 7 for the
mixing R, the N = 200 calibration pairs and the adapter, seed 909 for the
4,000 held-out evaluation tuples and their channels, seed 60001 for OMA
and NOMA-SIC, 20 dB, beta = 0.5, U = 4, d = 768, d_c = 128. For all eight
schemes and every user it records
  full-embedding cosine (the SER statistic),
  shared-block and private-block cosine in the latent frame (zhat = e R),
  shared and private top-1 hits against the held-out banks.
Since the private bank holds the normalized private coordinates, the
private-block cosine is exactly the query's cosine with its correct bank
entry. For every private miss it also records whether the returned
sentence is another user's private sentence of the same tuple or the
tuple's shared sentence (all of them belong to the held-out bank).

Bins of width 0.05 on [0, 1], a bin reported when it holds at least 30
users (the E9 rule):
  per_scheme_full   one curve per scheme against the full cosine
  per_scheme_private, per_scheme_shared  the same against the block cosines
  pooled4_full      the four Wiener-type receivers against the full
                    cosine (must reproduce data/e9_threshold.csv)
  pooled8_full      all eight schemes against the full cosine
  pooled8_private   all eight schemes, private top-1 against the
                    private-block cosine
  pooled8_shared    all eight schemes, shared top-1 against the
                    shared-block cosine

Check. The per-scheme means are compared with data/e9_retrieval.csv
(cos, ser, shared_top1, private_top1) and pooled4_full with
data/e9_threshold.csv. Those files were produced on the Mac
(data/environment.txt). On the u24 server the BERT-pool draws differ at
the Monte Carlo level (the spectral subspace error, which depends only on
the span of the PCA directions, reproduces to ten digits, so the likely
cause is the platform-dependent sign of the pool PCA directions). The
exact check is therefore against the 20 dB rows of
data/e21_conventional_references.csv, which ran the same E9 construction
on u24, for the four schemes the two scripts share.

Outputs: data/e9b_threshold_by_scheme.csv (bins),
data/e9b_scheme_summary.csv (per-scheme means), data/e9b_notes.txt
"""
import csv
import math
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp9_retrieval as e9
from semantic_mac import (EmbeddingPool, affinity_matrix, matched_filter,
                          demux_lmmse, demux_dr, random_orthogonal,
                          learn_structure_spectral, metrics, oma_observe,
                          demux_noma_genie, demux_sr)
from exp6_sideinfo import demux_block_lmmse
from exp2_structure import train_adapter

DATA = e9.DATA
U, D, DC = e9.U, e9.D, e9.DC
WIENER4 = ("DRspec", "DRadapt", "LMMSE", "BLMMSE")
EDGES = np.arange(0.0, 1.0001, 0.05)
MIN_N = 30


def ucos(a, b):
    an = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return (an * bn).sum(-1)


def estimates():
    """The E9 construction, line by line as in exp9_retrieval.main()."""
    rng = np.random.default_rng(7)
    obj = torch.load(e9.POOL_PT, map_location="cpu")
    X = obj.numpy() if torch.is_tensor(obj) else np.asarray(obj)
    pool = EmbeddingPool(X)
    a = math.sqrt(e9.BETA) * np.ones(U)
    B = affinity_matrix(a)
    rho = 10 ** (e9.SNR_DB / 10)
    R = random_orthogonal(D, rng)
    V_true = R[:, :DC]
    z_cal, _ = e9.sample_with_index(pool, e9.N_PAIR, a, rng, e9.IDX_TRAIN)
    x_cal = z_cal @ R.T
    V_spec, _ = learn_structure_spectral(x_cal, DC)
    Q, _ = np.linalg.qr(np.hstack([V_spec, rng.standard_normal((D, D - DC))]))
    W_ad = train_adapter(x_cal, a, Q, rng)
    V_adapt = W_ad.T[:, :DC]
    rng_e = np.random.default_rng(909)
    z, idx = e9.sample_with_index(pool, e9.BATCH_EVAL, a, rng_e, e9.IDX_EVAL)
    x = z @ R.T
    tilde, h = matched_filter(x, B, rho, rng_e)
    lm, _ = demux_lmmse(tilde, B, h, rho)
    rng_c = np.random.default_rng(60001)
    est = {
        "OMA": oma_observe(x, h, rho, rng_c),
        "NOMA": demux_noma_genie(x, h, rho, rng_c),
        "LMMSE": lm,
        "DRspec": demux_dr(tilde, B, h, rho, a, V_spec),
        "DRadapt": demux_dr(tilde, B, h, rho, a, V_adapt),
        "BLMMSE": demux_block_lmmse(tilde, B, h, rho, a, V_true),
        "DRoracle": demux_dr(tilde, B, h, rho, a, V_true),
        "SR": demux_sr(tilde, B, h),
    }
    return pool, R, z, x, idx, est


def argmax_pick(query, bank):
    """Bank row of the top-one match of each unit-normalized query."""
    q = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-12)
    b = bank / (np.linalg.norm(bank, axis=1, keepdims=True) + 1e-12)
    out = np.empty(len(q), dtype=int)
    for s in range(0, len(q), 2000):
        out[s:s + 2000] = (q[s:s + 2000] @ b.T).argmax(axis=1)
    return out


def bin_rows(kind, scheme, c, hs, hp):
    out = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (c >= lo) & (c < hi)
        n = int(m.sum())
        acc_s = float(hs[m].mean()) if n >= MIN_N else float("nan")
        acc_p = float(hp[m].mean()) if n >= MIN_N else float("nan")
        out.append((kind, scheme, f"{lo:.2f}", f"{hi:.2f}", n, acc_s, acc_p))
    return out


def main():
    pool, R, z, x, idx, est = estimates()
    bank_s = pool.pca_coords(e9.IDX_EVAL, DC)
    bank_p = pool.pca_coords(e9.IDX_EVAL, D - DC)
    pos_of = {g: i for i, g in enumerate(e9.IDX_EVAL)}
    truth_s = np.array([pos_of[g] for g in idx[:, 0]])
    rec = {}
    summary = []
    for name in e9.NAMES:
        e = est[name]
        cos, _, ser = metrics(e, x)
        zhat = e @ R
        cf, cs, cp, hs, hp = [], [], [], [], []
        miss_other, miss_shared, n_miss = 0, 0, 0
        allpos = np.array([[pos_of[g] for g in row] for row in idx])
        for u in range(U):
            truth_p = np.array([pos_of[g] for g in idx[:, u + 1]])
            hs.append(e9.top1(zhat[:, u, :DC], bank_s, truth_s))
            hp.append(e9.top1(zhat[:, u, DC:], bank_p, truth_p))
            # which sentence a private miss returns: another user's private
            # sentence of the same tuple, the tuple's shared sentence, or
            # any other sentence of the bank
            pick = argmax_pick(zhat[:, u, DC:], bank_p)
            miss = pick != truth_p
            others = np.delete(allpos[:, 1:], u, axis=1)
            n_miss += int(miss.sum())
            miss_other += int((miss & (pick[:, None] == others).any(1)).sum())
            miss_shared += int((miss & (pick == allpos[:, 0])).sum())
            # same expression as exp9 for the full cosine
            cf.append((e[:, u] / (np.linalg.norm(e[:, u], axis=1,
                                                 keepdims=True) + 1e-12)
                       * x[:, u]).sum(-1))
            cs.append(ucos(zhat[:, u, :DC], z[:, u, :DC]))
            cp.append(ucos(zhat[:, u, DC:], z[:, u, DC:]))
        r = {k: np.concatenate(v) for k, v in
             dict(cf=cf, cs=cs, cp=cp, hs=hs, hp=hp).items()}
        r["hs"] = r["hs"].astype(float)
        r["hp"] = r["hp"].astype(float)
        rec[name] = r
        summary.append((name, cos, ser, float(r["hs"].mean()),
                        float(r["hp"].mean()), float(r["cs"].mean()),
                        float(r["cp"].mean()),
                        miss_other / max(n_miss, 1),
                        miss_shared / max(n_miss, 1)))

    # --- reproduction check against data/e9_retrieval.csv ----------------
    notes = []
    # REF_DATA may point at a mirror of the canonical data/ folder
    ref = os.environ.get("REF_DATA", DATA)
    with open(os.path.join(ref, "e9_retrieval.csv")) as f:
        pub = {row["scheme"]: row for row in csv.DictReader(f)}
    all_exact = True
    for s in summary:
        p = pub[s[0]]
        ok = all(float(p[k]) == v for k, v in
                 zip(("cos", "ser", "shared_top1", "private_top1"), s[1:5]))
        all_exact &= ok
        notes.append(f"check {s[0]:9s} cos {s[1]!r} ser {s[2]!r} shared "
                     f"{s[3]!r} private {s[4]!r} exact={ok}")
    notes.append(f"per-scheme means reproduce e9_retrieval.csv exactly: "
                 f"{all_exact}")
    dmax = max(abs(float(pub[s[0]][k]) - v) for s in summary for k, v in
               zip(("cos", "ser", "shared_top1", "private_top1"), s[1:5]))
    notes.append(f"largest absolute difference to e9_retrieval.csv over the "
                 f"8 schemes and 4 columns: {dmax:.5f}")
    # E21 ran the same E9 construction (seeds 7, 909, 60001) on the u24
    # server; its 20 dB rows of the four shared schemes are the reference
    # for a run in the same environment
    with open(os.path.join(ref, "e21_conventional_references.csv")) as f:
        e21 = {row["scheme"]: row for row in csv.DictReader(f)
               if row["snr"] == "20"}
    ok21 = True
    for mine, theirs in (("OMA", "OMA"), ("NOMA", "NOMA-SIC"),
                         ("LMMSE", "LMMSE"), ("DRspec", "DR-spectral")):
        srow = [x for x in summary if x[0] == mine][0]
        # SER and top-1 are counts and must agree exactly; the mean cosine
        # may differ in the last binary digit (summation order)
        ok = all(float(e21[theirs][k]) == v for k, v in
                 zip(("ser", "shared_top1", "private_top1"), srow[2:5])) \
            and abs(float(e21[theirs]["cos"]) - srow[1]) < 1e-12
        ok21 &= ok
        notes.append(f"check vs E21 20 dB row {theirs}: ser, top-1 exact and "
                     f"cos within 1e-12: {ok} (cos difference "
                     f"{abs(float(e21[theirs]['cos']) - srow[1]):.1e})")
    notes.append(f"OMA, NOMA-SIC, LMMSE, DR-spectral reproduce the 20 dB rows "
                 f"of e21_conventional_references.csv: {ok21}")

    rows = []
    for name in e9.NAMES:
        r = rec[name]
        rows += bin_rows("per_scheme_full", name, r["cf"], r["hs"], r["hp"])
        rows += bin_rows("per_scheme_private", name, r["cp"], r["hs"],
                         r["hp"])
        rows += bin_rows("per_scheme_shared", name, r["cs"], r["hs"],
                         r["hp"])
    cat = lambda names, k: np.concatenate([rec[n][k] for n in names])
    rows += bin_rows("pooled4_full", "+".join(WIENER4), cat(WIENER4, "cf"),
                     cat(WIENER4, "hs"), cat(WIENER4, "hp"))
    rows += bin_rows("pooled8_full", "all8", cat(e9.NAMES, "cf"),
                     cat(e9.NAMES, "hs"), cat(e9.NAMES, "hp"))
    rows += bin_rows("pooled8_private", "all8", cat(e9.NAMES, "cp"),
                     cat(e9.NAMES, "hs"), cat(e9.NAMES, "hp"))
    rows += bin_rows("pooled8_shared", "all8", cat(e9.NAMES, "cs"),
                     cat(e9.NAMES, "hs"), cat(e9.NAMES, "hp"))

    # pooled4_full against the published binned curve
    with open(os.path.join(ref, "e9_threshold.csv")) as f:
        pub_thr = list(csv.DictReader(f))
    p4 = [r for r in rows if r[0] == "pooled4_full"]
    thr_ok = all(int(a["n"]) == b[4] and
                 (a["private_top1"] == "nan" and math.isnan(b[6])
                  or float(a["private_top1"]) == b[6])
                 for a, b in zip(pub_thr, p4))
    notes.append(f"pooled4_full reproduces e9_threshold.csv exactly: {thr_ok}")
    for a_, b_ in zip(pub_thr, p4):
        if int(a_["n"]) >= MIN_N or b_[4] >= MIN_N:
            notes.append(f"pooled4 bin {b_[2]}: n {b_[4]} (e9 {a_['n']}), "
                         f"private top-1 {b_[6]:.4f} (e9 "
                         f"{float(a_['private_top1']):.4f})")

    with open(os.path.join(DATA, "e9b_threshold_by_scheme.csv"), "w") as f:
        f.write("kind,scheme,cos_lo,cos_hi,n,shared_top1,private_top1\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")
    with open(os.path.join(DATA, "e9b_scheme_summary.csv"), "w") as f:
        f.write("scheme,cos,ser,shared_top1,private_top1,"
                "shared_block_cos,private_block_cos,"
                "private_miss_other_user_frac,private_miss_shared_frac\n")
        for s in summary:
            f.write(",".join(str(v) for v in s) + "\n")

    notes.append("scheme    full_cos  ser     shared_top1 private_top1 "
                 "shared_blk_cos private_blk_cos miss->other_user "
                 "miss->shared_sentence")
    for s in summary:
        notes.append(f"{s[0]:9s} {s[1]:.4f}   {s[2]:.4f}  {s[3]:.4f}      "
                     f"{s[4]:.4f}       {s[5]:.4f}         {s[6]:.4f}"
                     f"          {s[7]:.4f}           {s[8]:.4f}")

    # crossing of 50% retrieval per curve
    def first_cross(kind, scheme, col):
        for r in rows:
            if r[0] == kind and r[1] == scheme and r[4] >= MIN_N \
                    and r[col] >= 0.5:
                return r[2]
        return None
    for name in e9.NAMES:
        notes.append(f"{name:9s} private top-1 first >= 50% in full-cos bin "
                     f"{first_cross('per_scheme_full', name, 6)}, shared "
                     f"top-1 in bin {first_cross('per_scheme_full', name, 5)}")
    notes.append(f"pooled8 private top-1 first >= 50% in private-block-cos "
                 f"bin {first_cross('pooled8_private', 'all8', 6)}")
    notes.append(f"pooled8 shared top-1 first >= 50% in shared-block-cos "
                 f"bin {first_cross('pooled8_shared', 'all8', 5)}")
    for kind in ("pooled8_private", "pooled8_shared"):
        col = 6 if kind == "pooled8_private" else 5
        notes.append(kind + ": " + " ".join(
            f"[{r[2]}:{r[col]:.3f},n={r[4]}]" for r in rows
            if r[0] == kind and r[4] >= MIN_N))
    # does the block-cosine curve depend on the scheme? per-scheme private
    # top-1 within the private-block bins
    for lo in (0.30, 0.35, 0.40, 0.45, 0.50):
        parts = []
        for name in e9.NAMES:
            r = rec[name]
            m = (r["cp"] >= lo) & (r["cp"] < lo + 0.05)
            if m.sum() >= MIN_N:
                parts.append(f"{name}:{r['hp'][m].mean():.3f}(n={m.sum()})")
        notes.append(f"private-block bin {lo:.2f}: " + " ".join(parts))
    with open(os.path.join(DATA, "e9b_notes.txt"), "w") as f:
        f.write("\n".join(notes) + "\n")
    print("\n".join(notes))
    print("E9b done.")


if __name__ == "__main__":
    main()
