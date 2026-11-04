"""E24: real co-observation on MS-COCO captions, with no imposed
shared/private split (Reviewer 4, Weak Aspect 1 and Recommended Change 1).

Reviewer 4 asked for validation on real multi-device data in which the
shared/private decomposition is not imposed. E12 (exp12_corpus.py) is the
only such evidence so far, and its receivers are given share coefficients
fitted to the clean Gram of the evaluation tuples in each affinity bin,
which is partly oracle. E24 replaces the corpus construction by real
independent observations of one scene and removes the oracle.

Construction. Every MS-COCO image is one scene. Its captions were written
independently by different annotators, so they are independent
descriptions of the same scene. U = 4 of the captions of each image (a
seeded choice, in a seeded random order) are the 4 users' observations,
and each user transmits the unmodified BERT embedding of its caption.
Nothing is split, no common vector is injected, no share coefficient is
set. Embeddings come from code/extract_coco_captions.py (bert-base-uncased,
last layer, mean pooling, truncation 64) and are centered with the mean of
the calibration captions and normalized to unit norm, the
semantic_mac.EmbeddingPool convention with the mean taken from calibration
data only.

Partitions (disjoint images)
  calibration  the first 1,600 images of the seeded train2017 subset,
               N = 200 (first 200) and N = 1,600 tuples
  eval "val"   all 5,000 val2017 images, 5,000 tuples, retrieval bank of
               all 25,014 val2017 captions
  eval "tr20k" the next 20,000 train2017 images, 20,000 tuples, retrieval
               bank of all 100,000+ captions of those images

Side information, applied identically at the transmitter (signature
correlations B = affinity_matrix(a)) and at every affinity-aware receiver
  static   (proposed use) rank-one fit (exp12_corpus.rank_one_fit) to the
           mean clean Gram of the N calibration tuples, one coefficient
           vector a for every tuple. No evaluation data are used.
  oracle   (E12 style, for comparison) the measured mean pairwise cosine of
           each evaluation tuple is quantized to 0.05 in [0.05, 0.95], and
           the rank-one fit is made to the mean clean Gram of the
           EVALUATION tuples of each bin. Bins with fewer than 20 tuples are
           merged into the nearest bin with at least 20, so that every
           tuple is evaluated in both modes. The basis and every other
           calibration quantity still come from calibration data.

Receivers (Rayleigh fading, deep-fade clip H_MIN, matched-filter front
end of semantic_mac, SNR 10, 20, 30 dB in the per-coordinate convention
of E9/E12)
  OMA, NOMA     conventional orthogonal access and genie-aided NOMA-SIC
                (no affinity used)
  SR, SC        closed-form endpoints of the affinity-aware family
  LMMSE         affinity-aware, covariance-blind (Proposition 1)
  LMMSE-fullcov linear MMSE of the stacked U d model with the own and
                cross covariance of the calibration tuples
                (exp20_covariance_receiver.demux_fullcov), no basis
  DR-spec       decomposition receiver with the spectral basis estimated
                from the calibration tuples
  DR-adapt      same with the channel-trained adapter basis
                (exp2_structure.train_adapter on the calibration tuples)
  DR-MMSE       subspace-aware block LMMSE (exp6_sideinfo) with the
                spectral basis
  DR-cov        covariance-aware DR (exp20_covariance_receiver.demux_cov)
                with the spectral basis and block covariances from the
                calibration tuples
d_c in {32, 64, 128, 256} for the DR family.

Metrics. Mean cosine, NMSE, SER at tau = 0.45, exact top-1 caption
retrieval among all captions of the evaluation partition (a hit also when
the retrieved caption has identical text, since COCO contains duplicate
captions), and scene retrieval (the top-1 caption belongs to the image of
the transmitted caption). Standard errors are over tuples. The run also
reports the measured affinity statistics, the fitted share coefficients,
the eigen-spectrum of the spectral estimator, and how closely the
captions follow the structured model inside and outside the estimated
shared subspace.

Seeds. Caption choice default_rng(24). Channel and noise drawn once per
evaluation partition with default_rng(2400 + pool) in the draw order of
semantic_mac.matched_filter and reused (scaled) at every SNR, N, mode,
d_c and receiver (common random numbers). OMA/NOMA default_rng(97000 +
1000 pool + snr). Adapter default_rng(700 + d_c + N), torch seed 0.

Source and license. MS-COCO 2017 caption annotations (T.-Y. Lin et al.,
ECCV 2014, https://cocodataset.org), Creative Commons Attribution 4.0.

Run on u24 with CUDA for LMMSE-fullcov and retrieval:
    python code/exp24_coco_captions.py --coco ~/tmc_rev/data_ext/coco
Outputs: data/e24_coco_captions.csv, data/e24_notes.txt
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp2_structure as E2
from semantic_mac import (TAU, H_MIN, affinity_matrix, demux_sr, demux_sc,
                          demux_lmmse, demux_dr, learn_structure_spectral,
                          oma_observe, demux_noma_genie)
from exp6_sideinfo import demux_block_lmmse
from exp12_corpus import rank_one_fit
from exp20_covariance_receiver import (block_eig, demux_cov, fullcov_prior,
                                       demux_fullcov)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

U, D = 4, 768
N_CAL_IMG = 1600
N_CALS = [200, 1600]
SNRS = [10, 20, 30]
DC_GRID = [32, 64, 128, 256]
MODES = ["static", "oracle"]
SEED_TUP = 24
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
#  data
# ---------------------------------------------------------------------
def norm_text(t):
    return " ".join(t.lower().strip().rstrip(".").split())


def make_tuples(image_id, n_img_rng, images):
    """Caption indices (n_images, U): a seeded choice of U captions of each
    image, in random order."""
    pos = {}
    for k, i in enumerate(image_id):
        pos.setdefault(int(i), []).append(k)
    out = np.empty((len(images), U), dtype=int)
    for r, i in enumerate(images):
        c = pos[int(i)]
        out[r] = np.asarray(c)[n_img_rng.choice(len(c), size=U,
                                                replace=False)]
    return out


def load(coco):
    tr = torch.load(os.path.join(coco, "coco_bert_train21600.pt"),
                    weights_only=False)
    va = torch.load(os.path.join(coco, "coco_bert_val2017.pt"),
                    weights_only=False)
    return tr, va


# ---------------------------------------------------------------------
#  channel front end: semantic_mac.matched_filter with pre-drawn h, xi
# ---------------------------------------------------------------------
def draw_channel(batch, rng):
    hc = (rng.standard_normal((batch, U)) +
          1j * rng.standard_normal((batch, U))) / math.sqrt(2)
    h = np.clip(np.abs(hc), H_MIN, None)
    xi = rng.standard_normal((batch, U, D))
    return h, xi


def front_end(e, Bb, h, xi, rho):
    """Identical to matched_filter(e, B, rho, rng) given the same draws,
    with a per-tuple B (batch, U, U)."""
    s2 = 1.0 / rho
    tilde = np.einsum('buv,bvd,bv,bu->bud', Bb, e, h, 1.0 / h)
    L = np.linalg.cholesky(Bb)
    tilde += np.einsum('buv,bvd->bud', L, xi * math.sqrt(s2)) / h[:, :, None]
    return tilde


# ---------------------------------------------------------------------
#  metrics
# ---------------------------------------------------------------------
class Scorer:
    def __init__(self, Zbank, img_bank, txt_bank, truth_pos):
        self.bank = torch.as_tensor(Zbank, dtype=torch.float32, device=DEV)
        self.img = torch.as_tensor(img_bank, device=DEV)
        self.txt = torch.as_tensor(txt_bank, device=DEV)
        self.truth = torch.as_tensor(truth_pos.ravel(), device=DEV)
        self.shape = truth_pos.shape

    def retrieve(self, est):
        q = torch.as_tensor(est.reshape(-1, D), dtype=torch.float32,
                            device=DEV)
        q = torch.nn.functional.normalize(q, dim=1)
        top = torch.empty(len(q), dtype=torch.long, device=DEV)
        for s in range(0, len(q), 4096):
            top[s:s + 4096] = (q[s:s + 4096] @ self.bank.T).argmax(dim=1)
        exact = (self.txt[top] == self.txt[self.truth]).cpu().numpy()
        scene = (self.img[top] == self.img[self.truth]).cpu().numpy()
        return exact.reshape(self.shape), scene.reshape(self.shape)


def score(est, x, scorer):
    en = est / (np.linalg.norm(est, axis=2, keepdims=True) + 1e-12)
    cos = (en * x).sum(-1)                         # (n, U)
    nmse = ((est - x) ** 2).sum(-1)
    exact, scene = scorer.retrieve(est)
    out = {}
    n = len(x)
    for k, v in (("cos", cos), ("nmse", nmse), ("ser", cos < TAU),
                 ("top1", exact), ("scene", scene)):
        t = v.astype(float).mean(axis=1)            # per-tuple mean
        out[k] = float(t.mean())
        out[k + "_se"] = float(t.std(ddof=1) / math.sqrt(n))
    return out


def pct(v, ps=(5, 10, 25, 50, 75, 90, 95)):
    return " ".join(f"p{p}={np.percentile(v, p):.3f}" for p in ps)


# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True)
    ap.add_argument("--pools", default="val,tr20k")
    ap.add_argument("--fullcov-pools", default="val,tr20k")
    ap.add_argument("--out-tag", default="")
    ap.add_argument("--out-dir", default=DATA)
    ap.add_argument("--max-tuples", type=int, default=0,
                    help="smoke test only: keep the first tuples of a pool")
    args = ap.parse_args()
    t0 = time.time()
    notes = []

    def log(s):
        notes.append(s)
        print(s, flush=True)

    tr, va = load(args.coco)
    Xtr = tr["emb"].numpy().astype(np.float64)
    Xva = va["emb"].numpy().astype(np.float64)
    order = np.asarray(tr["image_order"])
    rng_t = np.random.default_rng(SEED_TUP)
    cal_imgs = order[:N_CAL_IMG]
    ev_imgs = order[N_CAL_IMG:]
    tup_cal = make_tuples(tr["image_id"], rng_t, cal_imgs)
    tup_tr = make_tuples(tr["image_id"], rng_t, ev_imgs)
    va_imgs = np.array(sorted(set(int(i) for i in va["image_id"])))
    tup_va = make_tuples(va["image_id"], rng_t, va_imgs)

    # normalization: center with the mean of the calibration captions
    cal_rows = np.isin(tr["image_id"], cal_imgs)
    mu = Xtr[cal_rows].mean(axis=0, keepdims=True)

    def normalize(X):
        Y = X - mu
        return Y / np.linalg.norm(Y, axis=1, keepdims=True)

    Ztr, Zva = normalize(Xtr), normalize(Xva)
    log(f"E24 MS-COCO captions, U={U}, D={D}, calibration images "
        f"{N_CAL_IMG} (N in {N_CALS}), centering mean from the "
        f"{int(cal_rows.sum())} calibration captions")

    # text ids for duplicate-aware exact retrieval
    vocab = {}
    txt_tr = np.array([vocab.setdefault(norm_text(t), len(vocab))
                       for t in tr["text"]])
    txt_va = np.array([vocab.setdefault(norm_text(t), len(vocab))
                       for t in va["text"]])
    img_tr = np.asarray(tr["image_id"])
    img_va = np.asarray(va["image_id"])
    ev_rows_tr = np.where(~cal_rows)[0]
    pools = {
        "val": dict(x=Zva[tup_va], tup=tup_va, bank=np.arange(len(Zva)),
                    Z=Zva, img=img_va, txt=txt_va),
        "tr20k": dict(x=Ztr[tup_tr], tup=tup_tr, bank=ev_rows_tr,
                      Z=Ztr, img=img_tr, txt=txt_tr),
    }
    for name, P in pools.items():
        bt = P["txt"][P["bank"]]
        dup = 1.0 - len(np.unique(bt)) / len(bt)
        log(f"pool {name}: {len(P['tup'])} tuples ({len(P['tup'])*U} "
            f"transmitted captions), bank {len(P['bank'])} captions, "
            f"duplicate caption texts in bank {100*dup:.2f}%")

    x_cal_all = Ztr[tup_cal]

    # ---------------- affinity statistics --------------------------------
    iu = np.triu_indices(U, 1)
    rng_ref = np.random.default_rng(99)
    for name, x in (("calib", x_cal_all), ("val", pools["val"]["x"]),
                    ("tr20k", pools["tr20k"]["x"])):
        G = np.einsum('nud,nvd->nuv', x, x)
        pw = G[:, iu[0], iu[1]].ravel()
        bt = G[:, iu[0], iu[1]].mean(axis=1)
        log(f"[affinity {name}] pairwise caption cosine, same image: mean "
            f"{pw.mean():.4f} sd {pw.std():.4f} {pct(pw)}")
        log(f"[affinity {name}] tuple-mean affinity beta: mean "
            f"{bt.mean():.4f} sd {bt.std():.4f} {pct(bt)}")
        # reference: captions of different images
        xf = x.reshape(-1, D)
        i1 = rng_ref.integers(len(xf), size=20000)
        i2 = rng_ref.integers(len(xf), size=20000)
        keep = (i1 // U) != (i2 // U)
        rc = (xf[i1[keep]] * xf[i2[keep]]).sum(1)
        log(f"[affinity {name}] reference, captions of different images: "
            f"mean {rc.mean():.4f} sd {rc.std():.4f} {pct(rc)}")
        hist, edges = np.histogram(bt, bins=np.arange(-0.1, 1.0001, 0.05))
        log(f"[affinity {name}] histogram of beta, bins of 0.05 from -0.10: "
            + " ".join(str(int(c)) for c in hist))

    a_cal, G_cal = {}, {}
    for N in N_CALS:
        xc = x_cal_all[:N]
        G_cal[N] = np.einsum('nud,nvd->nuv', xc, xc).mean(axis=0)
        a_cal[N] = rank_one_fit(G_cal[N])
        log(f"[share coef] static rank-one fit, N={N}: a = "
            + " ".join(f"{v:.4f}" for v in a_cal[N])
            + f", implied B_uv mean {np.mean(np.outer(a_cal[N], a_cal[N])[iu]):.4f}"
            + f", calibration Gram off-diagonal mean "
              f"{G_cal[N][iu].mean():.4f}")

    # ---------------- eigen-spectrum of the spectral estimator ----------
    for tag, xs in (("calib N=200", x_cal_all[:200]),
                    ("calib N=1600", x_cal_all),
                    ("tr20k (diagnostic only)", pools["tr20k"]["x"])):
        _, w = learn_structure_spectral(xs, 8)
        pos = w[w > 0]
        cum = np.cumsum(pos) / pos.sum()
        ks = {f: int(np.searchsorted(cum, f) + 1) for f in
              (0.5, 0.8, 0.9, 0.95)}
        rel = {r: int((w >= r * w[0]).sum()) for r in (0.1, 0.05, 0.01)}
        log(f"[spectrum {tag}] trace {w.sum():.4f}, positive eigenvalues "
            f"{len(pos)}, top-10 " + " ".join(f"{v:.4f}" for v in w[:10]))
        log(f"[spectrum {tag}] fraction of positive mass in top k: "
            + " ".join(f"k={k}:{cum[min(k, len(cum))-1]:.3f}" for k in
                       (8, 16, 32, 64, 128, 256))
            + "; k for 50/80/90/95%: "
            + " ".join(f"{int(100*f)}%:{k}" for f, k in ks.items())
            + "; eigenvalues above 10/5/1% of the largest: "
            + " ".join(f"{int(100*r)}%:{k}" for r, k in rel.items()))
        log(f"[spectrum {tag}] eigenvalues at k=1,2,4,8,16,32,64,128,256,"
            f"512: " + " ".join(f"{w[k-1]:.5f}" for k in
                                (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)))

    # ---------------- calibration artifacts ------------------------------
    cal = {}
    for N in N_CALS:
        xc = x_cal_all[:N]
        a = a_cal[N]
        sig_fc = fullcov_prior(xc)
        for dc in DC_GRID:
            Vs, _ = learn_structure_spectral(xc, dc)
            rng_q = np.random.default_rng(700 + dc + N)
            Q, _ = np.linalg.qr(np.hstack([Vs, rng_q.standard_normal(
                (D, D - dc))]))
            E2.DC = dc                     # train_adapter reads the global
            t1 = time.time()
            W = E2.train_adapter(xc, a, Q, rng_q)
            Va = W.T[:, :dc]
            eig = block_eig(xc, Vs, Q[:, dc:], a)
            cal[(N, dc)] = dict(Vs=Vs, Va=Va, eig=eig)
            # model-fit diagnostic on the val tuples
            xv = pools["val"]["x"]
            ys = xv @ Vs
            yp = xv - ys @ Vs.T
            es = (ys ** 2).sum(-1).mean()
            ysn = ys / (np.linalg.norm(ys, axis=2, keepdims=True) + 1e-12)
            al = np.einsum('nud,nvd->nuv', ysn, ysn)[:, iu[0], iu[1]].mean()
            cin = np.einsum('nud,nvd->nuv', ys, ys)[:, iu[0], iu[1]].mean()
            cout = np.einsum('nud,nvd->nuv', yp, yp)[:, iu[0], iu[1]].mean()
            log(f"[model fit N={N} d_c={dc}] val: shared-block energy "
                f"{es:.4f} (model a^2 = {np.mean(a**2):.4f}), cosine "
                f"between users' shared blocks {al:.4f} (model 1), cross "
                f"inner product inside {cin:.4f} / outside {cout:.4f} "
                f"(model outside 0), adapter {time.time()-t1:.0f} s")
        cal[N] = dict(a=a, sig_fc=sig_fc)

    # ---------------- evaluation ----------------------------------------
    rows = []
    for pid, pname in enumerate(args.pools.split(",")):
        P = pools[pname]
        if args.max_tuples:
            P["x"] = P["x"][:args.max_tuples]
            P["tup"] = P["tup"][:args.max_tuples]
        x = P["x"]
        n = len(x)
        bank_pos = {g: i for i, g in enumerate(P["bank"])}
        truth = np.vectorize(bank_pos.get)(P["tup"])
        scorer = Scorer(P["Z"][P["bank"]], P["img"][P["bank"]],
                        P["txt"][P["bank"]], truth)
        # oracle bins (E12 style), fitted on the evaluation Gram
        G = np.einsum('nud,nvd->nuv', x, x)
        beta = G[:, iu[0], iu[1]].mean(axis=1)
        q = np.clip(np.round(beta / 0.05) * 0.05, 0.05, 0.95).round(2)
        vals, cnt = np.unique(q, return_counts=True)
        ok = vals[cnt >= 20]
        q = ok[np.abs(q[:, None] - ok[None, :]).argmin(axis=1)]
        bins = {}
        for qb in np.unique(q):
            m = np.where(q == qb)[0]
            bins[qb] = (m, rank_one_fit(G[m].mean(axis=0)))
        log(f"[oracle bins {pname}] " + " ".join(
            f"{qb:.2f}:{len(m)}:a={a.mean():.3f}" for qb, (m, a) in
            bins.items()))

        h, xi = draw_channel(n, np.random.default_rng(2400 + pid))
        fc_ok = pname in args.fullcov_pools.split(",")
        for snr in SNRS:
            rho = 10 ** (snr / 10)
            rng_c = np.random.default_rng(97000 + 1000 * pid + snr)
            for nm, est in (("OMA", oma_observe(x, h, rho, rng_c)),
                            ("NOMA", demux_noma_genie(x, h, rho, rng_c))):
                r = score(est, x, scorer)
                rows.append(dict(pool=pname, n_cal=0, mode="none", snr=snr,
                                 d_c=0, scheme=nm, n=n, **r))
            for N in N_CALS:
                for mode in MODES:
                    if mode == "static":
                        groups = [(np.arange(n), cal[N]["a"])]
                    else:
                        groups = list(bins.values())
                    Bb = np.empty((n, U, U))
                    for m, a in groups:
                        Bb[m] = affinity_matrix(a)
                    tilde = front_end(x, Bb, h, xi, rho)

                    def run(fn):
                        out = np.empty_like(x)
                        for m, a in groups:
                            out[m] = fn(tilde[m], affinity_matrix(a), h[m],
                                        a)
                        return out

                    rec = [("SR", 0, lambda t, B, hh, a: demux_sr(t, B, hh)),
                           ("SC", 0, lambda t, B, hh, a: demux_sc(t, hh)),
                           ("LMMSE", 0, lambda t, B, hh, a:
                            demux_lmmse(t, B, hh, rho)[0])]
                    if fc_ok:
                        sig = cal[N]["sig_fc"]
                        rec.append(("LMMSE-fullcov", 0, lambda t, B, hh, a:
                                    demux_fullcov(t, B, hh, rho, sig)))
                    for dc in DC_GRID:
                        C = cal[(N, dc)]
                        rec += [
                            ("DR-spec", dc, lambda t, B, hh, a, C=C:
                             demux_dr(t, B, hh, rho, a, C["Vs"])),
                            ("DR-adapt", dc, lambda t, B, hh, a, C=C:
                             demux_dr(t, B, hh, rho, a, C["Va"])),
                            ("DR-MMSE", dc, lambda t, B, hh, a, C=C:
                             demux_block_lmmse(t, B, hh, rho, a, C["Vs"])),
                            ("DR-cov", dc, lambda t, B, hh, a, C=C:
                             demux_cov(t, B, hh, rho, a, *C["eig"])),
                        ]
                    for nm, dc, fn in rec:
                        t1 = time.time()
                        r = score(run(fn), x, scorer)
                        rows.append(dict(pool=pname, n_cal=N, mode=mode,
                                         snr=snr, d_c=dc, scheme=nm, n=n,
                                         **r))
                        print(f"  {pname} snr={snr} N={N} {mode} {nm} "
                              f"dc={dc} cos={r['cos']:.4f} ser={r['ser']:.4f}"
                              f" top1={r['top1']:.4f} scene={r['scene']:.4f}"
                              f" ({time.time()-t1:.1f} s, total "
                              f"{(time.time()-t0)/60:.1f} min)", flush=True)
            write(rows, notes, args.out_dir, args.out_tag)
    log(f"total run time {(time.time()-t0)/60:.1f} min")
    write(rows, notes, args.out_dir, args.out_tag)
    print("E24 done.")


KEYS = ["pool", "n_cal", "mode", "snr", "d_c", "scheme", "n", "cos",
        "cos_se", "nmse", "nmse_se", "ser", "ser_se", "top1", "top1_se",
        "scene", "scene_se"]


def write(rows, notes, out_dir, tag):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"e24_coco_captions{tag}.csv"),
              "w") as f:
        f.write(",".join(KEYS) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in KEYS) + "\n")
    with open(os.path.join(out_dir, f"e24_notes{tag}.txt"), "w") as f:
        f.write("\n".join(notes) + "\n")


if __name__ == "__main__":
    main()
