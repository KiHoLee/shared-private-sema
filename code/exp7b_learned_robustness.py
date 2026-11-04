"""E7b: robustness of the learned-receiver comparison of E7
(Reviewers 1.1b and 3.4).

Purpose
-------
E7 reports that the DR has a lower SER than the side-informed attention
receiver (Learned+SI) by 2.1 to 2.5 times at 0.5 <= beta <= 0.7. A
reviewer can object that (i) the network is fed the matched-filter
outputs after the division by h_u, which rescales every token by its own
gain, (ii) the 3000-step budget may be short, and (iii) the network is
small (14,787 parameters). This script tests the three objections with
the E7 construction left unchanged otherwise.

Arms (all with side information, i.e. the affinity and the channel gains
enter through the E7 context encoder and the attention-score biases)
-----------------------------------------------------------------------
  tokens  div     the published front end, tilde_u = M_u^T y / h_u
          prediv  tokens taken before the 1/h_u division, h_u tilde_u
                  = M_u^T y, with the same side-information inputs
  arch    base    the E7 network (dk=16, 4 heads), 14,787 parameters
          wide    twice the width (dk=32, 4 heads, inner width 128)
          deep    two stacked side-informed attention layers of the E7
                  type (dk=16 each), the second reading the output of
                  the first
  steps   3000    the E7 budget (snapshot of the 6000-step run; the
                  training stream and the optimizer are identical up to
                  that step, so it equals a 3000-step run)
          6000    twice the E7 budget

Arm (a) of the review request is tokens=div, arch=base, steps=3000, the
published Learned+SI, and must reproduce data/e7_learned_sideinfo.csv
within seed noise. Arm (b) is prediv/base. Arm (c) is the 6000-step
snapshot of (a) and (b). Arm (d) is wide and deep, with both front ends.

Setting
-------
Identical to exp7_learned_sideinfo.py: U=4, d=64, d_c=16, rho=10 dB,
Rayleigh fading with the H_MIN clip, batch 64, Adam lr 1e-3, beta drawn
uniformly from [0.05, 0.9] per training step, cosine-distortion loss.
Seeds 0 to 4, torch.manual_seed(seed), training RNG
default_rng(1000 + seed). Evaluation at all ten E7 affinities with
EVAL_BATCH=40000 draws (1.6e5 decisions) from default_rng(500 +
int(1000 beta)), the E7 evaluation sets, so the DR, DR-MMSE (the
subspace-aware LMMSE, CSV key BLMMSE in E7) and LMMSE rows reproduce the
published E7 values exactly. A held-out objective (400 draws per E7
affinity from default_rng(777), as in exp7_valcurve.py) is logged every
250 steps.

GPU when available (u24, GPU 1), CPU fallback.

Outputs
-------
  data/e7b_learned_robustness.csv  one row per (arm, beta): SER and cosine
                                   mean and SD over the five seeds, the
                                   per-seed SER, the DR, DR-MMSE and
                                   LMMSE SER on the same evaluation set,
                                   and the ratios to DR and DR-MMSE
  data/e7b_valcurve.csv            held-out objective vs step per arm, seed
  data/e7b_notes.txt               reproduction check and key lines
"""
import argparse
import copy
import math
import os
import time

import numpy as np
import torch

import exp7_learned_sideinfo as e7
from exp7_learned_sideinfo import (UserWiseAttention, gen_batch, U, D, DC,
                                   RHO, BATCH, SEEDS, BETAS, EVAL_BATCH)
from semantic_mac import metrics, demux_lmmse, demux_dr, DEVICE
from exp6_sideinfo import demux_block_lmmse

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
os.makedirs(DATA, exist_ok=True)

STEPS_SNAP = e7.STEPS          # 3000, the E7 budget
STEPS_LONG = 2 * e7.STEPS      # 6000
VAL_EVERY = 250
TOKENS = ("div", "prediv")
ARCHS = ("base", "wide", "deep")


class TwoLayerSI(torch.nn.Module):
    """Two stacked side-informed attention layers of the E7 type. Each
    layer is the E7 UserWiseAttention (residual update, unit-norm
    output), the second reads the first layer's per-user estimates as
    its tokens, and both receive the affinity and the gains."""

    def __init__(self):
        super().__init__()
        self.l1 = UserWiseAttention(U, D, side_info=True)
        self.l2 = UserWiseAttention(U, D, side_info=True)

    def forward(self, tilde, beta, h):
        return self.l2(self.l1(tilde, beta, h), beta, h)


def make_net(arch):
    if arch == "base":
        return UserWiseAttention(U, D, side_info=True)
    if arch == "wide":
        return UserWiseAttention(U, D, dk=32, side_info=True)
    if arch == "deep":
        return TwoLayerSI()
    raise ValueError(arch)


def to_tokens(tilde, h, mode):
    """div: the published tokens. prediv: M_u^T y, before the 1/h_u
    division (matched_filter returns tilde = M_u^T y / h_u)."""
    return tilde if mode == "div" else tilde * h[:, :, None]


def tens(x):
    return torch.tensor(x, dtype=torch.float32, device=DEVICE)


def build_val():
    rng_v = np.random.default_rng(777)
    val = []
    for beta in BETAS:
        z, tilde, h, _, _ = gen_batch(rng_v, 400, beta)
        val.append((beta, z, tilde, h))
    return val


def val_loss(net, val, mode):
    out = []
    with torch.no_grad():
        for beta, z, tilde, h in val:
            o = net(tens(to_tokens(tilde, h, mode)),
                    torch.full((len(z),), beta, dtype=torch.float32,
                               device=DEVICE), tens(h))
            out.append(float((1.0 - (o * tens(z)).sum(-1)).mean()))
    return float(np.mean(out)), out[BETAS.index(0.5)]


def train(mode, arch, seed, steps, val, vrows):
    """exp7_learned_sideinfo.train(side_info=True, seed) with a token
    transform, a selectable architecture, a snapshot at STEPS_SNAP and
    a held-out objective. The training stream is the E7 stream."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(1000 + seed)
    net = make_net(arch).to(DEVICE)
    n_par = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    snaps = {}
    for it in range(steps):
        beta = float(rng.uniform(0.05, 0.9))
        z, tilde, h, _, _ = gen_batch(rng, BATCH, beta)
        out = net(tens(to_tokens(tilde, h, mode)),
                  torch.full((BATCH,), beta, dtype=torch.float32,
                             device=DEVICE), tens(h))
        loss = (1.0 - (out * tens(z)).sum(-1)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (it + 1) % VAL_EVERY == 0:
            net.eval()
            va, vb = val_loss(net, val, mode)
            net.train()
            vrows.append((mode, arch, seed, it + 1, va, vb))
        if (it + 1) in (STEPS_SNAP, steps):
            snaps[it + 1] = copy.deepcopy(net.state_dict())
    return snaps, n_par


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: one seed, 500 steps, no output files")
    args = ap.parse_args()
    global STEPS_SNAP
    seeds = SEEDS[:1] if args.quick else SEEDS
    steps = 500 if args.quick else STEPS_LONG
    if args.quick:
        STEPS_SNAP = 250
    snap_steps = (STEPS_SNAP, steps)
    print(f"device={DEVICE} seeds={seeds} steps={snap_steps}", flush=True)

    val = build_val()
    vrows = []
    states, npar = {}, {}
    t0 = time.time()
    for mode in TOKENS:
        for arch in ARCHS:
            for seed in seeds:
                snaps, n_par = train(mode, arch, seed, steps, val, vrows)
                npar[arch] = n_par
                for st in snap_steps:
                    states[(mode, arch, st, seed)] = snaps[st]
                print(f"trained tokens={mode} arch={arch} seed={seed} "
                      f"params={n_par} final_val={vrows[-1][4]:.5f} "
                      f"t={time.time() - t0:.0f}s", flush=True)

    # evaluation on the E7 evaluation sets
    Vc = np.eye(D)[:, :DC]
    arms = [(m, a, st) for m in TOKENS for a in ARCHS for st in snap_steps]
    res, refs = {}, {}
    for beta in BETAS:
        rng_e = np.random.default_rng(500 + int(beta * 1000))
        z, tilde, h, a, B = gen_batch(rng_e, EVAL_BATCH, beta)
        lm, _ = demux_lmmse(tilde, B, h, RHO)
        refs[beta] = {
            "LMMSE": metrics(lm, z),
            "DRMMSE": metrics(demux_block_lmmse(tilde, B, h, RHO, a, Vc), z),
            "DR": metrics(demux_dr(tilde, B, h, RHO, a, Vc), z)}
        ht = tens(h)
        bt = torch.full((EVAL_BATCH,), beta, dtype=torch.float32,
                        device=DEVICE)
        tok = {m: tens(to_tokens(tilde, h, m)) for m in TOKENS}
        for (m, ar, st) in arms:
            sers, coss = [], []
            for seed in seeds:
                net = make_net(ar).to(DEVICE)
                net.load_state_dict(states[(m, ar, st, seed)])
                net.eval()
                with torch.no_grad():
                    o = net(tok[m], bt, ht)
                c, _, s = metrics(o.cpu().numpy().astype(np.float64), z)
                sers.append(s)
                coss.append(c)
            res[(m, ar, st, beta)] = (sers, coss)
        print(f"beta={beta:.2f} DR={refs[beta]['DR'][2]:.5f} "
              f"DRMMSE={refs[beta]['DRMMSE'][2]:.5f} " + " ".join(
                  f"{m}/{ar}/{st}={np.mean(res[(m, ar, st, beta)][0]):.5f}"
                  for (m, ar, st) in arms), flush=True)

    if args.quick:
        print("quick run done, nothing written")
        return

    # CSV
    hdr = ["arm", "tokens", "arch", "steps", "n_params", "beta",
           "ser_mean", "ser_sd", "cos_mean", "cos_sd"] + \
          [f"ser_seed{s}" for s in seeds] + \
          ["DR_ser", "DRMMSE_ser", "LMMSE_ser", "ratio_to_DR",
           "ratio_to_DRMMSE"]
    lines = [",".join(hdr)]
    for (m, ar, st) in arms:
        for beta in BETAS:
            sers, coss = res[(m, ar, st, beta)]
            r = refs[beta]
            sm = float(np.mean(sers))
            lines.append(",".join(str(x) for x in (
                [f"LearnedSI_{m}_{ar}_{st}", m, ar, st, npar[ar], beta, sm,
                 float(np.std(sers)), float(np.mean(coss)),
                 float(np.std(coss))] + sers +
                [r["DR"][2], r["DRMMSE"][2], r["LMMSE"][2],
                 sm / r["DR"][2], sm / r["DRMMSE"][2]])))
    with open(os.path.join(DATA, "e7b_learned_robustness.csv"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(DATA, "e7b_valcurve.csv"), "w") as f:
        f.write("tokens,arch,seed,step,val_loss_all,val_loss_b05\n")
        for v in vrows:
            f.write(f"{v[0]},{v[1]},{v[2]},{v[3]},{v[4]:.6f},{v[5]:.6f}\n")

    # notes: reproduction check against the published E7 CSV
    pub = np.genfromtxt(os.path.join(DATA, "e7_learned_sideinfo.csv"),
                        delimiter=",", names=True)
    notes = [f"E7b learned-receiver robustness, device={DEVICE}, "
             f"seeds {seeds}, eval {EVAL_BATCH} draws x {U} users per beta",
             f"parameters: " + ", ".join(f"{k}={v}" for k, v in npar.items()),
             "", "Reproduction check (arm a = div/base/3000 vs published "
             "LearnedSI; references vs published DR, BLMMSE, LMMSE)"]
    worst_ref = 0.0
    for i, beta in enumerate(BETAS):
        sers, _ = res[("div", "base", STEPS_SNAP, beta)]
        d = np.mean(sers) - pub["LearnedSI_ser"][i]
        sd = max(pub["LearnedSI_ser_sd"][i], np.std(sers), 1e-12)
        for k, kp in (("DR", "DR_ser"), ("DRMMSE", "BLMMSE_ser"),
                      ("LMMSE", "LMMSE_ser")):
            worst_ref = max(worst_ref, abs(refs[beta][k][2] - pub[kp][i]))
        notes.append(f"  beta={beta:.2f} published {pub['LearnedSI_ser'][i]:.5f}"
                     f" (sd {pub['LearnedSI_ser_sd'][i]:.5f}) rerun "
                     f"{np.mean(sers):.5f} (sd {np.std(sers):.5f}) diff "
                     f"{d:+.5f} = {d / sd:+.2f} sd")
    notes.append(f"  max |reference SER - published| over beta and DR, "
                 f"DR-MMSE, LMMSE = {worst_ref:.2e}")
    notes += ["", "SER mean (sd over seeds) and ratio to DR / to DR-MMSE"]
    for beta in BETAS:
        r = refs[beta]
        notes.append(f"beta={beta:.2f}  DR {r['DR'][2]:.5f}  DR-MMSE "
                     f"{r['DRMMSE'][2]:.5f}  LMMSE {r['LMMSE'][2]:.5f}")
        best = None
        for (m, ar, st) in arms:
            sers, _ = res[(m, ar, st, beta)]
            sm = float(np.mean(sers))
            if best is None or sm < best[0]:
                best = (sm, m, ar, st)
            notes.append(f"    {m:6s} {ar:4s} {st:5d}  {sm:.5f} "
                         f"({np.std(sers):.5f})  x{sm / r['DR'][2]:.3f} DR  "
                         f"x{sm / r['DRMMSE'][2]:.3f} DR-MMSE")
        notes.append(f"    best arm {best[1]}/{best[2]}/{best[3]}: "
                     f"{best[0]:.5f}, x{best[0] / r['DR'][2]:.3f} DR, "
                     f"x{best[0] / r['DRMMSE'][2]:.3f} DR-MMSE")
    notes += ["", "Held-out objective (mean over seeds) at 3000 and 6000 "
              "steps"]
    for m in TOKENS:
        for ar in ARCHS:
            for st in snap_steps:
                v = [x[4] for x in vrows if x[0] == m and x[1] == ar and
                     x[3] == st]
                notes.append(f"  {m:6s} {ar:4s} step {st:5d} "
                             f"val_all {np.mean(v):.5f}")
    notes.append(f"runtime {time.time() - t0:.0f} s")
    with open(os.path.join(DATA, "e7b_notes.txt"), "w") as f:
        f.write("\n".join(notes) + "\n")
    print("\n".join(notes))
    print("E7b done.")


if __name__ == "__main__":
    main()
