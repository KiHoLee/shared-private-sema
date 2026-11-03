"""E19: tracking against a pilot-free static design, net of the pilot cost.

In E3 the static comparator DR-static keeps the transmit masks at the true
affinity and fixes only the receiver at the time average a_bar, so it is not
a design without feedback. This experiment runs the E3 scene traversal and
speed sweep in closed loop (as exp3_closed_loop.py) and adds the design that
fixes both the masks and the receiver at a_bar:

  DR-genie         masks and receiver at the true affinity
  DR-tracked       masks and receiver at the pilot-tracked estimate
                   (Algorithm 1, closed loop)
  DR-static-rx     masks at the true affinity, receiver at a_bar (E3)
  DR-static        masks and receiver at a_bar, no pilots and no feedback
  LMMSE-static     masks and LMMSE receiver at a_bar

and reports the SER per embedding together with the correct embeddings per
slot, (1 - overhead)(1 - SER), where the overhead U n_p / (K L_d) is charged
to DR-tracked only. The time average a_bar is given to the static designs
at no cost, which favors them.

Outputs: data/e19_static_masks.csv, data/e19_notes.txt
"""
import math
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp3_mobility as e3
from semantic_mac import (affinity_matrix, demux_dr, demux_lmmse,
                          sample_latents_isotropic, AffinityTracker,
                          pilot_affinity_obs, metrics, H_MIN)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

U, D, DC, BATCH, RHO, K, LAM = (e3.U, e3.D, e3.DC, e3.BATCH, e3.RHO,
                                e3.K_PILOT, e3.LAM)
NP_GRID = [8, 16, 32, 64]
SPEEDS = [0.5, 1.0, 2.0, 4.0, 8.0]
REPS_TS, REPS_SP = 5, 8
NAMES = ["DR-genie", "DR-tracked", "DR-static-rx", "DR-static",
         "LMMSE-static"]
Vc = np.eye(D)[:, :DC]


def overhead(n_p):
    return U * n_p / (K * BATCH)


def front(z, Bm, h, xi):
    """Expected-Gram front end with masks built from the coefficients of Bm."""
    t = np.einsum('uv,bvd,bv,bu->bud', Bm, z, h, 1.0 / h)
    A = np.linalg.cholesky(Bm)
    return t + np.einsum('uv,bvd->bud', A, xi) / h[:, :, None]


def run(job):
    kind, n_p, si, rep = job
    if kind == "ts":
        a_t = e3.drive_through_profile(e3.T)
        rng = np.random.default_rng(3 + rep)
    else:
        rng = np.random.default_rng(1000 + 100 * si + rep)
        a_t = e3.mobility_trajectories(U, e3.T, SPEEDS[si], rng, **e3.MOB)
    a_bar = a_t.mean(0)
    B_bar = affinity_matrix(a_bar)
    tr = AffinityTracker(U, lam=LAM, a_init=float(a_bar.mean()))
    s = {k: [] for k in NAMES}
    for t in range(a_t.shape[0]):
        a = a_t[t]
        B = affinity_matrix(a)
        z = sample_latents_isotropic(BATCH, U, D, DC, a, rng)
        hc = (rng.standard_normal((BATCH, U)) +
              1j * rng.standard_normal((BATCH, U))) / math.sqrt(2)
        h = np.clip(np.abs(hc), H_MIN, None)
        xi = rng.standard_normal((BATCH, U, D)) * math.sqrt(1.0 / RHO)
        if t % K == 0:
            zp = sample_latents_isotropic(n_p, U, D, DC, a, rng)
            hp = np.clip(np.abs((rng.standard_normal(U) +
                                 1j * rng.standard_normal(U)) / np.sqrt(2)),
                         0.2, None)
            tr.update(pilot_affinity_obs(zp, hp, RHO, rng))
        a_hat = np.clip(tr.a, 0.02, 0.95)
        Bh = affinity_matrix(a_hat)
        t_true = front(z, B, h, xi)
        t_hat = front(z, Bh, h, xi)
        t_bar = front(z, B_bar, h, xi)
        out = {
            "DR-genie": demux_dr(t_true, B, h, RHO, a, Vc),
            "DR-tracked": demux_dr(t_hat, Bh, h, RHO, a_hat, Vc),
            "DR-static-rx": demux_dr(t_true, B_bar, h, RHO, a_bar, Vc),
            "DR-static": demux_dr(t_bar, B_bar, h, RHO, a_bar, Vc),
            "LMMSE-static": demux_lmmse(t_bar, B_bar, h, RHO)[0],
        }
        for k in NAMES:
            s[k].append(metrics(out[k], z)[2])
    return job, {k: float(np.mean(v)) for k, v in s.items()}


def main():
    jobs = [("ts", n_p, 0, r) for n_p in NP_GRID for r in range(REPS_TS)]
    jobs += [("sp", n_p, si, r) for n_p in NP_GRID
             for si in range(len(SPEEDS)) for r in range(REPS_SP)]
    workers = int(os.environ.get("E19_WORKERS", "8"))
    with Pool(workers) as p:
        res = dict(p.map(run, jobs))
    rows, notes = [], []
    for n_p in NP_GRID:
        ov = overhead(n_p)
        groups = [("traversal", [res[("ts", n_p, 0, r)]
                                 for r in range(REPS_TS)])]
        groups += [(f"speed_{SPEEDS[si]}", [res[("sp", n_p, si, r)]
                                            for r in range(REPS_SP)])
                   for si in range(len(SPEEDS))]
        for name, lst in groups:
            m = {k: float(np.mean([x[k] for x in lst])) for k in NAMES}
            good = {k: (1 - (ov if k == "DR-tracked" else 0.0)) * (1 - m[k])
                    for k in NAMES}
            rows.append((n_p, ov, name, m, good))
            notes.append(f"n_p={n_p:3d} overhead={ov:6.2%} {name:10s} SER "
                         + " ".join(f"{k}={m[k]:.4f}" for k in NAMES)
                         + " | correct/slot "
                         + " ".join(f"{k}={good[k]:.4f}" for k in NAMES))
            print(notes[-1], flush=True)
    with open(os.path.join(DATA, "e19_static_masks.csv"), "w") as f:
        f.write("n_p_per_user,overhead,trace,"
                + ",".join(f"{k}_ser" for k in NAMES) + ","
                + ",".join(f"{k}_good" for k in NAMES) + "\n")
        for n_p, ov, name, m, g in rows:
            f.write(f"{n_p},{ov},{name},"
                    + ",".join(f"{m[k]:.5f}" for k in NAMES) + ","
                    + ",".join(f"{g[k]:.5f}" for k in NAMES) + "\n")
    with open(os.path.join(DATA, "e19_notes.txt"), "w") as f:
        f.write("\n".join(notes) + "\n")
    print("E19 done.")


if __name__ == "__main__":
    main()
