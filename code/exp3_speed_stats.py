"""E3 speed sweep: per-seed spread and visited affinity.

data/e3_speed.csv stores only the seed means of the E3 speed sweep. This
script reruns the same sweep with the same seeds (exp3_mobility.py, seed
1000 + 100 * speed index + rep, eight trajectory seeds per speed) and keeps
every seed, so that the seed standard error of each receiver's SER and the
affinity each trajectory visits can be reported. The visited affinity is
the pairwise affinity a_u(t) a_v(t) averaged over the 300 slots and the six
user pairs. Trajectories are drawn independently per speed, so the speeds
differ in the affinities they visit as well as in their speed.

The seed means reproduce data/e3_speed.csv, which the script asserts.

The summary adds the paired gap of DR-tracked to DR-genie and a least-squares
fit of each SER on the visited affinity and log2 of the speed over the 40
trajectories. Run with the argument "summary" to recompute it from the
stored seeds without simulating.

Outputs: data/e3_speed_seeds.csv, data/e3_speed_stats.txt
"""
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp3_mobility as e3
from semantic_mac import mobility_trajectories

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
SPEEDS = [0.5, 1.0, 2.0, 4.0, 8.0]
REPS = 8
SHOW = ["DR-genie", "DR-tracked", "DR-static", "LMMSE"]


def pair_affinity(a_t):
    """Pairwise affinity a_u a_v averaged over slots and pairs u < v."""
    iu = np.triu_indices(a_t.shape[1], k=1)
    return float(np.mean([np.outer(a, a)[iu].mean() for a in a_t]))


def simulate():
    rows = []
    for si, sp in enumerate(SPEEDS):
        for rep in range(REPS):
            rng = np.random.default_rng(1000 + 100 * si + rep)
            a_t = mobility_trajectories(e3.U, e3.T, sp, rng, **e3.MOB)
            r = e3.run_trace(a_t, rng)
            rec = dict(speed=sp, seed=1000 + 100 * si + rep,
                       pair_affinity=pair_affinity(a_t))
            rec.update({f"{m}_ser": r[m][1] for m in e3.METHODS})
            rows.append(rec)
            print(f"speed={sp} rep={rep} " + " ".join(
                f"{m}={rec[m + '_ser']:.4f}" for m in SHOW), flush=True)
    with open(os.path.join(DATA, "e3_speed_seeds.csv"), "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def fit(y, X):
    """Least-squares coefficients and their standard errors."""
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    cov = e @ e / (len(y) - X.shape[1]) * np.linalg.inv(X.T @ X)
    return b, np.sqrt(np.diag(cov))


def summarize():
    with open(os.path.join(DATA, "e3_speed_seeds.csv")) as f:
        rows = [{k: float(v) for k, v in r.items()}
                for r in csv.DictReader(f)]
    with open(os.path.join(DATA, "e3_speed.csv")) as f:
        stored = {float(r["speed"]): r for r in csv.DictReader(f)}
    for r in rows:
        r["gap"] = r["DR-tracked_ser"] - r["DR-genie_ser"]
    lines = ["speed  pair_affinity(se)  " + "  ".join(
        f"{m}_ser(se)" for m in SHOW) + "  gap(paired se)  tracked/genie"
        "  max|mean-stored|"]
    for sp in SPEEDS:
        sub = [r for r in rows if r["speed"] == sp]

        def ms(key):
            v = np.array([r[key] for r in sub])
            return v.mean(), v.std(ddof=1) / math.sqrt(len(v))
        dev = max(abs(ms(f"{m}_ser")[0] - float(stored[sp][f"{m}_ser"]))
                  for m in e3.METHODS)
        assert dev < 1e-12, f"speed {sp}: means differ from e3_speed.csv by {dev}"
        pa = ms("pair_affinity")
        lines.append(f"{sp:5.1f}  {pa[0]:.4f}({pa[1]:.4f})  " + "  ".join(
            "{:.4f}({:.4f})".format(*ms(f"{m}_ser")) for m in SHOW)
            + "  {:.4f}({:.4f})".format(*ms("gap"))
            + f"  {ms('DR-tracked_ser')[0] / ms('DR-genie_ser')[0]:.3f}"
            + f"  {dev:.1e}")
    # pooled over the 40 trajectories: SER against visited affinity and
    # log2(speed), so that the speed effect is read at a fixed affinity
    aff = np.array([r["pair_affinity"] for r in rows])
    X = np.c_[np.ones(len(rows)), aff,
              np.log2([r["speed"] for r in rows])]
    lines.append("pooled over 40 trajectories:")
    for key in ["DR-genie_ser", "DR-tracked_ser", "DR-static_ser",
                "LMMSE_ser", "gap"]:
        y = np.array([r[key] for r in rows])
        b, se = fit(y, X)
        lines.append(f"  {key:15s} corr(affinity)={np.corrcoef(aff, y)[0, 1]:+.3f}"
                     f"  fit: affinity {b[1]:+.4f}({se[1]:.4f})"
                     f"  log2 speed {b[2]:+.4f}({se[2]:.4f})")
    with open(os.path.join(DATA, "e3_speed_stats.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    if len(sys.argv) < 2 or sys.argv[1] != "summary":
        simulate()
    summarize()
    print("E3 speed statistics done.")


if __name__ == "__main__":
    main()
