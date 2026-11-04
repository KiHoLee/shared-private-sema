"""E19b: the pilot-free designs of E19 with a causal estimate of the time average.

Purpose (Reviewer 1.2, 5.1, 5.9 and the AE on the mobility contribution).
In E19 the pilot-free designs DR-static and LMMSE-static build their masks
and their receiver from a_bar, the per-user mean of a_u(t) over the whole
300-slot evaluated trace, future slots included. That value is non-causal
and Algorithm 1 cannot obtain it. This script replaces a_bar by estimates a
deployment can form and applies each to both fixed designs (masks and
receiver), on the same front-end draws as the E19 reference rows:

  (a) warm-up   affinity pilots in every one of the first W slots
                (W in {10, 20}, n_w in {16, 64} pilot embeddings per user),
                the rank-one pilot fit of the tracker (pilot_affinity_obs),
                running mean of the observations, frozen after slot W.
                During the warm-up the running mean so far is used. The
                pilot resource W U n_w is charged against the T L_d data
                embeddings of the trace, the convention of E19.
  (b) offline   calibration energies of line 3 of Algorithm 1,
                a_hat^2 = mean ||V_c^T x_u||^2, from N_c clean tuples taken
                at random slots of 20 independent trajectories of the same
                mobility model (never the evaluated one), N_c in {20, 200},
                one common value pooled over users or one value per user.
                "cal200-common-mean" averages a_u instead of a_u^2, so it
                targets the same quantity as a_bar.
                The scene traversal is a deterministic profile, so its
                independent trajectories are the same profile and the
                offline estimate differs from a_bar only by slot sampling
                and by the rms/mean difference.
  (c) biased    a_bar + delta, delta in {-0.2, -0.1, +0.1, +0.2}, clipped
                to [0.02, 0.95], the clip range of the tracker.

It also runs a second tracker, DR-tracked-cal, whose initial value is the
offline common estimate (cal200-common) instead of mean(a_bar), fed with the
same pilot observations as DR-tracked.

Setting: E19 at n_p = 64 pilots per user every K = 5 slots (16% over all
users, the main runs), rho = 12 dB, U = 4, d = 64, d_c = 16, L_d = 320,
T = 300, scene traversal (seeds 3 to 7) and random waypoint at
0.5, 1, 2, 4, 8 m/slot (seeds 1000 + 100 * speed index + rep, rep 0 to 7).
The evaluated random stream is that of exp19_static_masks.py, so the five
reference methods reproduce the n_p = 64 rows of data/e19_static_masks.csv
exactly (asserted). Warm-up pilots and calibration use separate generators
seeded with SeedSequence([19211 or 19212, kind, speed index, rep]).

Outputs: data/e19b_causal_average.csv (per trace and method, SER and
correct embeddings per slot net of pilots, mean and standard error over the
runs or trajectories), data/e19b_perseed.csv (every run and trajectory, all
methods, for figure panels with error bars), data/e19b_notes.txt.
Run with the argument "summary" to rebuild the CSV and notes from
data/e19b_perseed.csv without simulating.
"""
import csv
import math
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp3_mobility as e3
import exp19_static_masks as e19
from semantic_mac import (affinity_matrix, demux_dr, demux_lmmse,
                          sample_latents_isotropic, AffinityTracker,
                          pilot_affinity_obs, estimate_a_from_clean,
                          metrics, H_MIN, D_REAL)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

U, D, DC, BATCH, RHO, K, LAM, T = (e3.U, e3.D, e3.DC, e3.BATCH, e3.RHO,
                                   e3.K_PILOT, e3.LAM, e3.T)
N_P = 64                         # tracker allocation of the main runs (16%)
SPEEDS = e19.SPEEDS
REPS_TS, REPS_SP = e19.REPS_TS, e19.REPS_SP
Vc = e19.Vc
A_LO, A_HI = 0.02, 0.95          # clip range of the tracker
WARM = [(10, 16), (10, 64), (20, 16), (20, 64)]   # (W slots, n_w per user)
CAL = [(20, "common", "rms"), (20, "peruser", "rms"),
       (200, "common", "rms"), (200, "peruser", "rms"),
       (200, "common", "mean")]
N_CAL_TRAJ = 20
DELTAS = [-0.2, -0.1, 0.1, 0.2]
REF = ["DR-genie", "DR-tracked", "DR-static-rx", "DR-static",
       "LMMSE-static"]


def warm_name(W, n_w):
    return f"warm{W}-np{n_w}"


def cal_name(n_c, mode, stat):
    return f"cal{n_c}-{mode}" + ("-mean" if stat == "mean" else "")


def bias_name(dl):
    return f"bias{dl:+.1f}"


VARIANTS = ([warm_name(*w) for w in WARM] + [cal_name(*c) for c in CAL]
            + [bias_name(dl) for dl in DELTAS])
METHODS = (REF + ["DR-tracked-cal"]
           + [f"{r}[{v}]" for v in VARIANTS for r in ("DR", "LMMSE")])


def pilot_fraction(method):
    """Pilot resource over the data resource of the trace (E19 convention)."""
    if method.startswith("DR-tracked"):
        return e19.overhead(N_P)
    for W, n_w in WARM:
        if f"[{warm_name(W, n_w)}]" in method:
            return W * U * n_w / (T * BATCH)
    return 0.0


def calibration(kind, si, rep):
    """Offline estimates from independent trajectories of the same model."""
    rng = np.random.default_rng([19212, kind, si, rep])
    if kind == 0:
        trajs = [e3.drive_through_profile(T)] * N_CAL_TRAJ
    else:
        trajs = [e3.mobility_trajectories(U, T, SPEEDS[si], rng, **e3.MOB)
                 for _ in range(N_CAL_TRAJ)]
    out = {}
    for n_c, mode, stat in CAL:
        ai = rng.integers(0, N_CAL_TRAJ, size=(n_c, U))
        ti = rng.integers(0, T, size=(n_c, U))
        a_s = np.array([[trajs[ai[n, u]][ti[n, u], u] for u in range(U)]
                        for n in range(n_c)])                    # (n_c, U)
        if stat == "rms":
            # clean calibration tuples, Algorithm 1 line 3
            x = np.stack([sample_latents_isotropic(1, U, D, DC, a_s[n], rng)[0]
                          for n in range(n_c)])                  # (n_c, U, D)
            est = estimate_a_from_clean(x, Vc)
            if mode == "common":
                est = np.full(U, math.sqrt(float(np.mean(est ** 2))))
        else:
            est = a_s.mean(0) if mode == "peruser" else \
                np.full(U, float(a_s.mean()))
        out[cal_name(n_c, mode, stat)] = np.clip(est, A_LO, A_HI)
    return out


def run(job):
    kind, si, rep = job                      # kind 0 traversal, 1 waypoint
    if kind == 0:
        a_t = e3.drive_through_profile(T)
        rng = np.random.default_rng(3 + rep)
    else:
        rng = np.random.default_rng(1000 + 100 * si + rep)
        a_t = e3.mobility_trajectories(U, T, SPEEDS[si], rng, **e3.MOB)
    a_bar = a_t.mean(0)
    B_bar = affinity_matrix(a_bar)
    tr = AffinityTracker(U, lam=LAM, a_init=float(a_bar.mean()))
    cal = calibration(kind, si, rep)
    tr_c = AffinityTracker(U, lam=LAM,
                           a_init=float(cal["cal200-common"].mean()))
    rng_w = np.random.default_rng([19211, kind, si, rep])
    wobs = {n_w: [] for n_w in sorted({n for _, n in WARM})}
    fixed = dict(cal)
    for dl in DELTAS:
        fixed[bias_name(dl)] = np.clip(a_bar + dl, A_LO, A_HI)
    s = {k: [] for k in METHODS}
    for t in range(T):
        a = a_t[t]
        B = affinity_matrix(a)
        z = sample_latents_isotropic(BATCH, U, D, DC, a, rng)
        hc = (rng.standard_normal((BATCH, U)) +
              1j * rng.standard_normal((BATCH, U))) / math.sqrt(2)
        h = np.clip(np.abs(hc), H_MIN, None)
        xi = rng.standard_normal((BATCH, U, D)) * math.sqrt(1.0 / RHO)
        if t % K == 0:
            zp = sample_latents_isotropic(N_P, U, D, DC, a, rng)
            hp = np.clip(np.abs((rng.standard_normal(U) +
                                 1j * rng.standard_normal(U)) / np.sqrt(2)),
                         0.2, None)
            obs = pilot_affinity_obs(zp, hp, RHO, rng)
            tr.update(obs)
            tr_c.update(obs)
        # warm-up pilots on their own generator (the stream above is E19's)
        if t < max(W for W, _ in WARM):
            for n_w in wobs:
                zw = sample_latents_isotropic(n_w, U, D, DC, a, rng_w)
                hw = np.clip(np.abs((rng_w.standard_normal(U) +
                                     1j * rng_w.standard_normal(U))
                                    / np.sqrt(2)), 0.2, None)
                wobs[n_w].append(pilot_affinity_obs(zw, hw, RHO, rng_w))
        est = dict(fixed)
        for W, n_w in WARM:
            est[warm_name(W, n_w)] = np.clip(
                np.mean(wobs[n_w][:min(t + 1, W)], axis=0), A_LO, A_HI)
        a_hat = np.clip(tr.a, A_LO, A_HI)
        a_hc = np.clip(tr_c.a, A_LO, A_HI)
        Bh, Bhc = affinity_matrix(a_hat), affinity_matrix(a_hc)
        t_true = e19.front(z, B, h, xi)
        t_hat = e19.front(z, Bh, h, xi)
        t_bar = e19.front(z, B_bar, h, xi)
        out = {
            "DR-genie": demux_dr(t_true, B, h, RHO, a, Vc),
            "DR-tracked": demux_dr(t_hat, Bh, h, RHO, a_hat, Vc),
            "DR-static-rx": demux_dr(t_true, B_bar, h, RHO, a_bar, Vc),
            "DR-static": demux_dr(t_bar, B_bar, h, RHO, a_bar, Vc),
            "LMMSE-static": demux_lmmse(t_bar, B_bar, h, RHO)[0],
            "DR-tracked-cal": demux_dr(e19.front(z, Bhc, h, xi), Bhc, h,
                                       RHO, a_hc, Vc),
        }
        for v in VARIANTS:
            Be = affinity_matrix(est[v])
            te = e19.front(z, Be, h, xi)
            out[f"DR[{v}]"] = demux_dr(te, Be, h, RHO, est[v], Vc)
            out[f"LMMSE[{v}]"] = demux_lmmse(te, Be, h, RHO)[0]
        for k in METHODS:
            s[k].append(metrics(out[k], z)[2])
    iu = np.triu_indices(U, k=1)
    rec = dict(trace="traversal" if kind == 0 else f"speed_{SPEEDS[si]}",
               rep=rep, seed=(3 + rep) if kind == 0 else 1000 + 100 * si + rep,
               mean_a=float(a_t.mean()),
               pair_affinity=float(np.mean([np.outer(x, x)[iu].mean()
                                            for x in a_t])))
    rec.update({f"abar{u}": float(a_bar[u]) for u in range(U)})
    for v in VARIANTS:
        if not v.startswith("warm"):
            rec.update({f"{v}_a{u}": float(est[v][u]) for u in range(U)})
    for W, n_w in WARM:
        rec.update({f"{warm_name(W, n_w)}_a{u}": float(
            np.clip(np.mean(wobs[n_w][:W], axis=0), A_LO, A_HI)[u])
            for u in range(U)})
    for k in METHODS:
        ser = float(np.mean(s[k]))
        rec[f"{k}_ser"] = ser
        rec[f"{k}_good"] = (1 - pilot_fraction(k)) * (1 - ser)
    print(f"{rec['trace']} rep={rep} DR-static={rec['DR-static_ser']:.4f} "
          f"DR[cal200-common]={rec['DR[cal200-common]_ser']:.4f}", flush=True)
    return rec


def mse(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))


def time_scale():
    """Slot to physical time, from the definitions of semantic_mac.py."""
    lines = ["Physical time scale (from L_d, d, K, lambda of the code)"]
    lines.append(f"  a slot carries L_d = {BATCH} embeddings per user; each "
                 f"embedding occupies d real channel uses, all U = {U} users "
                 f"superposed on the same uses")
    for dd, tag in ((D, "synthetic d=64"), (D_REAL, "BERT d=768")):
        lines.append(f"  {tag}: {BATCH * dd} real channel uses per slot "
                     f"({BATCH * dd // 2} complex); a pilot slot adds "
                     f"U n_p d = {U * N_P * dd} real uses on orthogonal "
                     f"resources")
    lines.append(f"  tracker time constant K/lambda = {K / LAM:.1f} slots; "
                 f"pilot period K = {K} slots; trace T = {T} slots")
    lines.append("  embedding rate r per user -> slot = L_d / r seconds; "
                 "speed in m/s = (m/slot) * r / L_d")
    for r in (10, 30, 640):
        sl = BATCH / r
        sp = ", ".join(f"{v:g} m/slot = {v / sl:.3g} m/s" for v in SPEEDS)
        lines.append(f"  r = {r:4d}/s: slot = {sl:.4g} s, T = {T * sl:.4g} s, "
                     f"K/lambda = {K / LAM * sl:.4g} s, real symbol rate "
                     f"{r * D:.4g}/s (d=64) or {r * D_REAL:.4g}/s (d=768); "
                     f"{sp}")
    for rs in (1e6, 1e7):
        for dd in (D, D_REAL):
            r = rs / dd
            sl = BATCH / r
            lines.append(f"  symbol rate {rs:.0e} real/s, d = {dd}: r = {r:.4g} "
                         f"embeddings/s per user, slot = {sl * 1e3:.4g} ms, "
                         f"0.5 to 8 m/slot = {0.5 / sl:.4g} to {8 / sl:.4g} m/s")
    lines.append("  conversely, a speed v in m/s maps to v L_d / r m/slot; "
                 "1.5 m/s (pedestrian) is 0.75 m/slot at r = 640/s and "
                 "16 m/slot at r = 30/s")
    return lines


def summarize(rows):
    with open(os.path.join(DATA, "e19_static_masks.csv")) as f:
        ref = {r["trace"]: r for r in csv.DictReader(f)
               if int(r["n_p_per_user"]) == N_P}
    traces = ["traversal"] + [f"speed_{sp}" for sp in SPEEDS]
    notes, out = [], []
    worst = 0.0
    for tr in traces:
        g = [r for r in rows if r["trace"] == tr]
        for k in REF:
            for q in ("ser", "good"):
                dv = abs(np.mean([r[f"{k}_{q}"] for r in g])
                         - float(ref[tr][f"{k}_{q}"]))
                worst = max(worst, dv)
    assert worst < 1e-5, f"reference rows differ from E19 by {worst}"
    notes.append(f"Reproduction check: the five E19 methods at n_p={N_P} "
                 f"match data/e19_static_masks.csv on all 6 traces, "
                 f"max |diff| = {worst:.1e} (CSV rounded to 5 decimals).")
    notes.append("Paired comparisons use the same fading, noise and "
                 "content draws for every method of a run.")
    notes.append("")
    for tr in traces:
        g = [r for r in rows if r["trace"] == tr]
        n = len(g)
        notes.append(f"== {tr} (n={n}, mean a={np.mean([r['mean_a'] for r in g]):.3f}, "
                     f"a_bar per user averaged over runs "
                     + " ".join(f"{np.mean([r[f'abar{u}'] for r in g]):.3f}"
                                for u in range(U)) + ")")
        for k in METHODS:
            sm, ss = mse([r[f"{k}_ser"] for r in g])
            gm, gs = mse([r[f"{k}_good"] for r in g])
            pf = pilot_fraction(k)
            out.append(dict(trace=tr, method=k, n=n, pilot_fraction=pf,
                            ser_mean=sm, ser_se=ss, good_mean=gm, good_se=gs))
        d_l = {}
        for v in ["oracle"] + VARIANTS:
            kd = "DR-static" if v == "oracle" else f"DR[{v}]"
            kl = "LMMSE-static" if v == "oracle" else f"LMMSE[{v}]"
            gd = [r[f"{kd}_good"] for r in g]
            gl = [r[f"{kl}_good"] for r in g]
            dm, ds = mse(np.array(gd) - np.array(gl))
            sm, ss = mse([r[f"{kd}_ser"] for r in g])
            lm, ls = mse([r[f"{kl}_ser"] for r in g])
            ratio = np.mean(gd) / np.mean(gl)
            vs_or = mse(np.array(gd) - np.array([r["DR-static_good"]
                                                 for r in g]))
            d_l[v] = ratio
            out.append(dict(trace=tr, method=f"gain[{v}]", n=n,
                            pilot_fraction=pilot_fraction(f"[{v}]"),
                            ser_mean=sm - lm, ser_se=float("nan"),
                            good_mean=ratio, good_se=ds / np.mean(gl)))
            est = ("" if v == "oracle" or v.startswith("warm") else
                   " a=" + " ".join(f"{np.mean([r[f'{v}_a{u}'] for r in g]):.3f}"
                                    for u in range(U)))
            if v.startswith("warm"):
                est = " a(frozen)=" + " ".join(
                    f"{np.mean([r[f'{v}_a{u}'] for r in g]):.3f}"
                    for u in range(U))
            notes.append(
                f"  {v:19s} pilots={pilot_fraction(f'[{v}]'):6.2%} "
                f"SER DR={sm:.4f}+-{ss:.4f} LMMSE={lm:.4f}+-{ls:.4f} | "
                f"correct/slot DR={np.mean(gd):.4f} LMMSE={np.mean(gl):.4f} "
                f"DR/LMMSE={ratio:.3f} (diff {dm:.4f}+-{ds:.4f}) "
                f"DR-oracleDR={vs_or[0]:+.4f}+-{vs_or[1]:.4f}{est}")
        for k in ("DR-genie", "DR-tracked", "DR-tracked-cal", "DR-static-rx"):
            sm, ss = mse([r[f"{k}_ser"] for r in g])
            gm, gs = mse([r[f"{k}_good"] for r in g])
            notes.append(f"  {k:19s} pilots={pilot_fraction(k):6.2%} "
                         f"SER={sm:.4f}+-{ss:.4f} correct/slot={gm:.4f}+-{gs:.4f}")
        gap = mse([r["DR-tracked_ser"] - r["DR-genie_ser"] for r in g])
        gap_s = mse([r["DR-static_ser"] - r["DR-genie_ser"] for r in g])
        gap_c = mse([r["DR[cal200-common]_ser"] - r["DR-genie_ser"]
                     for r in g])
        net = mse([r["DR-tracked_good"] - r["DR[cal200-common]_good"]
                   for r in g])
        notes.append(f"  paired SER gap to DR-genie: tracked {gap[0]:.4f}+-"
                     f"{gap[1]:.4f}, DR-static(oracle) {gap_s[0]:.4f}+-"
                     f"{gap_s[1]:.4f}, DR[cal200-common] {gap_c[0]:.4f}+-"
                     f"{gap_c[1]:.4f}; correct/slot tracked minus "
                     f"DR[cal200-common] {net[0]:+.4f}+-{net[1]:.4f}")
        notes.append("")
    key = ["Key lines (DR/LMMSE = ratio of mean correct embeddings per slot, "
           "both fixed designs using the same estimate)"]
    groups = {"warm-up": [v for v in VARIANTS if v.startswith("warm")],
              "offline calibration": [v for v in VARIANTS if v.startswith("cal")],
              "biased": [v for v in VARIANTS if v.startswith("bias")],
              "oracle": ["oracle"]}
    for name, trs in (("traversal", ["traversal"]),
                      ("random waypoint", traces[1:])):
        for gname, vs in groups.items():
            rr = [r["good_mean"] for r in out if r["trace"] in trs
                  and r["method"] in [f"gain[{v}]" for v in vs]]
            key.append(f"  {name:15s} {gname:19s} DR/LMMSE from "
                       f"{min(rr):.3f} to {max(rr):.3f}")
        d = [np.mean([r["DR-tracked_good"] - r["DR[cal200-common]_good"]
                      for r in rows if r["trace"] == t]) for t in trs]
        key.append(f"  {name:15s} DR-tracked (16% pilots) minus "
                   f"DR[cal200-common] correct/slot from {min(d):+.4f} "
                   f"to {max(d):+.4f}")
        d = [np.mean([abs(r["DR-tracked_good"] - r["DR-tracked-cal_good"])
                      for r in rows if r["trace"] == t]) for t in trs]
        key.append(f"  {name:15s} |DR-tracked minus DR-tracked-cal| "
                   f"correct/slot at most {max(d):.4f}")
    for t in traces:
        g = {r["method"]: r["good_mean"] for r in out if r["trace"] == t}
        cand = ["oracle"] + VARIANTS
        bd = max(cand, key=lambda v: g["DR-static" if v == "oracle"
                                       else f"DR[{v}]"])
        bl = max(cand, key=lambda v: g["LMMSE-static" if v == "oracle"
                                       else f"LMMSE[{v}]"])
        vd = g["DR-static" if bd == "oracle" else f"DR[{bd}]"]
        vl = g["LMMSE-static" if bl == "oracle" else f"LMMSE[{bl}]"]
        key.append(f"  {t:15s} best tested constant per receiver: DR {bd} "
                   f"{vd:.4f}, LMMSE {bl} {vl:.4f}, ratio {vd / vl:.3f}")
    notes = notes[:3] + key + [""] + notes[3:]
    notes += time_scale()
    with open(os.path.join(DATA, "e19b_causal_average.csv"), "w") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        for r in out:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    with open(os.path.join(DATA, "e19b_notes.txt"), "w") as f:
        f.write("E19b: causal estimates of the time average "
                "(code/exp19b_causal_average.py)\n"
                "gain[v] rows of the CSV: good_mean = DR/LMMSE ratio of mean "
                "correct embeddings per slot, good_se = SE of the paired "
                "difference over the LMMSE mean, ser_mean = DR minus LMMSE "
                "SER.\n\n")
        f.write("\n".join(notes) + "\n")
    print("\n".join(notes))


def main():
    path = os.path.join(DATA, "e19b_perseed.csv")
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        with open(path) as f:
            rows = [{k: (v if k == "trace" else float(v)) for k, v in r.items()}
                    for r in csv.DictReader(f)]
        summarize(rows)
        return
    jobs = [(0, 0, r) for r in range(REPS_TS)]
    jobs += [(1, si, r) for si in range(len(SPEEDS)) for r in range(REPS_SP)]
    workers = int(os.environ.get("E19B_WORKERS", "8"))
    with Pool(workers) as p:
        rows = p.map(run, jobs)
    with open(path, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    summarize(rows)
    print("E19b done.")


if __name__ == "__main__":
    main()
