"""Mask-realization check: the paper's experiments generate the
matched-filter outputs from the EXPECTED cross-Gram model
(semantic_mac.matched_filter). This script draws actual Haar-mixture
masks, passes the physical superposition r = sum_v h_v M_v x_v + n
through the realized masks, applies the same receivers, and compares
the SER against the expectation-model front end on identical latents,
gains, and noise seeds.

The realized per-entry Gram deviation is O(1/sqrt(d)); this check
quantifies its end-to-end effect at d=64 (theory/mobility setting) and
d=768 (real-embedding setting).

Outputs: data/e_mask_check.csv

With the argument "sweep" the script instead sweeps the SNR. Above the SNR
at which the noise energy d sigma^2 falls below the residual energy
U (1 - beta^2) of the realized cross-Grams, a receiver that keeps the
nominal sigma^2 fails, so the sweep also runs the DR and LMMSE receivers
with sigma^2 + U (1 - beta^2) / d. It also computes the exact posterior MSE
of the linear MMSE receiver that knows every realized mask (h = 1), which
floors at high SNR because the U outputs span at most d dimensions.

Sweep outputs: data/e_mask_sweep.csv, data/e_mask_floor.csv
"""
import math
import os
import sys
import numpy as np

from semantic_mac import (affinity_matrix, matched_filter, demux_sr,
                          demux_sc, demux_lmmse, demux_dr,
                          sample_latents_isotropic, random_orthogonal,
                          metrics)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

U = 4
BETA = 0.5


def haar_masks(B, d, rng):
    """M_u = sum_k A_uk U_k with A = chol(B), U_k independent Haar."""
    A = np.linalg.cholesky(B)
    Us = [random_orthogonal(d, rng) for _ in range(U)]
    return [sum(A[u, k] * Us[k] for k in range(U)) for u in range(U)]


def physical_front_end(z, h, Ms, rho, rng):
    """r = sum_v h_v M_v z_v + n (ambient AWGN), tilde_u = M_u^T r / h_u."""
    batch, Uu, d = z.shape
    sigma = math.sqrt(1.0 / rho)
    r = np.einsum('bv,vde,bve->bd', h,
                  np.stack(Ms), z) + rng.standard_normal((batch, d)) * sigma
    tilde = np.stack([(r @ Ms[u]) / h[:, u][:, None] for u in range(U)],
                     axis=1)
    return tilde


def run(d, d_c, batch, n_mask_draws, snr_db):
    RHO = 10 ** (snr_db / 10)
    a = math.sqrt(BETA) * np.ones(U)
    B = affinity_matrix(a)
    Vc = np.eye(d)[:, :d_c]
    diffs = {m: [] for m in ("SR", "SC", "LMMSE", "DR")}
    sers_r = {m: [] for m in diffs}
    sers_e = {m: [] for m in diffs}
    for t in range(n_mask_draws):
        rng = np.random.default_rng(7700 + 10 * t)
        z = sample_latents_isotropic(batch, U, d, d_c, a, rng)
        hc = (rng.standard_normal((batch, U)) +
              1j * rng.standard_normal((batch, U))) / math.sqrt(2)
        h = np.clip(np.abs(hc), 0.2, None)
        Ms = haar_masks(B, d, rng)
        gram_dev = max(np.abs(Ms[u].T @ Ms[v] - B[u, v] * np.eye(d)).max()
                       for u in range(U) for v in range(U))
        tilde_r = physical_front_end(z, h, Ms, RHO, rng)
        # expectation-model front end on the SAME latents and gains
        rng_e = np.random.default_rng(8800 + 10 * t)
        tilde_e = np.einsum('uv,bvd,bv,bu->bud', B, z, h, 1.0 / h)
        xi = rng_e.standard_normal((batch, U, d)) * math.sqrt(1.0 / RHO)
        A = np.linalg.cholesky(B)
        for b in range(batch):
            tilde_e[b] += (A @ xi[b]) / h[b][:, None]
        for name, tl in (("realized", tilde_r), ("expected", tilde_e)):
            out = {}
            out["SR"] = demux_sr(tl, B, h)
            out["SC"] = demux_sc(tl, h)
            out["LMMSE"], _ = demux_lmmse(tl, B, h, RHO)
            out["DR"] = demux_dr(tl, B, h, RHO, a, Vc)
            for m in diffs:
                _, _, s = metrics(out[m], z)
                (sers_r if name == "realized" else sers_e)[m].append(s)
        for m in diffs:
            diffs[m].append(sers_r[m][-1] - sers_e[m][-1])
        print(f"d={d} draw {t+1}/{n_mask_draws} gram_dev={gram_dev:.3f} "
              + " ".join(f"{m}:d={diffs[m][-1]:+.4f}" for m in diffs))
    return {m: (float(np.mean(sers_r[m])), float(np.mean(sers_e[m])),
                float(np.mean(diffs[m])), float(np.std(diffs[m])))
            for m in diffs}


def main():
    rows = []
    for d, d_c, batch, draws, snr_db in ((64, 16, 2000, 12, 10),
                                         (768, 128, 400, 6, 20)):
        res = run(d, d_c, batch, draws, snr_db)
        for m, (sr, se, md, sd) in res.items():
            rows.append((d, snr_db, m, sr, se, md, sd))
            print(f"d={d} snr={snr_db} {m}: realized={sr:.4f} "
                  f"expected={se:.4f} mean_diff={md:+.5f} std={sd:.5f}")
    with open(os.path.join(DATA, "e_mask_check.csv"), "w") as f:
        f.write("d,snr,method,ser_realized,ser_expected,mean_diff,"
                "std_diff\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print("mask check done.")


def sweep_cell(job):
    d, d_c, beta, snr_db, draw, batch = job
    rho = 10 ** (snr_db / 10)
    rho_i = 1.0 / (1.0 / rho + U * (1 - beta ** 2) / d)
    rng = np.random.default_rng([9100, d, int(beta * 1000), snr_db, draw])
    a = math.sqrt(beta) * np.ones(U)
    B = affinity_matrix(a)
    Vc = np.eye(d)[:, :d_c]
    z = sample_latents_isotropic(batch, U, d, d_c, a, rng)
    hc = (rng.standard_normal((batch, U)) +
          1j * rng.standard_normal((batch, U))) / math.sqrt(2)
    h = np.clip(np.abs(hc), 0.2, None)
    Ms = haar_masks(B, d, rng)
    t_r = physical_front_end(z, h, Ms, rho, rng)
    t_e = np.einsum('uv,bvd,bv,bu->bud', B, z, h, 1.0 / h)
    xi = rng.standard_normal((batch, U, d)) * math.sqrt(1.0 / rho)
    t_e = t_e + np.einsum('uv,bvd->bud', np.linalg.cholesky(B), xi) / h[:, :, None]
    est = {
        "E_SC": demux_sc(t_e, h), "E_LMMSE": demux_lmmse(t_e, B, h, rho)[0],
        "E_DR": demux_dr(t_e, B, h, rho, a, Vc),
        "R_SC": demux_sc(t_r, h), "R_LMMSE": demux_lmmse(t_r, B, h, rho)[0],
        "R_DR": demux_dr(t_r, B, h, rho, a, Vc),
        "I_LMMSE": demux_lmmse(t_r, B, h, rho_i)[0],
        "I_DR": demux_dr(t_r, B, h, rho_i, a, Vc),
    }
    return job, {k: metrics(v, z) for k, v in est.items()}


def floor_cell(job):
    """Exact posterior MSE of the LMMSE receiver from r with all realized
    masks known, h = 1, against Proposition 2."""
    d, d_c, beta, snr_db, draw = job
    rng = np.random.default_rng([9200, d, int(beta * 1000), snr_db, draw])
    a = math.sqrt(beta) * np.ones(U)
    B = affinity_matrix(a)
    Ms = np.stack(haar_masks(B, d, rng))
    s2 = 10 ** (-snr_db / 10)
    Mc, Mp = Ms[:, :, :d_c], Ms[:, :, d_c:]
    G = (a[:, None, None] * Mc).sum(0)
    C = (G @ G.T / d_c + sum((1 - beta) / (d - d_c) * Mp[u] @ Mp[u].T
                             for u in range(U)) + s2 * np.eye(d))
    Ci = np.linalg.inv(C)
    gain_s = beta / d_c ** 2 * np.trace(G.T @ Ci @ G)
    mse = [1 - gain_s - ((1 - beta) / (d - d_c)) ** 2
           * np.trace(Mp[u].T @ Ci @ Mp[u]) for u in range(U)]
    return job, float(np.mean(mse))


def prop2_mse(d, d_c, beta, snr_db):
    """Per-user DR error of Proposition 2 at h = 1."""
    s2 = 10 ** (-snr_db / 10)
    a = math.sqrt(beta) * np.ones(U)
    B = affinity_matrix(a)
    g = a * (1 + (U - 1) * beta)
    v_s = 1.0 / (g @ np.linalg.solve(s2 * B, g))
    V_p = (d - d_c) * s2 * np.linalg.inv(B)[0, 0]
    return (beta * d_c * v_s / (1 + d_c * v_s)
            + (1 - beta) * V_p / ((1 - beta) + V_p))


def sweep():
    from multiprocessing import Pool
    workers = int(os.environ.get("MASK_WORKERS", "8"))
    jobs = [(64, 16, b, s, k, 1000) for b in (0.4, 0.5, 0.7)
            for s in (10, 12, 15, 20, 25, 30) for k in range(20)]
    jobs += [(768, 128, 0.5, s, k, 200) for s in (20, 25, 30)
             for k in range(6)]
    with Pool(workers) as p:
        res = p.map(sweep_cell, jobs)
    cells = {}
    for job, r in res:
        cells.setdefault(job[:4], []).append(r)
    with open(os.path.join(DATA, "e_mask_sweep.csv"), "w") as f:
        f.write("d,d_c,beta,snr,method,cos,nmse,ser,draws\n")
        for key in sorted(cells):
            for m in cells[key][0]:
                v = np.array([r[m] for r in cells[key]])
                f.write(",".join(str(x) for x in key) + f",{m},"
                        + ",".join(f"{x:.5f}" for x in v.mean(0))
                        + f",{len(v)}\n")
                print(key, m, " ".join(f"{x:.4f}" for x in v.mean(0)),
                      flush=True)
    fjobs = [(64, 16, 0.5, s, k) for s in (10, 20, 30, 40, 60)
             for k in range(16)]
    fjobs += [(768, 128, 0.5, s, k) for s in (10, 20, 30, 40, 60)
              for k in range(4)]
    with Pool(workers) as p:
        fres = p.map(floor_cell, fjobs)
    fc = {}
    for job, v in fres:
        fc.setdefault(job[:4], []).append(v)
    with open(os.path.join(DATA, "e_mask_floor.csv"), "w") as f:
        f.write("d,d_c,beta,snr,mse_physical_lmmse,se,mse_prop2,draws\n")
        for key in sorted(fc):
            v = np.array(fc[key])
            f.write(",".join(str(x) for x in key)
                    + f",{v.mean():.5f},{v.std(ddof=1)/math.sqrt(len(v)):.5f},"
                    f"{prop2_mse(*key):.5f},{len(v)}\n")
            print("floor", key, f"{v.mean():.4f}", f"{prop2_mse(*key):.4f}",
                  flush=True)
    print("mask sweep done.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        sweep()
    else:
        main()
