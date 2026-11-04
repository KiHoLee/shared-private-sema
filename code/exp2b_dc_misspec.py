"""E2b: receiver shared dimension d_c misspecified (Reviewer 2.2).

Reviewer 2 asked what d_c means in practice. The response states that d_c
is a design parameter whenever the calibration set is too small to show the
knee of the spectrum, as at the N = 200 of E2. E2 itself gives the
receiver the true d_c = 128, so it does not test a wrong choice. This
script does.

Setting. The E2 configuration of exp2_structure.py (real BERT AG-News
contents, U = 4, d = 768, true d_c = 128, beta = 0.5, Rayleigh fading
clipped at h_min = 0.2, 4,000 held-out tuples per SNR). The E2 random chain
is replayed exactly (seed 7: mixing R, N = 200 calibration pairs, the
adapter training draws, then the evaluation draws of the ladder), so the
evaluation tuples and channels are those of data/e2_ladder.csv. From the
same N = 200 calibration pairs the spectral basis keeps the leading
d_c' eigenvectors, d_c' in {64, 96, 128, 160, 192}, and the DR is run with
that basis and that d_c' in its shrinkage. The oracle variant uses the
first d_c' columns of the true mixing R (for d_c' > 128 it includes
d_c' - 128 private directions). The LMMSE receiver needs no d_c.
Evaluated at 12, 16 and 20 dB (20 dB is the operating point asked for).

Check. The subspace error of the d_c' = 128 basis must reproduce
data/e2_caliberr.txt to the printed digits. The d_c' = 128 spectral row
and the LMMSE row are compared with the DR-spec and LMMSE columns of
data/e2_ladder.csv. That file was produced on the Mac, and on the u24
server the BERT-pool evaluation draws differ at the Monte Carlo level
(see exp9b_threshold_by_scheme.py), so this comparison is within Monte
Carlo error, not to the digit.

The function replay_e2 is also imported by exp2c_adapter_snr.py and
exp10b_clip_reference.py.

Outputs: data/e2b_dc_misspec.csv, data/e2b_notes.txt
"""
import copy
import math
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp2_structure as e2
from semantic_mac import (EmbeddingPool, affinity_matrix, matched_filter,
                          demux_lmmse, demux_dr, random_orthogonal,
                          learn_structure_spectral, subspace_error, metrics,
                          TAU)

DATA = e2.DATA
SNRS_EVAL = [12, 16, 20]
DC_RX = [64, 96, 128, 160, 192]


def load_pool():
    obj = torch.load(e2.POOL_PT, map_location="cpu")
    X = obj.numpy() if torch.is_tensor(obj) else np.asarray(obj)
    return EmbeddingPool(X)


def replay_e2(pool, keep_snrs, adapter_fn=None):
    """Replay the random chain of exp2_structure.main() exactly.

    The subspace and spectrum sweeps of E2 use their own generators and
    are skipped. The adapter is trained because its channel draws advance
    the main generator. adapter_fn(x_cal, a, Q, rng) defaults to
    exp2_structure.train_adapter. For every SNR in keep_snrs the returned
    dict holds (z, x, tilde, h, state), where state is the generator state
    just before matched_filter, so that a caller can redraw the same
    complex gains and noise with another clip."""
    adapter_fn = adapter_fn or e2.train_adapter
    rng = np.random.default_rng(7)
    a = math.sqrt(e2.BETA) * np.ones(e2.U)
    B = affinity_matrix(a)
    R = random_orthogonal(e2.D, rng)
    _, x_cal = e2.gen_clean(pool, e2.N_PAIR, a, R, rng, idx_pool=e2.IDX_TRAIN)
    V_spec, _ = learn_structure_spectral(x_cal, e2.DC)
    Q, _ = np.linalg.qr(np.hstack([
        V_spec, rng.standard_normal((e2.D, e2.D - e2.DC))]))
    W_ad = adapter_fn(x_cal, a, Q, rng)
    evals = {}
    for s in range(0, 21, 4):
        rho = 10 ** (s / 10)
        z, x = e2.gen_clean(pool, e2.BATCH_EVAL, a, R, rng,
                            idx_pool=e2.IDX_EVAL)
        state = copy.deepcopy(rng.bit_generator.state)
        tilde, h = matched_filter(x, B, rho, rng)
        if s in keep_snrs:
            evals[s] = (z, x, tilde, h, state)
    return dict(a=a, B=B, R=R, V_true=R[:, :e2.DC], x_cal=x_cal,
                V_spec=V_spec, Q=Q, W_ad=W_ad, evals=evals)


def read_ladder():
    """Published E2 ladder, keyed by SNR, values as floats."""
    import csv
    with open(os.path.join(DATA, "e2_ladder.csv")) as f:
        return {int(r["snr"]): {k: float(v) for k, v in r.items()}
                for r in csv.DictReader(f)}


def inst_err(e_hat, x_true, tau=TAU):
    e_n = e_hat / (np.linalg.norm(e_hat, axis=2, keepdims=True) + 1e-12)
    return ((e_n * x_true).sum(-1) < tau).astype(float).ravel()


def main():
    pool = load_pool()
    ch = replay_e2(pool, SNRS_EVAL)
    a, B, R, V_true = ch["a"], ch["B"], ch["R"], ch["V_true"]
    _, ev = learn_structure_spectral(ch["x_cal"], e2.DC)
    # all eigenvectors of the same calibration cross-covariance
    Vall, _ = learn_structure_spectral(ch["x_cal"], max(DC_RX))
    ladder = read_ladder()
    notes = []
    err128 = subspace_error(Vall[:, :e2.DC], V_true)
    notes.append(f"check: d_c'=128 spectral subspace error {err128:.10f} "
                 f"(e2_caliberr.txt spectral_err=0.5066663187245027)")
    rows = []
    for s in SNRS_EVAL:
        rho = 10 ** (s / 10)
        z, x, tilde, h, _ = ch["evals"][s]
        lm, _ = demux_lmmse(tilde, B, h, rho)
        m_lm = metrics(lm, x)
        rows.append((s, "LMMSE", "none", 0, float("nan"), float("nan"))
                    + m_lm + (float("nan"), float("nan")))
        ref = ladder[s]
        notes.append(f"check {s} dB LMMSE ser {m_lm[2]:.7f} nmse "
                     f"{m_lm[1]:.10f} (e2_ladder {ref['LMMSE_ser']:.7f}, "
                     f"{ref['LMMSE_nmse']:.10f})")
        err_ref = None
        for basis in ("spectral", "oracle"):
            for dc in DC_RX:
                Vc = Vall[:, :dc] if basis == "spectral" else R[:, :dc]
                out = demux_dr(tilde, B, h, rho, a, Vc)
                m = metrics(out, x)
                # captured fraction of the true shared subspace, and the
                # fraction of the receiver subspace that is truly shared
                ov = float(np.linalg.norm(V_true.T @ Vc) ** 2)
                cap, pur = ov / e2.DC, ov / dc
                err_i = inst_err(out, x)
                if basis == "spectral" and dc == e2.DC:
                    err_ref = err_i
                    notes.append(
                        f"check {s} dB DR-spec(d_c'=128) ser {m[2]:.7f} nmse "
                        f"{m[1]:.10f} (e2_ladder {ref['DR-spec_ser']:.7f}, "
                        f"{ref['DR-spec_nmse']:.10f})")
                rows.append((s, "DR", basis, dc, cap, pur) + m
                            + (err_i, None))
        # paired SER difference against the d_c'=128 spectral receiver
        for i, r in enumerate(rows):
            if r[0] == s and r[1] == "DR":
                d_i = r[9] - err_ref
                se = float(d_i.std(ddof=1) / math.sqrt(d_i.size))
                rows[i] = r[:9] + (float(d_i.mean()), se)
    with open(os.path.join(DATA, "e2b_dc_misspec.csv"), "w") as f:
        f.write("snr,receiver,basis,dc_rx,captured_shared_frac,"
                "rx_subspace_shared_frac,cos,nmse,ser,"
                "ser_minus_spec128,se_paired\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")
    for r in rows:
        line = (f"{r[0]:2d} dB {r[1]:5s} {r[2]:8s} d_c'={r[3]:3d} "
                f"cap={r[4]:.3f} pur={r[5]:.3f} cos={r[6]:.4f} "
                f"nmse={r[7]:.4f} ser={r[8]:.4f} "
                f"dSER_vs_spec128={r[9]:+.4f} (se {r[10]:.4f})")
        notes.append(line)
    notes.append("calibration eigenvalues 120-200 (N=200): " + " ".join(
        f"{i+1}:{ev[i]:.2e}" for i in (63, 95, 119, 126, 127, 128, 129,
                                       159, 191, 199, 200)))
    with open(os.path.join(DATA, "e2b_notes.txt"), "w") as f:
        f.write("\n".join(notes) + "\n")
    print("\n".join(notes))
    print("E2b done.")


if __name__ == "__main__":
    main()
