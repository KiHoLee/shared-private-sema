"""Private-block cost of zero forcing, averaged over fading.

check_blmmse_optimality.py compares the DR's zero forcing with scalar
shrinkage against the MMSE filter of Remark 1 at single channel draws. This
script averages the exact private-block mean-square errors of the two
filters over 20,000 clipped Rayleigh draws, at the E6 parameters (U=4,
d=64, d_c=16) and several affinities and SNRs.

Output: data/e6_optimality_fading.txt
"""
import math
import os
import numpy as np
from semantic_mac import affinity_matrix, H_MIN, U_MAIN as U, D_SYN as D, DC_SYN as DC

HERE = os.path.dirname(os.path.abspath(__file__))
d_p = D - DC
rng = np.random.default_rng(2026)
lines = ["beta snr_db  mse_DR  mse_MMSE  excess(%)  (private block, per user, fading average)"]
for rho_db in (10, 20):
    for beta in (0.3, 0.5, 0.7):
        a = math.sqrt(beta) * np.ones(U)
        B = affinity_matrix(a)
        Binv = np.linalg.inv(B)
        s2 = 10 ** (-rho_db / 10)
        Cp = np.diag((1 - a ** 2) / d_p)
        e_dr = e_mm = 0.0
        n = 20000
        for _ in range(n):
            h = np.clip(np.abs((rng.standard_normal(U) + 1j * rng.standard_normal(U))
                               / math.sqrt(2)), H_MIN, None)
            Gam = np.diag(1 / h) @ B @ np.diag(h)
            Cn = s2 * np.diag(1 / h) @ B @ np.diag(1 / h)
            Wm = Cp @ Gam.T @ np.linalg.inv(Gam @ Cp @ Gam.T + Cn)
            err_p = d_p * s2 * np.diag(Binv) / h ** 2
            sig_p = 1 - a ** 2
            Wd = np.diag(sig_p / (sig_p + err_p)) @ np.linalg.inv(Gam)
            for W, acc in ((Wd, "dr"), (Wm, "mm")):
                E = W @ Gam - np.eye(U)
                C = E @ Cp @ E.T + W @ Cn @ W.T
                v = d_p * np.trace(C) / U
                if acc == "dr":
                    e_dr += v
                else:
                    e_mm += v
        e_dr /= n
        e_mm /= n
        lines.append(f"{beta:.1f}  {rho_db:2d}  {e_dr:.4f}  {e_mm:.4f}  {100*(e_dr/e_mm-1):.1f}")
        print(lines[-1], flush=True)
with open(os.path.join(HERE, "..", "data", "e6_optimality_fading.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
