"""E7b summary: derived quantities quoted in the response to Reviewers 1.1b
and 3.4, computed from data/e7b_learned_robustness.csv only (no new
simulation, no seeds).

Lists, for each front end (tokens after or before the 1/h_u division),
the largest relative SER change that twice the width, two attention
layers or twice the training steps cause against the E7-sized network
at 3000 steps, over all affinities and over 0.5 <= beta <= 0.7. Also
lists the smallest ratio of any learned arm to DR-MMSE, the arms that
pass the DR, and the booktabs rows used in the response letter.

Output: data/e7b_summary.txt
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
R = {(r["arm"], r["beta"]): r for r in
     csv.DictReader(open(os.path.join(DATA, "e7b_learned_robustness.csv")))}
BETAS = sorted({b for _, b in R}, key=float)
ARMS = sorted({a for a, _ in R})
f = lambda arm, b, k: float(R[(arm, b)][k])

out = ["E7b summary (from data/e7b_learned_robustness.csv)", ""]
for t in ("div", "prediv"):
    ref = f"LearnedSI_{t}_base_3000"
    others = [a for a in ARMS if a.startswith(f"LearnedSI_{t}_") and a != ref]
    for scope, bs in (("all beta", BETAS),
                      ("0.5<=beta<=0.7", [b for b in BETAS
                                         if 0.5 <= float(b) <= 0.7])):
        ch = [(f(a, b, "ser_mean") / f(ref, b, "ser_mean") - 1, a, b)
              for a in others for b in bs]
        lo, hi = min(ch), max(ch)
        out.append(f"tokens={t} vs {ref}, {scope}: largest decrease "
                   f"{100 * lo[0]:+.1f}% ({lo[1]}, beta={lo[2]}), largest "
                   f"increase {100 * hi[0]:+.1f}% ({hi[1]}, beta={hi[2]})")
out.append("")
m = min((f(a, b, "ratio_to_DRMMSE"), a, b) for a in ARMS for b in BETAS)
out.append(f"smallest ratio to DR-MMSE over all arms and beta: {m[0]:.4f} "
           f"({m[1]}, beta={m[2]})")
below = [(a, b, f(a, b, "ratio_to_DR")) for a in ARMS for b in BETAS
         if f(a, b, "ratio_to_DR") < 1]
out.append("arms with SER below the DR: " + "; ".join(
    f"{a} beta={b} ratio {r:.4f}" for a, b, r in below))
out.append("")
out.append("best arm per beta (ratio to DR / to DR-MMSE) and published arm")
for b in BETAS:
    best = min((f(a, b, "ser_mean"), a) for a in ARMS)
    pub = "LearnedSI_div_base_3000"
    out.append(f"  beta={b}: best {best[1]} {best[0]:.5f} x"
               f"{f(best[1], b, 'ratio_to_DR'):.3f} / x"
               f"{f(best[1], b, 'ratio_to_DRMMSE'):.3f}; published "
               f"{f(pub, b, 'ser_mean'):.5f} x{f(pub, b, 'ratio_to_DR'):.3f}"
               f" / x{f(pub, b, 'ratio_to_DRMMSE'):.3f}; LMMSE "
               f"{f(pub, b, 'LMMSE_ser'):.5f}")
out += ["", "booktabs rows (beta = 0.3, 0.5, 0.7; SER mean +- SD over "
        "five seeds; ratio to DR / to DR-MMSE)"]
fm = {"0.3": "%.4f", "0.5": "%.4f", "0.7": "%.5f"}
lab = {"div": "after", "prediv": "before"}
an = {"base": "E7 size", "wide": "twice the width", "deep": "two layers"}
for t in ("div", "prediv"):
    for a in ("base", "wide", "deep"):
        for s in ("3000", "6000"):
            arm = f"LearnedSI_{t}_{a}_{s}"
            cells = [f"${fm[b] % f(arm, b, 'ser_mean')}\\pm"
                     f"{fm[b] % f(arm, b, 'ser_sd')}$ & "
                     f"{f(arm, b, 'ratio_to_DR'):.2f}/"
                     f"{f(arm, b, 'ratio_to_DRMMSE'):.2f}" for b in fm]
            out.append(f"{lab[t]} & {an[a]} & {s} & " + " & ".join(cells)
                       + r" \\")
with open(os.path.join(DATA, "e7b_summary.txt"), "w") as fh:
    fh.write("\n".join(out) + "\n")
print("\n".join(out))
