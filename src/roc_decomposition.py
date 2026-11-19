"""roc_decomposition.py  |  The ROC-AUC of DRIFT, Isolation Forest and autoencoder in the
four combinations of protocol and generator, from the stored results of
run_experiments.py, next to the values of the submitted version.

Run:  python revise1/src/roc_decomposition.py
"""
import json

import drift_core as dc

ex = json.load(open(dc.OUT_DIR / "experiments.json"))
tr = json.load(open(dc.OUT_DIR / "transductive.json"))
SUBMITTED = {"DRIFT": (0.952, 0.387), "IsolationForest": (0.936, 0.306), "DeepAutoencoder": (0.945, 0.363)}
# G1 and G2L share the span convention, so this decomposition isolates the artifacts and
# the protocol; the observation window (G2L -> G2) is a separate, last step.
cells = {"submitted protocol, G1 (rerun)": tr["G1"], "submitted protocol, G2L": tr["G2L"],
         "held-out protocol, G1": ex["E2_G1"], "held-out protocol, G2L": ex["E2_G2L"]}

print(f"{'setting':34s} " + "  ".join(f"{m[:6]:>16s}" for m in SUBMITTED))
print(f"{'submitted version (as published)':34s} " + "  ".join(f"{r:.3f} / {p:.3f}   " for r, p in SUBMITTED.values()))
for label, E in cells.items():
    print(f"{label:34s} " + "  ".join(
        f"{E[m]['roc_auc']['mean']:.3f} / {E[m]['pr_auc']['mean']:.3f}   " for m in SUBMITTED))
print("(each cell: ROC-AUC / PR-AUC)")

d = lambda a, b: cells[b]["DRIFT"]["roc_auc"]["mean"] - cells[a]["DRIFT"]["roc_auc"]["mean"]
a, b, c, e = list(cells)
print("\nDRIFT ROC-AUC, step by step")
print(f"  environment (submitted 0.952 -> rerun)          {cells[a]['DRIFT']['roc_auc']['mean'] - 0.952:+.3f}")
print(f"  path 1: generator first  G1->G2 (submitted protocol) {d(a, b):+.3f}, then protocol {d(b, e):+.3f}")
print(f"  path 2: protocol first   (G1) {d(a, c):+.3f}, then generator G1->G2 (held-out) {d(c, e):+.3f}")
print(f"  total                                           {cells[e]['DRIFT']['roc_auc']['mean'] - 0.952:+.3f}")
print(f"  then observation window G2L -> G2 (held-out)    "
      f"{ex['E1_main']['DRIFT']['roc_auc']['mean'] - cells[e]['DRIFT']['roc_auc']['mean']:+.3f}")
print("\nper seed, submitted protocol with G1 (rerun):",
      [round(r["DRIFT"]["roc_auc"], 4) for r in tr["G1_per_seed"]])
