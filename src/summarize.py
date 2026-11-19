"""summarize.py  |  Print the aggregated experiment results as text tables.

Run:  python revise1/src/summarize.py [experiments.json]
"""
import json
import sys

import drift_core as dc

path = sys.argv[1] if len(sys.argv) > 1 else str(dc.OUT_DIR / "experiments.json")
d = json.load(open(path))
g = lambda x: x["mean"]
sd = lambda x: x["std"]
CORE = ["Heuristic", "IsolationForest", "DeepAutoencoder", "DRIFT", "MaxRank"]


def tab(key, names=None, cols=("roc_auc", "pr_auc", "oracle_f1", "r_precision",
                               "P@0.01", "P@0.05", "R@0.01", "R@0.05", "R@0.1"), E=None):
    E = E if E is not None else d[key]
    print(f"\n== {key}   cols: {' '.join(cols)}")
    for m in (names or E):
        print(f"{m:16s}", "  ".join(f"{g(E[m][c]):.3f}±{sd(E[m][c]):.3f}" for c in cols))


def typ(key, names, tnames, b="0.05", E=None):
    E = E if E is not None else d[key]
    print(f"\n== {key}: per-typology recall at budget {b}   {' '.join(tnames)}")
    for m in names:
        print(f"{m:16s}", "  ".join(f"{g(E[m][f'R@{b}|{t}']):.3f}±{sd(E[m][f'R@{b}|{t}']):.3f}"
                                    for t in tnames))


print("sizes", {k: g(v) for k, v in d["sizes"].items()})
print("fit seconds", round(g(d["fit_seconds"]), 1), " score s/10k", round(g(d["score_seconds_per_10k"]), 2))
tab("E1_main")
for b in ("0.01", "0.02", "0.05", "0.1"):
    typ("E1_main", ["Heuristic", "PCA", "IsolationForest", "DeepAutoencoder", "Cascade", "DRIFT",
                    "MaxRank"], dc.TYPOLOGIES, b)
tab("E1v_val", cols=("roc_auc", "pr_auc", "oracle_f1", "R@0.05"))
for k in ("E2_G1", "E2_G3", "E2_H"):
    tab(k)
for c, E in d["E8_contaminated_fit"].items():
    tab(f"E8_contaminated_fit c={c}", E=E, cols=("roc_auc", "pr_auc", "R@0.05", "R@0.1"))
typ("E2_G1", CORE, dc.TYPOLOGIES)
typ("E2_G3", CORE, dc.TYPOLOGIES)
typ("E2_H", CORE, dc.HELDOUT)

print("\n== rank analysis (E1)")
for t, v in d["E1_rank_analysis"].items():
    print(f"{t:18s}", {k: round(g(x), 3) for k, x in v.items()})
print("component correlation", {k: (round(g(v), 3), round(sd(v), 3)) for k, v in d["E1_component_corr"].items()})

print("\n== hard negatives (ROC-AUC against high-activity real wallets only)")
for k, v in d["E1_hard_negatives"].items():
    if k.startswith("ntx"):
        print(k, {a: round(g(b), 3) for a, b in v.items()})
    else:
        print(k, {a: {c: round(g(e), 3) for c, e in b.items()} for a, b in v.items()})

print("\n== prevalence: roc pr P@0.01 P@0.05 R@0.05")
for c, v in d["E3_prevalence"].items():
    print(f"{c:6s}", " | ".join(m[:6] + " " + " ".join(f"{g(v[m][x]):.3f}" for x in
          ("roc_auc", "pr_auc", "P@0.01", "P@0.05", "R@0.05")) for m in CORE))

print("\n== noise sweep (ROC-AUC) and noise-matched negatives (AUC perturbed vs original, share above p95)")
for l, v in d["E4_noise"].items():
    print(f"{l:10s} G1", {m[:6]: round(g(v["G1"][m]["roc_auc"]), 3) for m in CORE},
          " G2", {m[:6]: round(g(v["G2"][m]["roc_auc"]), 3) for m in CORE})
    for cn in ("ctrl_G1_noise", "ctrl_consistent_noise"):
        print(f"     {cn:22s}", {m[:6]: (round(g(x["auc_perturbed_vs_original"]), 3),
                                         round(g(x["share_above_real_p95"]), 3)) for m, x in v[cn].items()})

print("\n== evasion: (roc, R@0.05)")
for o, v in d["E6_evasion"].items():
    print(f"{o:11s}", {m[:6]: (round(g(v[m]["roc_auc"]), 3), round(g(v[m]["R@0.05"]), 3)) for m in CORE})
for o, v in d["E6_interp"].items():
    print("interp", o, {m[:6]: (round(g(v[m]["roc_auc"]), 3), round(g(v[m]["R@0.05"]), 3)) for m in v})
