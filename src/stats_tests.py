"""stats_tests.py  |  Paired comparisons across seeds and the transductive table.

Each seed uses the same split and the same test anomalies for every
detector, so detectors are compared with paired tests over the ten seeds
(two-sided Wilcoxon signed-rank test, and the number of seeds won).

Run:  python revise1/src/stats_tests.py
"""
import json

import numpy as np
from scipy.stats import wilcoxon

import drift_core as dc

per = json.load(open(dc.OUT_DIR / "experiments_per_seed.json"))
out = {}


def paired(exp, a, b, metric):
    x = np.array([r[exp][a][metric] for r in per])
    y = np.array([r[exp][b][metric] for r in per])
    d = x - y
    try:
        p = float(wilcoxon(d).pvalue)
    except ValueError:
        p = float("nan")
    return {"mean_diff": float(d.mean()), "sd_diff": float(d.std(ddof=1)),
            "wins": int((d > 0).sum()), "n": int(len(d)), "wilcoxon_p": p}


PAIRS = [("DRIFT", "IsolationForest"), ("DRIFT", "DeepAutoencoder"), ("DRIFT", "Heuristic"),
         ("DRIFT", "MinRank"), ("DRIFT", "MaxRank"), ("DRIFT", "GeoMeanRank"),
         ("DRIFT", "Gated_tau0.35"), ("DRIFT", "ZMean"), ("DRIFT", "All5MeanRank"),
         ("DRIFT", "Cascade"), ("IsolationForest", "DeepAutoencoder")]
for exp in ("E1_main", "E2_G1", "E2_G3", "E2_H"):
    out[exp] = {}
    print(f"\n== {exp}: paired differences over {len(per)} seeds (A minus B)")
    for a, b in PAIRS:
        row = {}
        for metric in ("roc_auc", "pr_auc", "R@0.05"):
            row[metric] = paired(exp, a, b, metric)
        out[exp][f"{a}_vs_{b}"] = row
        print(f"{a:16s} vs {b:16s} " + "  ".join(
            f"{m}: {row[m]['mean_diff']:+.4f} (wins {row[m]['wins']}/{row[m]['n']}, p={row[m]['wilcoxon_p']:.3f})"
            for m in row))

# validation-based selection of the aggregation rule
rules = ["DRIFT", "MaxRank", "MinRank", "GeoMeanRank", "WAvg_AE0.75", "WAvg_AE0.25", "ZMean",
         "ZMax", "All5MeanRank", "Gated_tau0.5", "Gated_tau0.35", "Gated_tau0.2", "Gated_tau0.1"]
sel = {}
for metric in ("pr_auc", "roc_auc"):
    best = [max(rules, key=lambda m: r["E1v_val"][m][metric]) for r in per]
    sel[metric] = {m: best.count(m) for m in set(best)}
print("\nrule selected on the validation split, count over seeds:", sel)
gated = ["DRIFT", "Gated_tau0.5", "Gated_tau0.35", "Gated_tau0.2", "Gated_tau0.1"]
best_tau = [max(gated, key=lambda m: r["E1v_val"][m]["pr_auc"]) for r in per]
print("temperature selected on validation (PR-AUC):", {m: best_tau.count(m) for m in set(best_tau)})
out["validation_selection"] = {"all_rules": sel, "gated_family": {m: best_tau.count(m) for m in set(best_tau)}}

# test performance of the rule that the validation split selects in each seed
chosen = [max(rules, key=lambda m: r["E1v_val"][m]["pr_auc"]) for r in per]
vs = {}
for exp in ("E1_main", "E2_G3", "E2_H"):
    row = {}
    for metric in ("roc_auc", "pr_auc"):
        x = np.array([r[exp][c][metric] for r, c in zip(per, chosen)])
        d = x - np.array([r[exp]["DRIFT"][metric] for r in per])
        try:
            p = float(wilcoxon(d).pvalue)
        except ValueError:
            p = float("nan")
        row[metric] = {"mean": float(x.mean()), "std": float(x.std()), "mean_diff_vs_mean_rank": float(d.mean()),
                       "wins": int((d > 0).sum()), "wilcoxon_p": p}
    vs[exp] = row
    print(f"validation-selected rule on {exp}: PR-AUC {row['pr_auc']['mean']:.3f} "
          f"({row['pr_auc']['mean_diff_vs_mean_rank']:+.4f} against the mean rank, p={row['pr_auc']['wilcoxon_p']:.3f})")
out["validation_selected_rule"] = vs

# Holm correction over the family of PR-AUC comparisons that the manuscript reports:
# DRIFT against each component, the minimum of ranks and the gated rule, on three benchmarks
family = [(exp, b) for exp in ("E1_main", "E2_G3", "E2_H")
          for b in ("IsolationForest", "DeepAutoencoder", "MinRank", "Gated_tau0.35")]
raw_p = np.array([out[exp][f"DRIFT_vs_{b}"]["pr_auc"]["wilcoxon_p"] for exp, b in family])
order = np.argsort(raw_p)
adj = np.empty(len(raw_p))
running = 0.0
for i, j in enumerate(order):
    running = max(running, (len(raw_p) - i) * raw_p[j])
    adj[j] = min(running, 1.0)
out["holm_pr_auc_family"] = {f"{exp}|DRIFT_vs_{b}": {"p": float(p), "p_holm": float(a)}
                             for (exp, b), p, a in zip(family, raw_p, adj)}
sig = [a for p, a in zip(raw_p, adj) if p < 0.05]
out["holm_summary"] = {"n_tests": int(len(raw_p)), "n_raw_below_0.05": int((raw_p < 0.05).sum()),
                       "n_holm_below_0.05": int((adj < 0.05).sum()),
                       "max_holm_among_raw_significant": float(max(sig)) if sig else float("nan")}
print("Holm correction:", out["holm_summary"])

trans = json.load(open(dc.OUT_DIR / "transductive.json"))
print("\n== transductive protocol of the submission (5 seeds): roc pr oracle_f1 r_precision | recall@5% per typology")
for g in ("G1", "G2L", "G2"):
    print(g)
    for m in ["Heuristic", "OneClassSVM", "LOF", "PCA", "IsolationForest", "DeepAutoencoder",
              "DRIFT", "MaxRank", "MinRank", "Gated_tau0.35", "Gated_tau0.1"]:
        e = trans[g][m]
        print(f"  {m:16s}", "  ".join(f"{e[c]['mean']:.3f}±{e[c]['std']:.3f}"
                                      for c in ("roc_auc", "pr_auc", "oracle_f1", "r_precision")),
              " | ", "  ".join(f"{e['R@0.05|' + t]['mean']:.3f}" for t in dc.TYPOLOGIES))

(dc.OUT_DIR / "paired_tests.json").write_text(json.dumps(out, indent=1))
print(f"\n[save] {dc.OUT_DIR / 'paired_tests.json'}")
