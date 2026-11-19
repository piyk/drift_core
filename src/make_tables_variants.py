"""make_tables_variants.py  |  Numeric macros and supplementary tables for the G2/G2L check
and for the two pre-specified DRIFT variants.

Reads   outputs/variants/g2w.json, g2w_per_seed.json   (run_g2w.py)
        outputs/variants/variants.json, variants_per_seed.json, variants_external.json
                                                         (run_variants.py, variants_external.py)
Writes  manuscript/tables/numbers_variants.tex   macros GW... and VAR...
        manuscript/tables/tab_s_g2w.tex, tab_s_variants.tex, tab_s_variants_cont.tex

Run:  python revise1/src/make_tables_variants.py
"""
import json

import numpy as np
from scipy.stats import wilcoxon

import drift_core as dc

OUT = dc.OUT_DIR / "variants"
TAB = dc.ROOT / "revise1" / "manuscript" / "tables"
load = lambda n: json.load(open(OUT / n))
g, gs = load("g2w.json"), load("g2w_per_seed.json")
v, vs, x = load("variants.json"), load("variants_per_seed.json"), load("variants_external.json")
mac = {}
f3 = lambda a: f"{a:.3f}"
f2 = lambda a: f"{a:.2f}"
pc1 = lambda a: f"{100 * a:.1f}"
pval = lambda p: f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def write(name, lines, rule=True):
    (TAB / name).write_text("\n".join(lines) + ("\n\\hline" if rule else "") + "\n", encoding="utf-8")


def wil(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return 1.0 if np.allclose(a, b) else float(wilcoxon(a, b).pvalue)


# ---------------- G2 (primary) against G2L (span over the whole activity) ----------------
# GW... macros describe G2; the suffix Life marks the same quantity on G2L.
E = lambda bench, m, k: g[bench]["eval"][m][k]["mean"]
for key, m in (("drift", "DRIFT"), ("if", "IsolationForest"), ("ae", "DeepAutoencoder"),
               ("heur", "Heuristic"), ("vol", "VolumeReceived")):
    mac[f"GW{key}pr"] = f3(E("G2", m, "pr_auc"))
    mac[f"GW{key}prLife"] = f3(E("G2L", m, "pr_auc"))
    mac[f"GWpeel{key}"] = pc1(E("G2", m, "R@0.05|peeling_chain"))
    mac[f"GWpeel{key}Life"] = pc1(E("G2L", m, "R@0.05|peeling_chain"))
    mac[f"GWrecall{key}"] = pc1(E("G2", m, "R@0.05"))
pr = lambda bench, m: [r[bench]["eval"][m]["pr_auc"] for r in gs]
mac["GWdiffifLife"] = f3(abs(np.mean(np.subtract(pr("G2L", "DRIFT"), pr("G2L", "IsolationForest")))))
mac["GWpifLife"] = pval(wil(pr("G2L", "DRIFT"), pr("G2L", "IsolationForest")))
mac["GWrankae"] = f"{100 * g['G2']['median_rank']['peeling_chain']['DeepAutoencoder']['mean']:.0f}"
mac["GWrankif"] = f"{100 * g['G2']['median_rank']['peeling_chain']['IsolationForest']['mean']:.0f}"
mac["GWrankaeLife"] = f"{100 * g['G2L']['median_rank']['peeling_chain']['DeepAutoencoder']['mean']:.0f}"
mac["GWrankifLife"] = f"{100 * g['G2L']['median_rank']['peeling_chain']['IsolationForest']['mean']:.0f}"
mac["GWdays"] = f"{g['G2']['median_active_days']['peeling_chain']['mean']:.0f}"
mac["GWdaysLife"] = f"{g['G2L']['median_active_days']['peeling_chain']['mean']:.0f}"
mac["GWmatchdrift"] = f3(g["G2"]["matched_auc"]["DRIFT"]["mean"])
mac["GWmatchdriftLife"] = f3(g["G2L"]["matched_auc"]["DRIFT"]["mean"])
typ_short = {"peeling_chain": "peel", "mixer_passthrough": "mixer", "rapid_cashout": "cashout",
             "dormant_burst": "dormant"}

rows = [("Rule-based heuristic", "Heuristic"), ("Received value alone", "VolumeReceived"),
        ("One-Class SVM", "OneClassSVM"), ("Local Outlier Factor", "LOF"), ("PCA reconstruction", "PCA"),
        ("Isolation Forest", "IsolationForest"), ("Deep autoencoder", "DeepAutoencoder"),
        ("DRIFT (mean rank)", "DRIFT"), ("Minimum of ranks", "MinRank")]
lines = []
for label, m in rows:
    vals = [E("G2L", m, "pr_auc"), E("G2", m, "pr_auc"), E("G2L", m, "R@0.05"), E("G2", m, "R@0.05")]
    vals += [E(b, m, f"R@0.05|{t}") for t in typ_short for b in ("G2L", "G2")]
    lines.append(label + " & " + " & ".join(f3(a) for a in vals[:2]) + " & "
                 + " & ".join(pc1(a) for a in vals[2:]) + " \\\\")
write("tab_s_g2w.tex", lines)

# ---------------- variant 1: size-conditional ranks ----------------
M = lambda bench, m, k: v["main"][bench][m][k]["mean"]
mac["VARcprGtwo"] = f3(M("G2", "DRIFT-C", "pr_auc"))
mac["VARcprH"] = f3(M("H", "DRIFT-C", "pr_auc"))
mac["VARcmatchH"] = f3(v["matched_auc"]["H"]["DRIFT-C"]["mean"])
mac["VARdmatchH"] = f3(v["matched_auc"]["H"]["DRIFT"]["mean"])
mac["VARcmatchHp"] = pval(v["criteria"]["C1"]["H"]["p"])
mac["VARcmatchGthree"] = f3(v["matched_auc"]["G3"]["DRIFT-C"]["mean"])
mac["VARdmatchGthree"] = f3(v["matched_auc"]["G3"]["DRIFT"]["mean"])
o, od = x["ofac"]["DRIFT-C"], x["ofac"]["DRIFT"]
mac["VARcofac"] = f2(o["point"]["enrich10_matched"])
mac["VARcofaclo"] = f2(o["ci95_profile"]["enrich10_matched"][0])
mac["VARcofachi"] = f2(o["ci95_profile"]["enrich10_matched"][1])
mac["VARcofacS"] = pc1(o["point"]["top10"])
mac["VARcofacC"] = pc1(o["point"]["top10_ctrl"])
mac["VARcout"] = f3(o["point"]["p_outrank_matched"])
mac["VARdout"] = f3(od["point"]["p_outrank_matched"])
mac["VARcsrv"] = f3(x["services"]["DRIFT-C"]["auc_injected_vs_services"])
mac["VARdsrv"] = f3(x["services"]["DRIFT"]["auc_injected_vs_services"])
mac["VARchnk"] = f3(v["hard_negatives_G2"]["ntx_ge_1000"]["DRIFT-C"]["mean"])
mac["VARdhnk"] = f3(v["hard_negatives_G2"]["ntx_ge_1000"]["DRIFT"]["mean"])
mac["VARucpone"] = f3(M("H", "DRIFT-UC", "P@0.01"))
mac["VARdpone"] = f3(M("H", "DRIFT", "P@0.01"))
mac["VARcmatchHsmallK"] = f3(v["sensitivity"]["K"]["200"]["H"]["matched_auc"]["mean"])
mac["VARcmatchHlargeK"] = f3(v["sensitivity"]["K"]["1000"]["H"]["matched_auc"]["mean"])
mac["VARcexc"] = f"{100 * v['exceedance']['DRIFT-C']['0.95']['mean']:.1f}"
mac["VARdexc"] = f"{100 * v['exceedance']['DRIFT']['0.95']['mean']:.1f}"

# ---------------- variant 2: trimmed reference ----------------
T1 = v["criteria"]["T1"]
d2 = [-T1[f"G2|{c}"]["mean_diff"] for c in ("0.005", "0.01", "0.02", "0.05")]
mac["VARtGtwolo"] = f3(min(d2))
mac["VARtGtwohi"] = f3(max(d2))
mac["VARtHmax"] = f3(max(-T1[f"H|{c}"]["mean_diff"] for c in ("0.005", "0.01", "0.02", "0.05")))
mac["VARtcleanH"] = f3(-v["criteria"]["T2"]["H"]["mean_diff"])
mac["VARtcleanHp"] = pval(v["criteria"]["T2"]["H"]["p"])
mac["VARtdropntx"] = f"{v['trim']['dropped_median_n_tx']['mean']:,.0f}".replace(",", "{,}")
mac["VARtexc"] = f"{100 * v['exceedance']['DRIFT-T']['0.95']['mean']:.1f}"
mac["VARtofac"] = f2(x["ofac"]["DRIFT-T"]["point"]["enrich10_matched"])
mac["VARtsrv"] = f3(x["services"]["DRIFT-T"]["auc_injected_vs_services"])
sq = v["sensitivity"]["q"]
mac["VARtqloclean"] = f3(sq["0.01|clean|G2"]["pr_auc"]["mean"])
mac["VARtqhiclean"] = f3(sq["0.05|clean|G2"]["pr_auc"]["mean"])
mac["VARtqlocont"] = f3(sq["0.01|0.01|G2"]["pr_auc"]["mean"])
mac["VARtqhicont"] = f3(sq["0.05|0.01|G2"]["pr_auc"]["mean"])

lines = []
for label, m in (("Isolation Forest", "IsolationForest"), ("Deep autoencoder", "DeepAutoencoder"),
                 ("DRIFT", "DRIFT"), ("DRIFT-C (conditional ranks)", "DRIFT-C"),
                 ("DRIFT-UC (both kinds of rank)", "DRIFT-UC"), ("DRIFT-T (trimmed reference)", "DRIFT-T"),
                 ("Received value alone", "VolumeReceived")):
    vals = [f3(M(b, m, "pr_auc")) for b in ("G2", "G3", "H")]
    vals += [f3(v["matched_auc"][b][m]["mean"]) for b in ("G3", "H")]
    s = x["services"].get(m)
    vals.append(f3(s["auc_injected_vs_services"]) if s else "n/a")
    o = x["ofac"].get(m)
    vals.append(f2(o["point"]["enrich10_matched"]) if o else "n/a")
    lines.append(label + " & " + " & ".join(vals) + " \\\\")
write("tab_s_variants.tex", lines)

lines = []
for bench in ("G2", "G3", "H"):
    for c in ("0.005", "0.01", "0.02", "0.05"):
        d = v["contaminated"][bench][c]
        t = T1[f"{bench}|{c}"]
        lines.append(f"{bench} & {float(c) * 100:g}\\% & "
                     + " & ".join(f3(d[m]["pr_auc"]["mean"]) for m in ("IsolationForest", "DRIFT", "DRIFT-T"))
                     + f" & {t['mean_diff']:+.3f} ({pval(t['p'])}) \\\\".replace("+", "$+$").replace("-", "$-$", 1))
write("tab_s_variants_cont.tex", lines)

write("numbers_variants.tex", ["\\newcommand{\\" + k + "}{" + val + "}" for k, val in mac.items()], rule=False)
print(f"{len(mac)} macros written")
for k, val in mac.items():
    print(f"  {k} = {val}")
