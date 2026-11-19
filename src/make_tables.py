"""make_tables.py  |  LaTeX table bodies generated from the result files.

Every number in the tables of the manuscript is written by this script,
so the manuscript cannot drift from the stored results.

Run:  python revise1/src/make_tables.py [ofac_tag]
"""
import json
import sys

import numpy as np

import drift_core as dc

TAB = dc.ROOT / "revise1" / "manuscript" / "tables"
TAB.mkdir(parents=True, exist_ok=True)
d = json.load(open(dc.OUT_DIR / "experiments.json"))
tr = json.load(open(dc.OUT_DIR / "transductive.json"))
g = lambda x: x["mean"]
s = lambda x: x["std"]
f3 = lambda v: f"{v:.3f}"
pm = lambda x: f"{g(x):.3f}\\,$\\pm$\\,{s(x):.3f}"

NAME = {"Heuristic": "Rule-based heuristic", "OneClassSVM": "One-Class SVM",
        "LOF": "Local Outlier Factor", "PCA": "PCA reconstruction",
        "IsolationForest": "Isolation Forest", "DeepAutoencoder": "Deep autoencoder",
        "DRIFT": "DRIFT (mean rank)"}


def bold_best(vals, fmt=f3):
    best = int(np.argmax(vals))
    return [("\\textbf{" + fmt(v) + "}") if i == best else fmt(v) for i, v in enumerate(vals)]


def write(name, lines, rule=True):
    # The closing \hline is part of the file: \hline cannot follow \input in a tabular.
    body = "\n".join(lines) + ("\n\\hline" if rule else "") + "\n"
    (TAB / name).write_text(body, encoding="utf-8")
    print("wrote", name)


# ---------------- generator audit ----------------
rows = ["Heuristic", "IsolationForest", "DeepAutoencoder", "DRIFT"]
# audit chain: G1 (artifacts) -> G2L (artifacts removed, span over the whole activity)
# -> G2 (span measured as the scraper measures real wallets, the primary benchmark)
cols = [[g(tr["G1"][m]["pr_auc"]) for m in rows], [g(tr["G2L"][m]["pr_auc"]) for m in rows],
        [g(d["E2_G1"][m]["pr_auc"]) for m in rows], [g(d["E2_G2L"][m]["pr_auc"]) for m in rows],
        [g(d["E1_main"][m]["pr_auc"]) for m in rows],
        [g(d["E4_noise"]["default"]["ctrl_G1_noise"][m]["auc_perturbed_vs_original"]) for m in rows],
        [g(d["E4_noise"]["default"]["ctrl_consistent_noise"][m]["auc_perturbed_vs_original"]) for m in rows]]
SHORT = {"Heuristic": "Heuristic", "IsolationForest": "Isolation Forest",
         "DeepAutoencoder": "Autoencoder", "DRIFT": "DRIFT"}
write("tab_audit.tex", [f"{SHORT[m]} & " + " & ".join(f3(c[i]) for c in cols) + " \\\\"
                        for i, m in enumerate(rows)])

# ---------------- main comparison ----------------
rows = ["Heuristic", "OneClassSVM", "LOF", "PCA", "IsolationForest", "DeepAutoencoder", "DRIFT"]
# Per-generator results of every row. The volume ranking and the PyOD detectors come
# from run_extensions.py, which uses the same splits and the same injected wallets.
SRC = {m: {"G2": d["E1_main"][m], "G3": d["E2_G3"][m], "H": d["E2_H"][m]} for m in rows}
p_ext = dc.OUT_DIR / "extensions.json"
if p_ext.exists():
    xt = json.load(open(p_ext))
    extra = [("VolumeReceived", "Received value alone", xt["E11_volume"])]
    if "E13_pyod" in xt:
        extra += [(k, lab, xt["E13_pyod"]) for k, lab in (("ECOD", "ECOD"), ("COPOD", "COPOD"),
                                                          ("DIF", "Deep Isolation Forest"),
                                                          ("DeepSVDD", "Deep SVDD"))]
    for k, lab, src in extra:
        NAME[k] = lab
        SRC[k] = {gen: src[gen][k] for gen in ("G2", "G3", "H")}
    added = [k for k, _, _ in extra]
    rows = rows[:1] + added[:1] + rows[1:4] + added[1:] + rows[4:]
E = {m: SRC[m]["G2"] for m in rows}
best_roc = int(np.argmax([g(E[m]["roc_auc"]) for m in rows]))
best_pr = int(np.argmax([g(E[m]["pr_auc"]) for m in rows]))
simple = [("oracle_f1", "G2"), ("P@0.01", "G2"), ("R@0.05", "G2"), ("R@0.1", "G2"),
          ("roc_auc", "G3"), ("pr_auc", "G3"), ("roc_auc", "H"), ("pr_auc", "H")]
cells = [bold_best([g(SRC[m][gen][k]) for m in rows]) for k, gen in simple]
lines = []
for i, m in enumerate(rows):
    roc = pm(E[m]["roc_auc"])
    pr = pm(E[m]["pr_auc"])
    if i == best_roc:
        roc = "\\textbf{" + f3(g(E[m]["roc_auc"])) + "}\\,$\\pm$\\," + f3(s(E[m]["roc_auc"]))
    if i == best_pr:
        pr = "\\textbf{" + f3(g(E[m]["pr_auc"])) + "}\\,$\\pm$\\," + f3(s(E[m]["pr_auc"]))
    lines.append(f"{NAME[m]} & {roc} & {pr} & " + " & ".join(c[i] for c in cells) + " \\\\")
write("tab_main.tex", lines)

# ---------------- aggregation rules ----------------
rules = [("DRIFT", "Mean of ranks (DRIFT)"), ("GeoMeanRank", "Geometric mean of ranks"),
         ("MinRank", "Minimum of ranks"), ("MaxRank", "Maximum of ranks"),
         ("WAvg_AE0.75", "Weighted mean, AE 0.75"), ("WAvg_AE0.25", "Weighted mean, AE 0.25"),
         ("Gated_tau0.5", "Gated, $\\tau=0.50$"), ("Gated_tau0.35", "Gated, $\\tau=0.35$"),
         ("Gated_tau0.2", "Gated, $\\tau=0.20$"), ("Gated_tau0.1", "Gated, $\\tau=0.10$"),
         ("ZMean", "Mean of $z$-scores"), ("ZMax", "Maximum of $z$-scores"),
         ("All5MeanRank", "Five-detector mean rank"), ("Cascade", "Cascade (IF on AE codes)")]
keys = [k for k, _ in rules]
cells = [bold_best([g(d["E1_main"][k]["roc_auc"]) for k in keys]),
         bold_best([g(d["E1_main"][k]["pr_auc"]) for k in keys]),
         bold_best([g(d["E2_G3"][k]["pr_auc"]) for k in keys]),
         bold_best([g(d["E2_H"][k]["pr_auc"]) for k in keys])]
write("tab_fusion.tex", [f"{lab} & " + " & ".join(c[i] for c in cells) + " \\\\"
                         for i, (_, lab) in enumerate(rules)])

# ---------------- evasion ----------------
# The manuscript shows the strongest setting of each operation; make_tables_ext.py writes
# the table with all settings for the supplement.
ops = [("none", "No modification"), ("dilate_30", "Time dilation $\\times 30$"),
       ("retain_0.5", "Retention 50\\%"), ("split_10", "Splitting over 10 addresses"),
       ("split_30", "Splitting over 30 addresses"),
       ("combined", "Combined ($\\times 10$, 30\\%, 10)")]
Ev = d["E6_evasion"]
write("tab_evasion.tex", [
    f"{lab} & {f3(g(Ev[k]['IsolationForest']['roc_auc']))} & {f3(g(Ev[k]['DeepAutoencoder']['roc_auc']))} & "
    f"{f3(g(Ev[k]['DRIFT']['roc_auc']))} & {f3(g(Ev[k]['DRIFT']['R@0.05']))} & "
    f"{f3(g(Ev[k]['Heuristic']['R@0.05']))} \\\\" for k, lab in ops])

# ---------------- OFAC ----------------
tag = sys.argv[1] if len(sys.argv) > 1 else "full"
path = dc.OUT_DIR / f"ofac_external_{tag}.json"
if path.exists():
    o = json.load(open(path))
    rows = ["Heuristic", "OneClassSVM", "LOF", "PCA", "IsolationForest", "DeepAutoencoder", "DRIFT"]
    lines = []
    for m in rows:
        x = o["detectors"][m]
        p, ci = x["point"], x["ci95"]
        lines.append(
            f"{NAME[m]} & {p['median_pct']:.1f} & {100 * p['top10']:.1f} "
            f"({100 * ci['top10'][0]:.1f}--{100 * ci['top10'][1]:.1f}) & "
            f"{100 * x['top10_same_type']:.1f} & {100 * p['top10_ctrl']:.1f} & "
            f"{p['enrich10_matched']:.2f} ({ci['enrich10_matched'][0]:.2f}--{ci['enrich10_matched'][1]:.2f}) & "
            f"{p['p_outrank_matched']:.3f} ({ci['p_outrank_matched'][0]:.3f}--{ci['p_outrank_matched'][1]:.3f}) \\\\")
    write("tab_s_ofac_full.tex", lines)       # all intervals, for the supplement
    # single-column table of the article: the interval of the matched enrichment only
    short = {"Heuristic": "Heuristic", "OneClassSVM": "One-Class SVM", "LOF": "LOF", "PCA": "PCA",
             "IsolationForest": "Isolation Forest", "DeepAutoencoder": "Autoencoder", "DRIFT": "DRIFT"}
    lines = []
    for m in rows:
        x = o["detectors"][m]
        p, ci = x["point"], x["ci95"]
        lines.append(
            f"{short[m]} & {p['median_pct']:.1f} & {100 * p['top10']:.1f} & "
            f"{100 * x['top10_same_type']:.1f} & {100 * p['top10_ctrl']:.1f} & "
            f"{p['enrich10_matched']:.2f} ({ci['enrich10_matched'][0]:.2f}--{ci['enrich10_matched'][1]:.2f}) & "
            f"{p['p_outrank_matched']:.3f} \\\\")
    write("tab_ofac.tex", lines)

    # numeric macros used in the prose of the external check
    D = o["detectors"]
    pt = lambda m, k: D[m]["point"][k]
    ci = lambda m, k, i: D[m]["ci95"][k][i]
    st = o["script_types_scored"]
    sc = o["script_types_corpus"]
    n_corpus = o["corpus_size"]
    de = o["describe"]
    mac = {
        "OFACn": f"{o['n_scored']}",
        "OFACinactive": f"{o['n_without_activity']}",
        "OFACmedian": f"{pt('DRIFT', 'median_pct'):.1f}",
        "OFACtop": f"{100 * pt('DRIFT', 'top10'):.1f}",
        "OFACtoplo": f"{100 * ci('DRIFT', 'top10', 0):.1f}",
        "OFACtophi": f"{100 * ci('DRIFT', 'top10', 1):.1f}",
        "OFACenrich": f"{pt('DRIFT', 'enrich10_corpus'):.1f}",
        "OFACsame": f"{100 * D['DRIFT']['top10_same_type']:.1f}",
        "OFACctrl": f"{100 * pt('DRIFT', 'top10_ctrl'):.1f}",
        "OFACmatched": f"{pt('DRIFT', 'enrich10_matched'):.2f}",
        "OFACmatchedlo": f"{ci('DRIFT', 'enrich10_matched', 0):.2f}",
        "OFACmatchedhi": f"{ci('DRIFT', 'enrich10_matched', 1):.2f}",
        "OFACout": f"{pt('DRIFT', 'p_outrank_matched'):.3f}",
        "OFACmatchedfive": f"{pt('DRIFT', 'enrich5_matched'):.2f}",
        "OFACmatchedone": f"{pt('DRIFT', 'enrich1_matched'):.2f}",
        "OFACmatchedIF": f"{pt('IsolationForest', 'enrich10_matched'):.2f}",
        "OFACmatchedAE": f"{pt('DeepAutoencoder', 'enrich10_matched'):.2f}",
        "OFACmatchedHeur": f"{pt('Heuristic', 'enrich10_matched'):.2f}",
        "OFACmatchedLOF": f"{pt('LOF', 'enrich10_matched'):.2f}",
        "OFACoutmax": f"{max(pt(m, 'p_outrank_matched') for m in rows):.3f}",
        "OFACabovemedian": f"{o['drift_above_corpus_median']}",
        "OFACrecvS": f"{de['sanctioned']['median_received_btc']:.1f}",
        "OFACrecvC": f"{de['corpus']['median_received_btc']:.3f}",
        "OFACzeroS": f"{100 * de['sanctioned']['zero_balance_share']:.1f}",
        "OFACzeroM": f"{100 * de['matched_controls']['zero_balance_share']:.1f}",
        "OFACsmdmaxbefore": f"{max(abs(v['before']) for v in o['smd'].values()):.2f}",
        "OFACsmdmaxafter": f"{max(abs(v['after']) for v in o['smd'].values()):.2f}",
    }
    # TeX macro names cannot contain digits, hence the short type names
    tname = {"Legacy": "Pkh", "P2SH": "Sh", "P2WPKH": "Wpkh", "P2WSH": "Wsh", "Taproot": "Tr"}
    for t, a in tname.items():
        mac["OFACn" + a] = f"{st.get(t, 0)}"
        mac["OFACcorpus" + a] = f"{100 * sc.get(t, 0) / n_corpus:.1f}"
        v = D["DRIFT"]["by_type"].get(t)
        mac["OFACtop" + a] = f"{100 * v['top10']:.1f}" if v else "n/a"
        mac["OFACctrl" + a] = f"{100 * v['top10_ctrl']:.1f}" if v else "n/a"
    # list provenance (when the sample was restricted to the SDN-derived list)
    li = o.get("list") or {}
    if li:
        mac.update({"OFAClistentries": f"{li['n_list_entries']}",
                    "OFAClistbtc": f"{li['n_bitcoin_addresses']}",
                    "OFACwithfeatures": f"{li['n_with_features']}",
                    "OFACnofeatures": f"{len(li['without_features'])}"})
    mac["OFACbelowmedian"] = f"{o['n_scored'] - o['drift_above_corpus_median']}"
    mac["OFACabovemedianpct"] = f"{100 * o['drift_above_corpus_median'] / o['n_scored']:.0f}"
    mac["OFAColdshare"] = f"{100 * (st.get('Legacy', 0) + st.get('P2SH', 0)) / o['n_scored']:.0f}"
    mac["OFACtypeshare"] = f"{100 * D['DRIFT']['top10_same_type'] / 10:.1f}"   # same-type enrichment over 10%
    for m, a in (("IsolationForest", "IF"), ("DeepAutoencoder", "AE")):
        mac["OFACtop" + a] = f"{100 * pt(m, 'top10'):.1f}"
    # ROC-AUC of DRIFT in the four combinations of protocol and generator, and the
    # contribution of each generator artefact (roc_recheck.py)
    # artifact and protocol decomposition on the shared span convention (G1 -> G2L)
    r_sub1, r_sub2 = g(tr["G1"]["DRIFT"]["roc_auc"]), g(tr["G2L"]["DRIFT"]["roc_auc"])
    r_held1, r_held2 = g(d["E2_G1"]["DRIFT"]["roc_auc"]), g(d["E2_G2L"]["DRIFT"]["roc_auc"])
    mac.update({"ROCsubGone": f3(r_sub1), "ROCsubGtwo": f3(r_sub2),
                "ROCheldGone": f3(r_held1), "ROCheldGtwo": f3(r_held2),
                "ROCdropart": f"{r_sub1 - r_sub2:.2f}", "ROCgainproto": f"{r_held2 - r_sub2:.2f}"})
    p4 = dc.OUT_DIR / "roc_recheck.json"
    if p4.exists():
        rr = json.load(open(p4))["transductive"]
        mac.update({"ARTnoise": f3(rr["noise"]["DRIFT"]["roc"]["mean"]),
                    "ARTunc": f3(rr["unc"]["DRIFT"]["roc"]["mean"]),
                    "ARTtype": f3(rr["type"]["DRIFT"]["roc"]["mean"])})
    # independent recheck without matching (ofac_recheck.py)
    p3 = dc.OUT_DIR / "ofac_recheck.json"
    if p3.exists() and tag != "v7_356":
        rk = json.load(open(p3))
        strata = {r["bin"]: r for r in rk["strata_received"]}
        # alternative designs on the same covariates; the design that also matches on an empty
        # balance is a separate check of the observation time and is reported on its own
        others = [v["matched_enrichment"] for k, v in rk["matching"].items()
                  if "(paper)" not in k and "empty" not in k]
        mac.update({
            "RCKqninety": f"{rk['volume']['corpus_q90_btc']:.2f}",
            "RCKabove": f"{100 * rk['volume']['sanctioned_above_corpus_q90']:.1f}",
            "RCKcorpusmid": f"{100 * strata['10-100']['corpus_top10']:.1f}",
            "RCKcorpushigh": f"{100 * strata['>= 100']['corpus_top10']:.1f}",
            "RCKexpected": f"{100 * rk['standardised']['received value']['expected_top10']:.1f}",
            "RCKmatchlo": f"{min(others):.2f}",
            "RCKmatchhi": f"{max(others):.2f}",
            "RCKdesigns": {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
                           8: "eight", 9: "nine"}.get(len(others), str(len(others))),
        })
    # the 356 addresses of the submitted version (continuity check)
    p2 = dc.OUT_DIR / "ofac_external_v7_356.json"
    if p2.exists() and tag != "v7_356":
        o2 = json.load(open(p2))
        x2 = o2["detectors"]["DRIFT"]
        mac.update({
            "SUBn": f"{o2['n_scored']}",
            "SUBtop": f"{100 * x2['point']['top10']:.1f}",
            "SUBctrl": f"{100 * x2['point']['top10_ctrl']:.1f}",
            "SUBmatched": f"{x2['point']['enrich10_matched']:.2f}",
            "SUBmatchedlo": f"{x2['ci95']['enrich10_matched'][0]:.2f}",
            "SUBmatchedhi": f"{x2['ci95']['enrich10_matched'][1]:.2f}",
        })
    write("numbers.tex", ["\\newcommand{\\" + k + "}{" + v + "}" for k, v in mac.items()], rule=False)
else:
    print("no OFAC results for tag", tag)
