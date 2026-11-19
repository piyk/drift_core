"""make_tables_features.py  |  Numeric macros and supplementary tables of the pre-specified
feature revision (revise1/features_prespecification.md).

Reads   outputs/features/features.json, outputs/features/parts/*.json   (run_features.py)
        outputs/features/features_external.json                         (features_external.py)
        the raw corpus (source of each record, for the observation behind change 1)
Writes  manuscript/tables/numbers_features.tex   macros FT...
        manuscript/tables/tab_s_feat_*.tex
Checks  the qualitative statements of the supplementary section against the results
        (outputs/features/features_claims_check.json); exits with status 1 if one fails.

Run:  python revise1/src/make_tables_features.py
"""
import json
import sys

import numpy as np

import drift_core as dc
import features_v2 as fv

OUT = dc.OUT_DIR / "features"
TAB = dc.ROOT / "revise1" / "manuscript" / "tables"
F = json.load(open(OUT / "features.json"))
X = json.load(open(OUT / "features_external.json"))
R, CR = F["results"], F["criteria"]
mac = {}
f3 = lambda a: f"{a:.3f}"
f2 = lambda a: f"{a:.2f}"
pc0 = lambda a: f"{100 * a:.0f}"
pc1 = lambda a: f"{100 * a:.1f}"
pval = lambda p: "0.002" if p < 0.0025 else (f"{p:.3f}" if p < 0.1 else f"{p:.2f}")
LETTERS = {"F21": "Ftw", "W21": "Wtw", "B": "B", "BR": "BR", "BR-u": "BRu"}
SET_LABEL = {"F21": "21 features (article)", "W21": "W21: window rate and burst", "B": "B: basis of five",
             "BR": "BR: basis and size-relative", "BR-u": "BR-u: BR without winsorization"}
DET_LABEL = {"Heuristic": "Rule-based heuristic", "VolumeReceived": "Received value alone",
             "OneClassSVM": "One-Class SVM", "LOF": "Local Outlier Factor", "PCA": "PCA reconstruction",
             "ECOD": "ECOD", "COPOD": "COPOD", "DIF": "Deep Isolation Forest", "DeepSVDD": "Deep SVDD",
             "IsolationForest": "Isolation Forest", "DeepAutoencoder": "Deep autoencoder",
             "Cascade": "Cascade", "All5MeanRank": "Mean rank of five", "MinRank": "Minimum of ranks",
             "DRIFT": "DRIFT (mean rank)", "DRIFT-S": "DRIFT-S (ranks within script type)"}
BENCH = [("G2", "G2"), ("G3", "G3"), ("H", "H"), ("Hp", "H$'$")]


def write(name, lines):
    (TAB / name).write_text("\n".join(lines) + "\n\\hline\n", encoding="utf-8")


def m_(s, kind, g, m, k=None):
    d = R[s][kind][g][m] if k is None else R[s][kind][g][m][k]
    return d["mean"]


pr = lambda s, g, m: m_(s, "main", g, m, "pr_auc")
roc = lambda s, g, m: m_(s, "main", g, m, "roc_auc")
mat = lambda s, g, m: m_(s, "matched_auc", g, m)
hn = lambda s, g, m, t="ntx_ge_100": R[s]["hard_negatives"][g][t][m]["mean"]
ext = X["sets"]

# ---------------- the observation behind change 1 (raw corpus, no detector) ----------------
raw = dc.load_data()
n = raw["n_tx"].values.astype(float)
bc = ((raw["avg_tx_value"].fillna(0) != 0) | (raw["max_tx_value"].fillna(0) != 0)).values
big = n > 50
mac["FTncorpus"] = f"{len(raw):,}"
mac["FTnbc"] = f"{int(bc.sum()):,}"
mac["FTbcpct"] = pc1(bc.mean())
mac["FTbcpctint"], mac["FTlifepctint"] = pc0(bc.mean()), pc0(1.0 - bc.mean())
mac["FTnbig"] = f"{int(big.sum()):,}"
mac["FTnbiglife"] = f"{int((big & ~bc).sum()):,}"
mac["FTnbigbc"] = f"{int((big & bc).sum()):,}"
infl = n[big & ~bc] / 50.0
mac["FTinflmed"] = f"{np.median(infl):.1f}"
mac["FTinflpn"] = f"{np.percentile(infl, 90):.0f}"
prod = (raw["tx_per_day"] * raw["active_days"]).values
mac["FTbcrefsabove"] = pc0(float((prod[bc] > n[bc] + 1e-6).mean()))

# ---------------- the 21-feature column must equal the stored results of the article ----------------
st = json.load(open(dc.OUT_DIR / "experiments.json"))
ex = json.load(open(dc.OUT_DIR / "extensions.json"))
oo = json.load(open(dc.OUT_DIR / "ofac_external_sdn.json"))["detectors"]["DRIFT"]
cf = json.load(open(dc.OUT_DIR / "ofac_crossfit_sdn.json"))
bb = json.load(open(dc.OUT_DIR / "babd_external.json"))
repro = []
for g, key in (("G2", "E1_main"), ("G3", "E2_G3"), ("H", "E2_H")):
    for m in ("IsolationForest", "DeepAutoencoder", "DRIFT", "All5MeanRank", "MinRank"):
        repro.append((f"{g} {m} PR-AUC", pr("F21", g, m), st[key][m]["pr_auc"]["mean"]))
    for m in ("ECOD", "COPOD", "DIF", "DeepSVDD"):
        repro.append((f"{g} {m} PR-AUC", pr("F21", g, m), ex["E13_pyod"][g][m]["pr_auc"]["mean"]))
    repro.append((f"{g} DRIFT matched", mat("F21", g, "DRIFT"), ex["E12_matched"][g]["DRIFT"]["mean"]))
repro.append(("G2 DRIFT hard negatives", hn("F21", "G2", "DRIFT"), st["E1_hard_negatives"]["ntx_ge_100"]["DRIFT"]["mean"]))
e21 = ext["F21"]
repro.append(("OFAC in-sample top10", e21["ofac_insample"]["DRIFT"]["point"]["top10"], oo["point"]["top10"]))
repro.append(("OFAC in-sample control top10", e21["ofac_insample"]["DRIFT"]["point"]["top10_ctrl"], oo["point"]["top10_ctrl"]))
repro.append(("OFAC same type", e21["ofac_insample"]["DRIFT"]["top10_same_type"], oo["top10_same_type"]))
repro.append(("OFAC cross-fitted enrichment", e21["ofac_crossfit"]["DRIFT"]["point"]["enrich10_matched"],
              cf["cross_fitted"]["detectors"]["DRIFT"]["point"]["enrich10_matched"]))
repro.append(("OFAC cross-fitted, balance-matched", e21["ofac_crossfit_zero_balance"]["DRIFT"]["point"]["enrich10_matched"],
              cf["cross_fitted_zero_balance"]["detectors"]["DRIFT"]["point"]["enrich10_matched"]))
repro.append(("services AUC", e21["services"]["DRIFT"]["auc_injected_vs_services"],
              bb["high_activity_services"]["detectors"]["DRIFT"]["auc_injected_vs_service"]))
repro.append(("BABD P2PKH", e21["babd_sample1"]["within"]["Legacy"]["DRIFT"]["auc"],
              bb["auc_within_script_type"]["Legacy"]["DRIFT"]["auc"]))
bad = [(k, a, b) for k, a, b in repro if abs(a - b) > 1e-12]
print(f"21-feature column against the stored results: {len(repro)} values, {len(bad)} differ")
for k, a, b in bad:
    print(f"  differs: {k}: {a} (this run) against {b} (stored)")
mac["FTrepron"] = str(len(repro))

# ---------------- design facts ----------------
mac["FTspecsha"] = F["spec_sha256"][:12]
assert X["spec_sha256"] == F["spec_sha256"] == fv.spec_digest(), "results do not belong to the frozen specification"
mac["FTbins"] = f"{R['BR']['count_bins']['mean']:.0f}"
mac["FTminbin"] = f"{R['BR']['min_bin_size']['mean']:.0f}"

# ---------------- criteria ----------------
S1, S3, S4, S2 = CR["S1"], CR["S3"], CR["S4"], X["criteria"]["S2"]
yes = lambda b: "yes" if b else "no"
rows = []
for g, gl in (("G3", "G3"), ("H", "H"), ("Hp", "H$'$")):
    c = S1[g]
    rows.append(f"S1: matched AUC & {gl} & {f3(c['mean_b'])} & {f3(c['mean_a'])} & ${c['mean_diff']:+.3f}$ & "
                f"{pval(c['p_holm'])} & {yes(c['holds'])} \\\\")
rows.append(f"S2: AUC against verified services & G2 & {f3(S2['auc_F21'])} & {f3(S2['auc_BR'])} & "
            f"${S2['diff']:+.3f}$ & n/a & {yes(S2['holds'])} \\\\")
rows.append(f"S3: ROC-AUC, $\\geq$100 transactions & G2 & {f3(S3['mean_b'])} & {f3(S3['mean_a'])} & "
            f"${S3['mean_diff']:+.3f}$ & {pval(S3['p'])} & {yes(S3['holds'])} \\\\")
s4m =[m for m in S4 if m != "holds"]
nmet = sum(S4[m]["holds"] for m in s4m)
rows.append(f"S4: PR-AUC above all nine detectors & H$'$ & n/a & {f3(S4[s4m[0]]['mean_a'])} & "
            f"{nmet} of 9 & see Table~\\ref{{tab:s-feat-s4}} & {yes(S4['holds'])} \\\\")
write("tab_s_feat_criteria.tex", rows)
for g, key in (("G3", "Gthree"), ("H", "H"), ("Hp", "Hp")):
    c = S1[g]
    mac[f"FTsone{key}old"], mac[f"FTsone{key}new"] = f3(c["mean_b"]), f3(c["mean_a"])
    mac[f"FTsone{key}diff"] = f"{c['mean_diff']:+.3f}"
    mac[f"FTsone{key}p"] = pval(c["p_holm"])
    mac[f"FTsone{key}wins"] = str(c["wins"])
mac["FTstwoold"], mac["FTstwonew"] = f3(S2["auc_F21"]), f3(S2["auc_BR"])
mac["FTstwodiff"] = f"{S2['diff']:+.3f}"
mac["FTsthreeold"], mac["FTsthreenew"] = f3(S3["mean_b"]), f3(S3["mean_a"])
mac["FTsthreep"] = pval(S3["p"])
mac["FTsfourdrift"] = f3(S4[s4m[0]]["mean_a"])
mac["FTsfourmet"] = str(nmet)
best = max(s4m, key=lambda m: S4[m]["mean_b"])
mac["FTsfourbest"] = DET_LABEL[best]
mac["FTsfourbestpr"] = f3(S4[best]["mean_b"])
crit = {"S1": S1["holds"], "S2": S2["holds"], "S3": S3["holds"], "S4": S4["holds"]}
mac["FTnmet"] = str(sum(crit.values()))
all_met = all(crit.values())

rows = []
for m in s4m:
    c = S4[m]
    rows.append(f"{DET_LABEL[m]} & {f3(c['mean_b'])} & ${c['mean_diff']:+.3f}$ & {c['wins']} & "
                f"{pval(c['p'])} & {pval(c['p_holm'])} \\\\")
write("tab_s_feat_s4.tex", rows)

# ---------------- DRIFT under every feature set ----------------
rows = []
for s in fv.SETS:
    vals = [pr(s, g, "DRIFT") for g, _ in BENCH] + [mat(s, g, "DRIFT") for g in ("G3", "H", "Hp")]
    vals += [hn(s, "G2", "DRIFT"), hn(s, "G2", "DRIFT", "ntx_ge_1000"),
             ext[s]["services"]["DRIFT"]["auc_injected_vs_services"]]
    rows.append(f"{SET_LABEL[s]} & " + " & ".join(f3(v) for v in vals) + " \\\\")
s = "BR"
vals = [pr(s, g, "DRIFT-S") for g, _ in BENCH] + [mat(s, g, "DRIFT-S") for g in ("G3", "H", "Hp")]
vals += [hn(s, "G2", "DRIFT-S"), hn(s, "G2", "DRIFT-S", "ntx_ge_1000"),
         ext[s]["services"]["DRIFT-S"]["auc_injected_vs_services"]]
rows.append("DRIFT-S on BR & " + " & ".join(f3(v) for v in vals) + " \\\\")
vals = [pr("F21", g, "VolumeReceived") for g, _ in BENCH] + [mat("F21", g, "VolumeReceived") for g in ("G3", "H", "Hp")]
vals += [hn("F21", "G2", "VolumeReceived"), hn("F21", "G2", "VolumeReceived", "ntx_ge_1000"),
         ext["F21"]["services"]["VolumeReceived"]["auc_injected_vs_services"]]
rows.append("Received value alone & " + " & ".join(f3(v) for v in vals) + " \\\\")
write("tab_s_feat_sets.tex", rows)
for s in fv.SETS:
    L = LETTERS[s]
    for g, gk in (("G2", "Gtwo"), ("G3", "Gthree"), ("H", "H"), ("Hp", "Hp")):
        mac[f"FT{L}pr{gk}"] = f3(pr(s, g, "DRIFT"))
        mac[f"FT{L}roc{gk}"] = f3(roc(s, g, "DRIFT"))
        mac[f"FT{L}mat{gk}"] = f3(mat(s, g, "DRIFT"))
    mac[f"FT{L}hn"] = f3(hn(s, "G2", "DRIFT"))
    mac[f"FT{L}hnk"] = f3(hn(s, "G2", "DRIFT", "ntx_ge_1000"))
    mac[f"FT{L}fp"] = pc0(R[s]["fp_at_5pct"]["G2"]["DRIFT"]["mean"])
    mac[f"FT{L}srv"] = f3(ext[s]["services"]["DRIFT"]["auc_injected_vs_services"])
    mac[f"FT{L}exc"] = pc1(R[s]["exceedance_p95"]["DRIFT"]["mean"])
for g, gk in (("G2", "Gtwo"), ("G3", "Gthree"), ("H", "H"), ("Hp", "Hp")):
    mac[f"FTvolpr{gk}"] = f3(pr("F21", g, "VolumeReceived"))
    mac[f"FTvolmat{gk}"] = f3(mat("F21", g, "VolumeReceived"))

# ---------------- every detector, 21 features against BR ----------------
DETS = ["Heuristic", "VolumeReceived", "OneClassSVM", "LOF", "PCA", "ECOD", "COPOD", "DIF", "DeepSVDD",
        "IsolationForest", "DeepAutoencoder", "Cascade", "All5MeanRank", "MinRank", "DRIFT"]


def pair_rows(fn, gs):
    out = []
    for m in DETS:
        cells = []
        for g in gs:
            a, b = fn("F21", g, m), fn("BR", g, m)
            cells += [f3(a), f3(b)]
        out.append(f"{DET_LABEL[m]} & " + " & ".join(cells) + " \\\\")
    return out


def bold_best(lines, ncol):
    """Bold the highest value of each column, among the rows of this table."""
    vals = [[float(c) for c in l.rstrip(" \\").split(" & ")[1:]] for l in lines]
    best = [max(v[j] for v in vals) for j in range(ncol)]
    out = []
    for l, v in zip(lines, vals):
        head = l.split(" & ")[0]
        cells = [("\\textbf{%s}" % f3(x)) if abs(x - best[j]) < 5e-4 else f3(x) for j, x in enumerate(v)]
        out.append(head + " & " + " & ".join(cells) + " \\\\")
    return out


write("tab_s_feat_pr.tex", bold_best(pair_rows(pr, ["G2", "G3", "H", "Hp"]), 8))
write("tab_s_feat_matched.tex", bold_best(
    pair_rows(lambda s, g, m: mat(s, g, m) if g != "HN" else hn(s, "G2", m), ["G3", "H", "Hp", "HN"]), 8))
for m, key in (("IsolationForest", "if"), ("DeepAutoencoder", "ae"), ("ECOD", "ecod"), ("COPOD", "copod"),
               ("All5MeanRank", "allfive"), ("MinRank", "min"), ("DIF", "dif"), ("OneClassSVM", "svm"),
               ("LOF", "lof"), ("PCA", "pca"), ("DeepSVDD", "svdd")):
    for s in ("F21", "BR"):
        L = LETTERS[s]
        mac[f"FT{L}{key}prGtwo"] = f3(pr(s, "G2", m))
        mac[f"FT{L}{key}prHp"] = f3(pr(s, "Hp", m))
        mac[f"FT{L}{key}matHp"] = f3(mat(s, "Hp", m))
        mac[f"FT{L}{key}hn"] = f3(hn(s, "G2", m))
mac["FTheurprHp"] = f3(pr("F21", "Hp", "Heuristic"))

# ---------------- per-typology recall on H' at the 5% budget ----------------
rows = []
for m in ("VolumeReceived", "Heuristic", "IsolationForest", "DeepAutoencoder", "DRIFT"):
    for s in (("F21",) if m in ("VolumeReceived", "Heuristic") else ("F21", "BR")):
        short = {"IsolationForest": "Isolation Forest", "DeepAutoencoder": "Deep autoencoder", "DRIFT": "DRIFT"}
        lab = DET_LABEL[m] if m in ("VolumeReceived", "Heuristic") else (
            f"{short[m]}, " + ("21 features" if s == "F21" else "BR"))
        cells = [pc0(m_(s, "main", "Hp", m, f"R@0.05|{t}")) for t in fv.HPRIME] + [pc0(m_(s, "main", "Hp", m, "R@0.05"))]
        rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
write("tab_s_feat_htyp.tex", rows)
for t, key in zip(fv.HPRIME, ("ransom", "donation", "escrow", "otc")):
    for s in ("F21", "BR"):
        mac[f"FT{LETTERS[s]}rec{key}"] = pc0(m_(s, "main", "Hp", "DRIFT", f"R@0.05|{t}"))
    mac[f"FTvolrec{key}"] = pc0(m_("F21", "main", "Hp", "VolumeReceived", f"R@0.05|{t}"))
# single hop and dust on H (the results that motivated change 3)
for t, key in (("single_hop", "hop"), ("dust_fan", "dust")):
    for s in ("F21", "BR"):
        for m, mk in (("DRIFT", "drift"), ("IsolationForest", "if"), ("DeepAutoencoder", "ae")):
            mac[f"FT{LETTERS[s]}{key}{mk}"] = pc0(m_(s, "main", "H", m, f"R@0.05|{t}"))

# ---------------- external checks ----------------
rows = []
for s in fv.SETS + ["BR-S"]:
    ss, m = ("BR", "DRIFT-S") if s == "BR-S" else (s, "DRIFT")
    e = ext[ss]
    oi, oc, oz = e["ofac_insample"][m], e["ofac_crossfit"][m], e["ofac_crossfit_zero_balance"][m]
    w = e["babd_sample1"]["within"]
    lab = "DRIFT-S on BR" if s == "BR-S" else SET_LABEL[s]
    rows.append(f"{lab} & {pc1(oi['point']['top10'])} & {pc1(oi['top10_same_type'])} & "
                f"{pc1(oi['point']['top10_ctrl'])} & {f2(oc['point']['enrich10_matched'])} "
                f"({f2(oc['ci95_profile'][0])}--{f2(oc['ci95_profile'][1])}) & "
                f"{f2(oz['point']['enrich10_matched'])} ({f2(oz['ci95_profile'][0])}--{f2(oz['ci95_profile'][1])}) & "
                f"{pc0(oi['p2wsh_top10'])}/{pc0(oi['p2wsh_top10_ctrl'])} & "
                f"{f3(w['Legacy'][m]['auc'])} & {f3(w['P2SH'][m]['auc'])} \\\\")
write("tab_s_feat_external.tex", rows)
for s in fv.SETS:
    L, e = LETTERS[s], ext[s]
    oi, oc, oz = e["ofac_insample"]["DRIFT"], e["ofac_crossfit"]["DRIFT"], e["ofac_crossfit_zero_balance"]["DRIFT"]
    mac[f"FT{L}ofacS"], mac[f"FT{L}ofacT"] = pc1(oi["point"]["top10"]), pc1(oi["top10_same_type"])
    mac[f"FT{L}ofacC"] = pc1(oi["point"]["top10_ctrl"])
    mac[f"FT{L}ofacenr"] = f2(oi["point"]["enrich10_matched"])
    mac[f"FT{L}cfenr"] = f2(oc["point"]["enrich10_matched"])
    mac[f"FT{L}cfenrlo"], mac[f"FT{L}cfenrhi"] = f2(oc["ci95_profile"][0]), f2(oc["ci95_profile"][1])
    mac[f"FT{L}zbenr"] = f2(oz["point"]["enrich10_matched"])
    mac[f"FT{L}zbenrlo"], mac[f"FT{L}zbenrhi"] = f2(oz["ci95_profile"][0]), f2(oz["ci95_profile"][1])
    mac[f"FT{L}wshS"], mac[f"FT{L}wshC"] = pc0(oi["p2wsh_top10"]), pc0(oi["p2wsh_top10_ctrl"])
    mac[f"FT{L}babdpkh"] = f3(e["babd_sample1"]["within"]["Legacy"]["DRIFT"]["auc"])
    mac[f"FT{L}babdsh"] = f3(e["babd_sample1"]["within"]["P2SH"]["DRIFT"]["auc"])
    mac[f"FT{L}babd"] = f3(e["babd_sample1"]["overall"]["DRIFT"]["auc"])
    mac[f"FT{L}srvtop"] = pc0(e["services"]["DRIFT"]["top10_services"])
    mac[f"FT{L}sancsrv"] = f3(e["services"]["DRIFT"]["auc_sanctioned_vs_services"])
oiS = ext["BR"]["ofac_insample"]["DRIFT-S"]
mac["FTSofacS"], mac["FTSofacC"] = pc1(oiS["point"]["top10"]), pc1(oiS["point"]["top10_ctrl"])
mac["FTSwshS"], mac["FTSwshC"] = pc0(oiS["p2wsh_top10"]), pc0(oiS["p2wsh_top10_ctrl"])
mac["FTScfenr"] = f2(ext["BR"]["ofac_crossfit"]["DRIFT-S"]["point"]["enrich10_matched"])

# ---------------- controls: noise-matched negatives and validity ----------------
for s in ("F21", "BR", "BR-u"):
    L = LETTERS[s]
    for c, ck in (("ctrl_G1_noise", "ctrlGone"), ("ctrl_consistent_noise", "ctrlcons")):
        for m, mk in (("DeepAutoencoder", "ae"), ("IsolationForest", "if"), ("DRIFT", "drift")):
            mac[f"FT{L}{ck}{mk}"] = f3(R[s]["noise_control"][c][m]["mean"])
rows = []
for kind in ("global", "local", "cluster", "dependency"):
    cells = []
    for s in ("F21", "BR"):
        v = R[s]["validity"][kind]
        cells += [pc0(v["invalid_feature_space"]["mean"]),
                  f3(v["feature_space"]["DeepAutoencoder"]["mean"]), f3(v["repaired"]["DeepAutoencoder"]["mean"]),
                  f3(v["feature_space"]["IsolationForest"]["mean"]), f3(v["repaired"]["IsolationForest"]["mean"])]
    rows.append(f"{kind.capitalize()} & " + " & ".join(cells) + " \\\\")
write("tab_s_feat_validity.tex", rows)
for s in ("F21", "BR"):
    L = LETTERS[s]
    inv = [R[s]["validity"][k]["invalid_feature_space"]["mean"] for k in ("global", "local", "cluster", "dependency")]
    mac[f"FT{L}invlo"], mac[f"FT{L}invhi"] = pc0(min(inv)), pc0(max(inv))
    mac[f"FT{L}invreal"] = pc1(R[s]["validity"]["violations_real"]["any"]["mean"])
    ae_fs = [R[s]["validity"][k]["feature_space"]["DeepAutoencoder"]["mean"] for k in ("global", "local", "cluster", "dependency")]
    ae_rp = [R[s]["validity"][k]["repaired"]["DeepAutoencoder"]["mean"] for k in ("global", "local", "cluster", "dependency")]
    mac[f"FT{L}aefsmin"], mac[f"FT{L}aefsmax"] = f3(min(ae_fs)), f3(max(ae_fs))
    mac[f"FT{L}aerplo"], mac[f"FT{L}aerphi"] = f3(min(ae_rp)), f3(max(ae_rp))

mac["FTallmet"] = "met" if all_met else "did not meet"

# ---------------- further values of the text ----------------
mac["FTsfourp"] = pval(max(S4[m]["p_holm"] for m in s4m))
mac["FTsfourminmargin"] = f3(min(S4[m]["mean_diff"] for m in s4m))
mac["FTsfourminwins"] = str(min(S4[m]["wins"] for m in s4m))
for s in ("F21", "BR"):
    for m, key in (("IsolationForest", "if"), ("DeepAutoencoder", "ae"), ("ECOD", "ecod"), ("COPOD", "copod"),
                   ("DeepSVDD", "svdd")):
        mac[f"FT{LETTERS[s]}srv{key}"] = f3(ext[s]["services"][m]["auc_injected_vs_services"])
for s in fv.SETS:
    L = LETTERS[s]
    for m, key in (("IsolationForest", "if"), ("DeepAutoencoder", "ae")):
        mac[f"FT{L}{key}matGthree"] = f3(mat(s, "G3", m))
        mac[f"FT{L}{key}matH"] = f3(mat(s, "H", m))
for s in ("W21", "B", "BR-u"):
    mac[f"FT{LETTERS[s]}hopdrift"] = pc0(m_(s, "main", "H", "DRIFT", "R@0.05|single_hop"))
    mac[f"FT{LETTERS[s]}hopae"] = pc0(m_(s, "main", "H", "DeepAutoencoder", "R@0.05|single_hop"))
for s, key in (("F21", "Ftw"), ("BR", "BR")):
    mac[f"FT{key}structdrift"] = pc0(m_(s, "main", "H", "DRIFT", "R@0.05|structuring"))
first = OUT / "external_parts_first_crossfit" / "features_external_first.json"
if first.exists():
    X1 = json.load(open(first))["sets"]
    ch = [abs(X1[s][k]["DRIFT"]["point"]["enrich10_matched"] - ext[s][k]["DRIFT"]["point"]["enrich10_matched"])
          for s in fv.SETS if s != "F21" for k in ("ofac_crossfit", "ofac_crossfit_zero_balance")]
    mac["FTcfrerunmax"] = f2(max(ch))
    mac["FTcffirst"] = f2(X1["F21"]["ofac_crossfit"]["DRIFT"]["point"]["enrich10_matched"])

# ---------------- qualitative statements of the supplementary section ----------------
D = lambda s, g: pr(s, g, "DRIFT")
gain = lambda m: pr("BR", "G2", m) - pr("F21", "G2", m)
mac["FTngainmore"] = {5: "five", 6: "six", 7: "seven", 8: "eight", 9: "all"}.get(
    sum(gain(m) > gain("DRIFT") for m in s4m), str(sum(gain(m) > gain("DRIFT") for m in s4m)))
claims = {
    "S1 failed, and only on H": (not S1["holds"]) and S1["G3"]["holds"] and S1["Hp"]["holds"] and not S1["H"]["holds"],
    "on H the margin was below 0.05 while the adjusted P was below 0.05": S1["H"]["mean_diff"] < 0.05 and S1["H"]["p_holm"] < 0.05,
    "S2 failed": not S2["holds"],
    "S3 held": S3["holds"],
    "S4 held": S4["holds"],
    "the revision as a whole did not meet its criteria": not all_met,
    "the strongest detector on H' was the autoencoder": best == "DeepAutoencoder",
    "under BR the autoencoder lost and the Isolation Forest gained against the services":
        ext["BR"]["services"]["DeepAutoencoder"]["auc_injected_vs_services"] < ext["F21"]["services"]["DeepAutoencoder"]["auc_injected_vs_services"] - 0.2
        and ext["BR"]["services"]["IsolationForest"]["auc_injected_vs_services"] > ext["F21"]["services"]["IsolationForest"]["auc_injected_vs_services"],
    "ECOD, COPOD and Deep SVDD on BR exceeded 0.94 against the services":
        all(ext["BR"]["services"][m]["auc_injected_vs_services"] > 0.94 for m in ("ECOD", "COPOD", "DeepSVDD")),
    "the PR-AUC of DRIFT rose on every benchmark under BR": all(D("BR", g) > D("F21", g) for g in ("G2", "G3", "H", "Hp")),
    "on G2 at least five of the nine detectors gained more than DRIFT": sum(gain(m) > gain("DRIFT") for m in s4m) >= 5,
    "on G2 under BR DRIFT stayed below its Isolation Forest": D("BR", "G2") < pr("BR", "G2", "IsolationForest"),
    "S1 adjusted P was below 0.01 on G3 and H'": S1["G3"]["p_holm"] < 0.01 and S1["Hp"]["p_holm"] < 0.01,
    "every learned detector except LOF and the autoencoder gained PR-AUC on G2":
        all(pr("BR", "G2", m) > pr("F21", "G2", m) for m in ("OneClassSVM", "PCA", "ECOD", "COPOD", "DIF", "DeepSVDD", "IsolationForest"))
        and pr("BR", "G2", "DeepAutoencoder") < pr("F21", "G2", "DeepAutoencoder"),
    "on G2 under BR the Isolation Forest and ECOD exceeded DRIFT":
        pr("BR", "G2", "IsolationForest") > D("BR", "G2") and pr("BR", "G2", "ECOD") > D("BR", "G2"),
    "on H' the five-detector mean exceeded DRIFT": pr("BR", "Hp", "All5MeanRank") > D("BR", "Hp"),
    "on H' received value was within 0.005 of DRIFT": abs(pr("F21", "Hp", "VolumeReceived") - D("BR", "Hp")) < 0.005,
    "on the hard negatives the Isolation Forest and ECOD on BR exceeded DRIFT":
        hn("BR", "G2", "IsolationForest") > hn("BR", "G2", "DRIFT") and hn("BR", "G2", "ECOD") > hn("BR", "G2", "DRIFT"),
    "B exceeded BR in matched AUC on G3, H and H'": all(mat("B", g, "DRIFT") > mat("BR", g, "DRIFT") for g in ("G3", "H", "Hp")),
    "B exceeded BR in PR-AUC on H and H'": D("B", "H") > D("BR", "H") and D("B", "Hp") > D("BR", "Hp"),
    "the single hop stayed unfound by DRIFT under BR": m_("BR", "main", "H", "DRIFT", "R@0.05|single_hop") <= 0.01,
    "the autoencoder lost most of its single-hop recall under BR":
        m_("BR", "main", "H", "DeepAutoencoder", "R@0.05|single_hop") < 0.1 < m_("F21", "main", "H", "DeepAutoencoder", "R@0.05|single_hop"),
    "W21 alone raised the AUC against wallets with >= 1,000 transactions": hn("W21", "G2", "DRIFT", "ntx_ge_1000") > hn("F21", "G2", "DRIFT", "ntx_ge_1000") + 0.05,
    "the G1 noise control of the autoencoder fell under BR": R["BR"]["noise_control"]["ctrl_G1_noise"]["DeepAutoencoder"]["mean"] < R["F21"]["noise_control"]["ctrl_G1_noise"]["DeepAutoencoder"]["mean"] - 0.1,
    "the identity-preserving control stayed near 0.5 under BR": abs(R["BR"]["noise_control"]["ctrl_consistent_noise"]["DRIFT"]["mean"] - 0.5) < 0.02,
    "feature-space samples stayed mostly invalid under BR": min(R["BR"]["validity"][k]["invalid_feature_space"]["mean"] for k in ("global", "local", "cluster", "dependency")) > 0.8,
    "the autoencoder still separated feature-space samples better than repaired ones under BR":
        all(R["BR"]["validity"][k]["feature_space"]["DeepAutoencoder"]["mean"] > R["BR"]["validity"][k]["repaired"]["DeepAutoencoder"]["mean"] for k in ("global", "local", "cluster", "dependency")),
    "the P2WSH shares fell below half": ext["BR"]["ofac_insample"]["DRIFT"]["p2wsh_top10"] < 0.5 and ext["BR"]["ofac_insample"]["DRIFT"]["p2wsh_top10_ctrl"] < 0.5,
    "the same-type share moved toward 10%": ext["BR"]["ofac_insample"]["DRIFT"]["top10_same_type"] < ext["F21"]["ofac_insample"]["DRIFT"]["top10_same_type"],
    "balance-matched, cross-fitted: profile interval above one under BR": ext["BR"]["ofac_crossfit_zero_balance"]["DRIFT"]["ci95_profile"][0] > 1.0,
    "balance-matched, cross-fitted: profile interval includes one under the 21 features": ext["F21"]["ofac_crossfit_zero_balance"]["DRIFT"]["ci95_profile"][0] <= 1.0,
    "standard matching, cross-fitted: profile interval includes one under BR": ext["BR"]["ofac_crossfit"]["DRIFT"]["ci95_profile"][0] <= 1.0 <= ext["BR"]["ofac_crossfit"]["DRIFT"]["ci95_profile"][1],
    "within P2SH the order stayed inverted": ext["BR"]["babd_sample1"]["within"]["P2SH"]["DRIFT"]["auc"] < 0.5,
    "the calibration of the rank threshold was kept": abs(R["BR"]["exceedance_p95"]["DRIFT"]["mean"] - 0.05) < 0.005,
    "the 21-feature column reproduced the article": not bad,
}
failed = [k for k, v in claims.items() if not v]
(OUT / "features_claims_check.json").write_text(json.dumps({k: bool(v) for k, v in claims.items()}, indent=1))
print(f"claims checked: {len(claims)}, not supported: {len(failed)}")
for k in failed:
    print("  NOT SUPPORTED:", k)

(TAB / "numbers_features.tex").write_text(
    "% generated by src/make_tables_features.py from outputs/features; do not edit\n"
    + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(mac.items())) + "\n", encoding="utf-8")
print(f"wrote numbers_features.tex ({len(mac)} macros) and tab_s_feat_*.tex")
print("criteria:", crit)
sys.exit(1 if failed else 0)
