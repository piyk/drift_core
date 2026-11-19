"""
make_tables_text.py  |  Every run-dependent number of the running text, as a macro
==================================================================================
Writes manuscript/tables/numbers_text.tex (macros \\TX...) from the stored results, so
that the running text of the manuscript, the supplement and the response letter types
no result by hand. It also checks the qualitative statements that the text makes about
these results (for example "in every seed" or "did not differ") and prints every
statement that the stored results do not support.

Run:  python revise1/src/make_tables_text.py [outputs_dir] [--print]
"""
import json
import sys
from pathlib import Path

REV = Path(__file__).resolve().parents[1]
OUT = Path(next((a for a in sys.argv[1:] if not a.startswith("--")), REV / "outputs"))
TAB = REV / "manuscript" / "tables"
WORDS = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
         8: "eight", 9: "nine", 10: "ten"}

load = lambda n: json.load(open(OUT / n))
e, tr, p, sep = load("experiments.json"), load("transductive.json"), load("paired_tests.json"), load("separability.json")
x = load("extensions.json")
mean = lambda d: d["mean"]
f3 = lambda v: f"{v:.3f}"
f2 = lambda v: f"{v:.2f}"
pct0 = lambda v: f"{100 * v:.0f}"
pct1 = lambda v: f"{100 * v:.1f}"
pval = lambda v: f"{v:.3f}" if v < 0.1 else f"{v:.2f}"
mac, claims = {}, []


def claim(text, ok):
    claims.append((text, bool(ok)))


def E(block, det, metric):
    return mean(e[block][det][metric])


def P(block, a, b, metric="pr_auc"):
    return p[block][f"{a}_vs_{b}"][metric]


# ---------------- main comparison (G2, held-out) ----------------
for key, det in (("drift", "DRIFT"), ("if", "IsolationForest"), ("ae", "DeepAutoencoder"),
                 ("heur", "Heuristic"), ("min", "MinRank")):
    mac[f"TX{key}rocGtwo"] = f3(E("E1_main", det, "roc_auc"))
    mac[f"TX{key}prGtwo"] = f3(E("E1_main", det, "pr_auc"))
    mac[f"TX{key}prGone"] = f3(E("E2_G1", det, "pr_auc"))
    mac[f"TX{key}rocGthree"] = f3(E("E2_G3", det, "roc_auc"))
    mac[f"TX{key}prGthree"] = f3(E("E2_G3", det, "pr_auc"))
    mac[f"TX{key}rocH"] = f3(E("E2_H", det, "roc_auc"))
    mac[f"TX{key}prH"] = f3(E("E2_H", det, "pr_auc"))
mac["TXdriftrocsdGtwo"] = f3(e["E1_main"]["DRIFT"]["roc_auc"]["std"])
mac["TXdriftprsdGtwo"] = f3(e["E1_main"]["DRIFT"]["pr_auc"]["std"])
mac["TXcompmaxprH"] = f3(max(E("E2_H", "IsolationForest", "pr_auc"), E("E2_H", "DeepAutoencoder", "pr_auc")))
mac["TXallfiverocGtwo"] = f3(E("E1_main", "All5MeanRank", "roc_auc"))
# G2L: the G2 draws with the span over the whole activity (audit step between G1 and G2);
# the effect of the artifacts is G1 against G2L, which share the span convention
for key, det in (("drift", "DRIFT"), ("if", "IsolationForest"), ("ae", "DeepAutoencoder"), ("heur", "Heuristic")):
    mac[f"TX{key}prGtwoL"] = f3(E("E2_G2L", det, "pr_auc"))
    mac[f"TX{key}rocGtwoL"] = f3(E("E2_G2L", det, "roc_auc"))
drop = 1 - E("E2_G2L", "DRIFT", "pr_auc") / E("E2_G1", "DRIFT", "pr_auc")
mac["TXdropGonepct"] = pct0(drop)
mac["TXinflGonepct"] = pct0(E("E2_G1", "DRIFT", "pr_auc") / E("E2_G2L", "DRIFT", "pr_auc") - 1)
mac["TXwindowgain"] = f3(E("E1_main", "DRIFT", "pr_auc") - E("E2_G2L", "DRIFT", "pr_auc"))

d_if, d_ae = P("E1_main", "DRIFT", "IsolationForest"), P("E1_main", "DRIFT", "DeepAutoencoder")
d_ae_roc = P("E1_main", "DRIFT", "DeepAutoencoder", "roc_auc")
mac["TXdiffifGtwo"] = f3(abs(d_if["mean_diff"]))
mac["TXpifGtwo"] = pval(d_if["wilcoxon_p"])
mac["TXdiffaerocGtwo"] = f3(d_ae_roc["mean_diff"])
mac["TXdiffaeGtwo"] = f3(d_ae["mean_diff"])
mac["TXpaeGtwo"] = pval(d_ae["wilcoxon_p"])
claim("G2: DRIFT and the Isolation Forest did not differ (P >= 0.05)", d_if["wilcoxon_p"] >= 0.05)
claim("G2: DRIFT exceeded the autoencoder in every seed", d_ae["wins"] == d_ae["n"] and d_ae_roc["mean_diff"] > 0)
for bench, tag in (("E2_G3", "Gthree"), ("E2_H", "H")):
    a, b = P(bench, "DRIFT", "IsolationForest"), P(bench, "DRIFT", "DeepAutoencoder")
    mac[f"TXdiffif{tag}"] = f3(a["mean_diff"])
    mac[f"TXpif{tag}"] = pval(a["wilcoxon_p"])
    mac[f"TXdiffae{tag}"] = f3(b["mean_diff"])
    mac[f"TXpae{tag}"] = pval(b["wilcoxon_p"])
    mac[f"TXpmax{tag}"] = pval(max(a["wilcoxon_p"], b["wilcoxon_p"]))
    mac[f"TXifwins{tag}"] = WORDS[a["wins"]]
# "Exceeded" is claimed only where the difference stays significant after the Holm correction
# that Section V-C specifies; an uncorrected P below 0.05 is reported as such, not as "exceeded".
holm = p["holm_pr_auc_family"]
hp = lambda bench, other: holm[f"{bench}|DRIFT_vs_{other}"]["p_holm"]
mac["TXpifGthreeholm"] = pval(hp("E2_G3", "IsolationForest"))
both_holm = {b: all(P(b, "DRIFT", c)["mean_diff"] > 0 and hp(b, c) < 0.05
                    for c in ("IsolationForest", "DeepAutoencoder")) for b in ("E2_G3", "E2_H")}
claim("H: DRIFT exceeded both components, significant after Holm correction", both_holm["E2_H"])
claim("G3: DRIFT exceeded the autoencoder after Holm correction; its margin over the Isolation Forest had "
      "P < 0.05 uncorrected but was not significant after correction (as the text states)",
      P("E2_G3", "DRIFT", "DeepAutoencoder")["mean_diff"] > 0 and hp("E2_G3", "DeepAutoencoder") < 0.05
      and P("E2_G3", "DRIFT", "IsolationForest")["mean_diff"] > 0
      and P("E2_G3", "DRIFT", "IsolationForest")["wilcoxon_p"] < 0.05 and hp("E2_G3", "IsolationForest") >= 0.05)
holm_sum = p["holm_summary"]
holm_lost = [k for k, v in holm.items() if v["p"] < 0.05 and v["p_holm"] >= 0.05]
claim("Holm family: every raw-significant comparison except DRIFT against the Isolation Forest on G3 "
      "remained significant", holm_lost == ["E2_G3|DRIFT_vs_IsolationForest"]
      and holm_sum["n_raw_below_0.05"] - holm_sum["n_holm_below_0.05"] == 1)
stronger = lambda bench: "IF" if E(bench, "IsolationForest", "pr_auc") > E(bench, "DeepAutoencoder", "pr_auc") else "AE"
claim("the stronger component was the Isolation Forest on G2 and G3 and the autoencoder on H",
      (stronger("E1_main"), stronger("E2_G3"), stronger("E2_H")) == ("IF", "IF", "AE"))
names_main = ["Heuristic", "OneClassSVM", "LOF", "PCA", "IsolationForest", "DeepAutoencoder", "DRIFT"]
h_roc = {m: E("E2_H", m, "roc_auc") for m in names_main}
h_pr = {m: E("E2_H", m, "pr_auc") for m in names_main}
vol_h = x["E11_volume"]["H"]["VolumeReceived"]
pyod_h = {k: v for k, v in x["E13_pyod"]["H"].items() if isinstance(v, dict)}
all_h_roc = dict(h_roc, Volume=mean(vol_h["roc_auc"]), **{k: mean(v["roc_auc"]) for k, v in pyod_h.items()})
all_h_pr = dict(h_pr, Volume=mean(vol_h["pr_auc"]), **{k: mean(v["pr_auc"]) for k, v in pyod_h.items()})
claim("H: DRIFT had the highest PR-AUC of all scores in Table IV (every individual detector and baseline)",
      max(all_h_pr, key=all_h_pr.get) == "DRIFT")
claim("H: two aggregation rules of Table V (minimum of ranks, five-detector mean) had a higher PR-AUC than "
      "DRIFT, so 'highest' is claimed only against the detectors and baselines of Table IV",
      E("E2_H", "MinRank", "pr_auc") > E("E2_H", "DRIFT", "pr_auc")
      and E("E2_H", "All5MeanRank", "pr_auc") > E("E2_H", "DRIFT", "pr_auc"))
claim("H: only ECOD had a higher ROC-AUC than DRIFT",
      {k for k, v in all_h_roc.items() if v > all_h_roc["DRIFT"]} <= {"ECOD"})
g2_pr = dict({d: E("E1_main", d, "pr_auc") for d in names_main}, Volume=mean(x["E11_volume"]["G2"]["VolumeReceived"]["pr_auc"]),
             **{k: mean(v["pr_auc"]) for k, v in x["E13_pyod"]["G2"].items() if isinstance(v, dict)})
g3_pr = dict({d: E("E2_G3", d, "pr_auc") for d in names_main}, Volume=mean(x["E11_volume"]["G3"]["VolumeReceived"]["pr_auc"]),
             **{k: mean(v["pr_auc"]) for k, v in x["E13_pyod"]["G3"].items() if isinstance(v, dict)})
claim("G2: ECOD and COPOD gave the highest PR-AUC, volume matching COPOD",
      sorted(g2_pr, key=g2_pr.get, reverse=True)[0] == "ECOD"
      and round(g2_pr["Volume"], 3) == round(g2_pr["COPOD"], 3))
claim("G3: received value alone gave the highest PR-AUC", max(g3_pr, key=g3_pr.get) == "Volume")
claim("conclusion: ECOD and received value exceeded DRIFT on G2 and G3 (PR-AUC)",
      all(g[k] > g["DRIFT"] for g in (g2_pr, g3_pr) for k in ("ECOD", "Volume")))

# ---------------- precision at the 1% budget (Section VI-B, practical-role paragraph) ----------------
P1 = lambda block, det: E(block, det, "P@0.01")
def single_p1(block, bench):
    """P@1% of every single detector and baseline of Table IV (no aggregation rule except DRIFT)."""
    d = {m: P1(block, m) for m in names_main}
    d["Volume"] = mean(x["E11_volume"][bench]["VolumeReceived"]["P@0.01"])
    d.update({k: mean(v["P@0.01"]) for k, v in x["E13_pyod"][bench].items() if isinstance(v, dict)})
    return d
p1 = {"G2": single_p1("E1_main", "G2"), "G3": single_p1("E2_G3", "G3"), "H": single_p1("E2_H", "H")}
rank1 = {b: sorted(v, key=v.get, reverse=True) for b, v in p1.items()}
mac["TXponeH"], mac["TXponeHif"] = f3(p1["H"]["DRIFT"]), f3(p1["H"]["IsolationForest"])
mac["TXponeGtwo"], mac["TXponeGtwodif"] = f3(p1["G2"]["DRIFT"]), f3(p1["G2"]["DIF"])
mac["TXponeGtwoheur"] = f3(p1["G2"]["Heuristic"])
mac["TXponeGthree"], mac["TXponeGthreeheur"] = f3(p1["G3"]["DRIFT"]), f3(p1["G3"]["Heuristic"])
claim("P@1% on H: DRIFT highest of every single detector and baseline, the Isolation Forest next",
      rank1["H"][:2] == ["DRIFT", "IsolationForest"])
claim("P@1% on G2: the Deep Isolation Forest and the heuristic were the two most precise single scores",
      rank1["G2"][:2] == ["DIF", "Heuristic"])
claim("P@1% on G3: the heuristic was the most precise single score", rank1["G3"][0] == "Heuristic")
claim("P@1% favored DRIFT only on H", rank1["G2"][0] != "DRIFT" and rank1["G3"][0] != "DRIFT")
ratio1 = [p1[b]["Heuristic"] / p1[b]["DRIFT"] for b in ("G2", "G3")]
claim("at a 1% budget the heuristic was two to four times as precise as DRIFT on G2 and G3",
      all(2 <= r_ <= 4 for r_ in ratio1))

# ---------------- generic anomaly types of Table III: the autoencoder against DRIFT ----------------
from scipy.stats import wilcoxon as _wilcoxon
rep_seeds = load("adbench_repair_per_seed.json")
adb_tests = {}
for t_ in ("local", "cluster", "dependency"):
    for space in ("feature_space", "repaired"):
        dd = [s_[t_][space]["DeepAutoencoder"] - s_[t_][space]["DRIFT"] for s_ in rep_seeds]
        adb_tests[(t_, space)] = (sum(v_ > 0 for v_ in dd), len(dd), float(_wilcoxon(dd).pvalue))
mac["TXadbaep"] = pval(max(v_[2] for v_ in adb_tests.values()))
adb_ae_wins = all(w_ == n_ for w_, n_, _ in adb_tests.values())
def _adb_sig_above(t_, sp, c):
    dd = [s_[t_][sp][c] - s_[t_][sp]["DRIFT"] for s_ in rep_seeds]
    if all(abs(v_) < 1e-15 for v_ in dd):
        return False
    return sum(dd) > 0 and float(_wilcoxon(dd).pvalue) < 0.05
# a component significantly above DRIFT on any generic type of Table III contradicts an unscoped never-below claim
adb_contradicts = any(_adb_sig_above(t_, sp, c) for t_ in ("global", "local", "cluster", "dependency")
                      for sp in ("feature_space", "repaired") for c in ("DeepAutoencoder", "IsolationForest"))
claim("Table III global type: DRIFT was not significantly below the autoencoder after repair (so the exception is three of the four types)",
      not _adb_sig_above("global", "repaired", "DeepAutoencoder"))
claim("Table III types local, clustered, dependency: the autoencoder exceeded DRIFT in every seed, "
      "before and after repair, P < 0.05 each",
      adb_ae_wins and all(v_[2] < 0.05 for v_ in adb_tests.values()))

# ---------------- generator audit ----------------
mac["TXtrGonedriftroc"] = f3(mean(tr["G1"]["DRIFT"]["roc_auc"]))
mac["TXtrGonedriftrocsd"] = f3(tr["G1"]["DRIFT"]["roc_auc"]["std"])
mac["TXtrGonedriftpr"] = f3(mean(tr["G1"]["DRIFT"]["pr_auc"]))
mac["TXtrGonedriftprsd"] = f3(tr["G1"]["DRIFT"]["pr_auc"]["std"])
mac["TXtrGtwoLdriftpr"] = f3(mean(tr["G2L"]["DRIFT"]["pr_auc"]))
nz = e["E4_noise"]
mac["TXctrlGoneae"] = f3(mean(nz["default"]["ctrl_G1_noise"]["DeepAutoencoder"]["auc_perturbed_vs_original"]))
mac["TXctrlGoneshare"] = pct0(mean(nz["default"]["ctrl_G1_noise"]["DeepAutoencoder"]["share_above_real_p95"]))
mac["TXctrlconsae"] = f3(mean(nz["default"]["ctrl_consistent_noise"]["DeepAutoencoder"]["auc_perturbed_vs_original"]))
widths = ["tight", "default", "wide", "very_wide"]
ctrl_ae = [mean(nz[w]["ctrl_G1_noise"]["DeepAutoencoder"]["auc_perturbed_vs_original"]) for w in widths]
mac["TXctrlGonelo"], mac["TXctrlGonehi"] = f3(ctrl_ae[0]), f3(ctrl_ae[-1])
claim("the G1 control AUC of the autoencoder rose with the noise width", all(a < b for a, b in zip(ctrl_ae, ctrl_ae[1:])))
g1_roc = [mean(nz[w]["G1"]["DRIFT"]["roc_auc"]) for w in widths]
mac["TXnoiseGonelo"], mac["TXnoiseGonehi"] = f3(g1_roc[0]), f3(g1_roc[-1])
claim("the ROC-AUC of DRIFT on G1 rose with the noise width", g1_roc[-1] > g1_roc[0])
g2_roc = [mean(nz[w]["G2"]["DRIFT"]["roc_auc"]) for w in widths]
mac["TXnoiseGtwolo"], mac["TXnoiseGtwohi"] = f3(min(g2_roc)), f3(max(g2_roc))
claim("the heuristic's PR-AUC rose slightly from G1 to G2L (held-out)",
      E("E2_G2L", "Heuristic", "pr_auc") > E("E2_G1", "Heuristic", "pr_auc"))
claim("every learned detector fell from G1 to G2L (held-out)",
      all(E("E2_G2L", d, "pr_auc") < E("E2_G1", d, "pr_auc") for d in ("IsolationForest", "DeepAutoencoder", "DRIFT")))
claim("the observation window raised DRIFT, the Isolation Forest, and the autoencoder from G2L to G2",
      all(E("E1_main", d, "pr_auc") > E("E2_G2L", d, "pr_auc") for d in ("IsolationForest", "DeepAutoencoder", "DRIFT")))
mac["TXrfroc"], mac["TXrfpr"] = f3(mean(sep["G2"]["rf_roc_auc"])), f3(mean(sep["G2"]["rf_pr_auc"]))
mac["TXrfrocGone"], mac["TXrfprGone"] = f"{mean(sep['G1']['rf_roc_auc']):.4f}", f3(mean(sep["G1"]["rf_pr_auc"]))
mac["TXsinglefeat"] = f3(mean(sep["G2"]["single_feature_roc_auc"]))

# ---------------- aggregation rules ----------------
gate = P("E1_main", "DRIFT", "Gated_tau0.35")
mac["TXgatediff"] = f3(gate["mean_diff"])
mac["TXgatep"] = pval(gate["wilcoxon_p"])
claim("the gate at tau 0.35 lowered PR-AUC on G2 significantly", gate["mean_diff"] > 0 and gate["wilcoxon_p"] < 0.05)
for bench, tag in (("E1_main", "Gtwo"), ("E2_G3", "Gthree"), ("E2_H", "H")):
    m_ = P(bench, "DRIFT", "MinRank")
    mac[f"TXmin{tag}"] = f3(-m_["mean_diff"])
    mac[f"TXminp{tag}"] = pval(m_["wilcoxon_p"])
    mac[f"TXminwins{tag}"] = WORDS[m_["n"] - m_["wins"]]
claim("minimum of ranks better than the mean on G2 and G3 (P < 0.05) and not different on H (P >= 0.05)",
      -P("E1_main", "DRIFT", "MinRank")["mean_diff"] > 0 and P("E1_main", "DRIFT", "MinRank")["wilcoxon_p"] < 0.05
      and -P("E2_G3", "DRIFT", "MinRank")["mean_diff"] > 0 and P("E2_G3", "DRIFT", "MinRank")["wilcoxon_p"] < 0.05
      and P("E2_H", "DRIFT", "MinRank")["wilcoxon_p"] >= 0.05)
vs = p["validation_selection"]
mac["TXvalgatedn"] = WORDS[vs["gated_family"].get("DRIFT", 0)]
mac["TXvalminn"] = WORDS[vs["all_rules"]["pr_auc"].get("MinRank", 0)]
claim("within the gated family the validation split selected the uniform limit in all ten seeds",
      vs["gated_family"].get("DRIFT", 0) == 10)
for bench in ("E1_main", "E2_G3", "E2_H"):
    claim(f"{bench}: the maximum of ranks was worse than the mean",
          E(bench, "MaxRank", "pr_auc") < E(bench, "DRIFT", "pr_auc"))
claim("rules on standardized raw scores were worse than rank rules on G3 and H",
      all(E(b, z, "pr_auc") < E(b, "DRIFT", "pr_auc") for b in ("E2_G3", "E2_H") for z in ("ZMean", "ZMax")))
rules = ["DRIFT", "MaxRank", "MinRank", "GeoMeanRank", "WAvg_AE0.75", "WAvg_AE0.25", "ZMean", "ZMax", "All5MeanRank",
         "Gated_tau0.5", "Gated_tau0.35", "Gated_tau0.2", "Gated_tau0.1", "Cascade"]
benches = ("E1_main", "E2_G3", "E2_H")
claim("the five-detector mean rank gave the highest PR-AUC among the rules on all three benchmarks",
      all(max(rules, key=lambda r: E(b, r, "pr_auc")) == "All5MeanRank" for b in benches))
five = [P(b, "DRIFT", "All5MeanRank") for b in benches]
mac["TXallfivepmax"] = pval(max(t["wilcoxon_p"] for t in five))
claim("the five-detector mean rank exceeded DRIFT in PR-AUC on all three benchmarks (P < 0.05 each)",
      all(t["mean_diff"] < 0 and t["wilcoxon_p"] < 0.05 for t in five))
mac["TXvalallfiven"] = WORDS[vs["all_rules"]["pr_auc"].get("All5MeanRank", 0)]
mac["TXvalwavgn"] = WORDS[vs["all_rules"]["pr_auc"].get("WAvg_AE0.25", 0)]
claim("the validation split selected only the five-detector mean, the minimum of ranks, and the weighted mean "
      "with autoencoder weight 0.25, in ten seeds together",
      set(vs["all_rules"]["pr_auc"]) == {"All5MeanRank", "MinRank", "WAvg_AE0.25"}
      and sum(vs["all_rules"]["pr_auc"].values()) == 10)
mac["TXholmadj"] = WORDS[holm_sum["n_holm_below_0.05"]]
claim("the Holm family is DRIFT against the two components, the minimum of ranks, and the gated rule at tau 0.35 "
      "on G2, G3, and H",
      set(holm) == {f"{b}|DRIFT_vs_{c}" for b in ("E1_main", "E2_G3", "E2_H")
                    for c in ("IsolationForest", "DeepAutoencoder", "MinRank", "Gated_tau0.35")})
vsel = p["validation_selected_rule"]
claim("the rule selected on validation labels exceeded DRIFT in PR-AUC on all three benchmarks (P < 0.05 each)",
      all(vsel[b]["pr_auc"]["mean_diff_vs_mean_rank"] > 0 and vsel[b]["pr_auc"]["wilcoxon_p"] < 0.05 for b in benches))
claim("performance decreased as tau decreased on all three benchmarks",
      all(E(b, "Gated_tau0.5", "pr_auc") > E(b, "Gated_tau0.35", "pr_auc") > E(b, "Gated_tau0.2", "pr_auc")
          > E(b, "Gated_tau0.1", "pr_auc") for b in ("E1_main", "E2_G3", "E2_H")))
rank_rules = ["DRIFT", "MaxRank", "MinRank", "GeoMeanRank", "WAvg_AE0.75", "WAvg_AE0.25",
              "Gated_tau0.5", "Gated_tau0.35", "Gated_tau0.2", "Gated_tau0.1"]
max_weakest = all(min(rank_rules, key=lambda r_: E(b, r_, "pr_auc")) == "MaxRank" for b in benches)
claim("the maximum of ranks was the weakest rank rule on all three benchmarks (PR-AUC)", max_weakest)
agreement = max_weakest and all(-P(b, "DRIFT", "MinRank")["mean_diff"] > 0 for b in ("E1_main", "E2_G3"))
claim("agreement, not coverage: the minimum of ranks beat the mean on G2 and G3 and the maximum was the weakest rank rule",
      agreement)
claim("the cascade was the weakest hybrid",
      all(E(b, "Cascade", "pr_auc") < min(E(b, r, "pr_auc") for r in rules if r != "Cascade")
          for b in ("E1_main", "E2_G3", "E2_H")))

# ---------------- typology coverage ----------------
R = lambda block, det, typ: E(block, det, f"R@0.05|{typ}")
mac["TXdormdrift"], mac["TXdormif"] = pct0(R("E1_main", "DRIFT", "dormant_burst")), pct0(R("E1_main", "IsolationForest", "dormant_burst"))
mac["TXcashdrift"], mac["TXcashif"] = pct0(R("E1_main", "DRIFT", "rapid_cashout")), pct0(R("E1_main", "IsolationForest", "rapid_cashout"))
mac["TXmixdrift"], mac["TXmixif"] = pct0(R("E1_main", "DRIFT", "mixer_passthrough")), pct0(R("E1_main", "IsolationForest", "mixer_passthrough"))
mac["TXmixae"] = pct0(R("E1_main", "DeepAutoencoder", "mixer_passthrough"))
claim("DRIFT recalled more dormant bursts and cash-outs but fewer mixers than the Isolation Forest",
      R("E1_main", "DRIFT", "dormant_burst") > R("E1_main", "IsolationForest", "dormant_burst")
      and R("E1_main", "DRIFT", "rapid_cashout") > R("E1_main", "IsolationForest", "rapid_cashout")
      and R("E1_main", "DRIFT", "mixer_passthrough") < R("E1_main", "IsolationForest", "mixer_passthrough"))
mac["TXhopae"] = pct0(R("E2_H", "DeepAutoencoder", "single_hop"))
mac["TXhopif"] = pct1(R("E2_H", "IsolationForest", "single_hop"))
mac["TXhopdrift"] = pct1(R("E2_H", "DRIFT", "single_hop"))
mac["TXhopmax"] = pct0(R("E2_H", "MaxRank", "single_hop"))
mac["TXhoarddrift"] = pct0(R("E2_H", "DRIFT", "hoard"))
mac["TXdustdrift"] = pct0(R("E2_H", "DRIFT", "dust_fan"))
mac["TXstructdrift"] = pct0(R("E2_H", "DRIFT", "structuring"))
claim("DRIFT recalled every hoard wallet", R("E2_H", "DRIFT", "hoard") == 1.0)
claim("the heuristic recalled every mixer and no dormant burst",
      R("E1_main", "Heuristic", "mixer_passthrough") == 1.0 and R("E1_main", "Heuristic", "dormant_burst") == 0.0)
claim("on H the heuristic recalled every dust wallet but no hoard and no single hop",
      R("E2_H", "Heuristic", "dust_fan") == 1.0 and R("E2_H", "Heuristic", "hoard") == 0.0
      and R("E2_H", "Heuristic", "single_hop") == 0.0)
mac["TXtrpeelae"] = pct0(mean(tr["G1"]["DeepAutoencoder"]["R@0.05|peeling_chain"]))
mac["TXtrpeeldrift"] = pct0(mean(tr["G1"]["DRIFT"]["R@0.05|peeling_chain"]))
mac["TXaepeelmax"] = pct0(max(R("E2_G2L", "DeepAutoencoder", "peeling_chain"),
                              mean(tr["G2L"]["DeepAutoencoder"]["R@0.05|peeling_chain"])))
ra = e["E1_rank_analysis"]["peeling_chain"]
mac["TXpeelrankae"], mac["TXpeelrankif"] = pct0(mean(ra["median_r_ae"])), pct0(mean(ra["median_r_if"]))
vol_g2 = x["E11_volume"]["G2"]["VolumeReceived"]
for key, det in (("drift", "DRIFT"), ("if", "IsolationForest"), ("ae", "DeepAutoencoder"), ("heur", "Heuristic")):
    mac[f"TXpeel{key}"] = pct1(R("E1_main", det, "peeling_chain"))
    mac[f"TXpeel{key}L"] = pct1(R("E2_G2L", det, "peeling_chain"))
mac["TXpeelvol"] = pct1(mean(vol_g2["R@0.05|peeling_chain"]))
learned_peel = [R("E1_main", d, "peeling_chain") for d in ("IsolationForest", "DeepAutoencoder", "DRIFT")]
mac["TXpeellearnedlo"], mac["TXpeellearnedhi"] = pct1(min(learned_peel)), pct1(max(learned_peel))
TYPS = ["peeling_chain", "mixer_passthrough", "rapid_cashout", "dormant_burst"]
claim("on G2L peeling chains were the hardest typology for IF, AE, and DRIFT",
      all(min(TYPS, key=lambda t: R("E2_G2L", d, t)) == "peeling_chain"
          for d in ("IsolationForest", "DeepAutoencoder", "DRIFT")))
claim("on G2 peeling chains were still the hardest typology for DRIFT",
      min(TYPS, key=lambda t: R("E1_main", "DRIFT", t)) == "peeling_chain")
claim("on G2 DRIFT recalled at least as many peeling chains as the autoencoder (fusion did not suppress it)",
      R("E1_main", "DRIFT", "peeling_chain") >= R("E1_main", "DeepAutoencoder", "peeling_chain"))
claim("on G2 DRIFT exceeded the autoencoder at each reported budget (P@1%, R@5%, R@10%)",
      all(E("E1_main", "DRIFT", k) > E("E1_main", "DeepAutoencoder", k) for k in ("P@0.01", "R@0.05", "R@0.1")))
order = lambda b: sorted(("DRIFT", "IsolationForest", "DeepAutoencoder"), key=lambda d: -E(b, d, "pr_auc"))
claim("the PR-AUC order of DRIFT, IF, and AE is the same on G2L and G2", order("E2_G2L") == order("E1_main"))
claim("received value alone exceeded DRIFT, IF, and AE on G2 (PR-AUC)",
      mean(vol_g2["pr_auc"]) > max(E("E1_main", d, "pr_auc") for d in ("DRIFT", "IsolationForest", "DeepAutoencoder")))
claim("on G2 the heuristic recalled more peeling chains than every learned detector",
      all(R("E1_main", "Heuristic", "peeling_chain") > R("E1_main", d, "peeling_chain")
          for d in ("IsolationForest", "DeepAutoencoder", "DRIFT")))

# ---------------- prevalence, contamination, hard negatives ----------------
pv = e["E3_prevalence"]
levels = ["0.005", "0.01", "0.02", "0.05", "0.1"]
rocs = [mean(pv[c]["DRIFT"]["roc_auc"]) for c in levels]
mac["TXprevroclo"], mac["TXprevrochi"] = f3(min(rocs)), f3(max(rocs))
mac["TXprevplo"], mac["TXprevphi"] = f3(mean(pv["0.005"]["DRIFT"]["P@0.05"])), f3(mean(pv["0.1"]["DRIFT"]["P@0.05"]))
mac["TXprevrlo"], mac["TXprevrhi"] = f2(mean(pv["0.005"]["DRIFT"]["R@0.05"])), f2(mean(pv["0.1"]["DRIFT"]["R@0.05"]))
ct = e["E8_contaminated_fit"]
mac["TXcontdriftone"], mac["TXcontdriftfive"] = f3(mean(ct["0.01"]["DRIFT"]["pr_auc"])), f3(mean(ct["0.05"]["DRIFT"]["pr_auc"]))
mac["TXcontifone"], mac["TXcontiffive"] = f3(mean(ct["0.01"]["IsolationForest"]["pr_auc"])), f3(mean(ct["0.05"]["IsolationForest"]["pr_auc"]))
claim("the Isolation Forest exceeded DRIFT from 0.5% contamination onward",
      all(mean(ct[c]["IsolationForest"]["pr_auc"]) > mean(ct[c]["DRIFT"]["pr_auc"]) for c in ct))
hn = e["E1_hard_negatives"]
for key, det in (("drift", "DRIFT"), ("if", "IsolationForest"), ("ae", "DeepAutoencoder"), ("heur", "Heuristic")):
    mac[f"TXhn{key}"] = f3(mean(hn["ntx_ge_100"][det]))
mac["TXhnkdrift"] = f3(mean(hn["ntx_ge_1000"]["DRIFT"]))
claim("against wallets with at least 1,000 transactions every detector except LOF was below 0.5",
      all((mean(hn["ntx_ge_1000"][d]) < 0.5) == (d != "LOF") for d in
          ["Heuristic", "OneClassSVM", "LOF", "PCA", "IsolationForest", "DeepAutoencoder", "Cascade", "DRIFT", "MaxRank"]))
mac["TXfpdrift"] = pct0(mean(hn["fp_composition_at_5pct"]["DRIFT"]["share_high_activity"]))
mac["TXfpheur"] = pct1(mean(hn["fp_composition_at_5pct"]["Heuristic"]["share_high_activity"]))
mac["TXhnshare"] = pct1(mean(hn["ntx_ge_100"]["base_rate"]))

# ---------------- evasion ----------------
ev = e["E6_evasion"]
mac["TXevdilroc"] = f3(mean(ev["dilate_30"]["DRIFT"]["roc_auc"]))
mac["TXevdilrec"] = f3(mean(ev["dilate_30"]["DRIFT"]["R@0.05"]))
mac["TXevnonerec"] = f3(mean(ev["none"]["DRIFT"]["R@0.05"]))
mac["TXevretroc"] = f3(mean(ev["retain_0.5"]["DRIFT"]["roc_auc"]))
mac["TXevsplitroc"] = f3(mean(ev["split_30"]["DRIFT"]["roc_auc"]))
mac["TXevsplitrec"] = f3(mean(ev["split_30"]["DRIFT"]["R@0.05"]))
claim("retention made the wallets more anomalous", mean(ev["retain_0.5"]["DRIFT"]["roc_auc"]) > E("E1_main", "DRIFT", "roc_auc"))
# recall at the 5% budget as fractions, as in Table VI
mac["TXevcombheur"] = f3(mean(ev["combined"]["Heuristic"]["R@0.05"]))
mac["TXevnoneheur"] = f3(mean(ev["none"]["Heuristic"]["R@0.05"]))
mac["TXevcombrec"] = f3(mean(ev["combined"]["DRIFT"]["R@0.05"]))
mac["TXevretrec"] = f3(mean(ev["retain_0.5"]["DRIFT"]["R@0.05"]))
mac["TXevsplitheur"] = f3(mean(ev["split_30"]["Heuristic"]["R@0.05"]))
claim("splitting over 30 addresses lowered the heuristic's recall to about that of DRIFT (within 0.05)",
      abs(mean(ev["split_30"]["Heuristic"]["R@0.05"]) - mean(ev["split_30"]["DRIFT"]["R@0.05"])) < 0.05)
claim("the combined modification lowered the recall of the heuristic",
      mean(ev["combined"]["Heuristic"]["R@0.05"]) < mean(ev["none"]["Heuristic"]["R@0.05"]))
claim("the combined modification removed most of the heuristic's recall at the 5% budget",
      mean(ev["combined"]["Heuristic"]["R@0.05"]) < 0.5 * mean(ev["none"]["Heuristic"]["R@0.05"]))
claim("the combined modification raised the recall of DRIFT at the 5% budget, and 50% retention alone raised it more",
      mean(ev["none"]["DRIFT"]["R@0.05"]) < mean(ev["combined"]["DRIFT"]["R@0.05"]) < mean(ev["retain_0.5"]["DRIFT"]["R@0.05"]))
claim("the heuristic was more sensitive than the learned detectors to splitting",
      mean(ev["split_30"]["Heuristic"]["R@0.05"]) / mean(ev["none"]["Heuristic"]["R@0.05"])
      < mean(ev["split_30"]["DRIFT"]["R@0.05"]) / mean(ev["none"]["DRIFT"]["R@0.05"]))
it = e["E6_interp"]
mac["TXinterpaelo"] = f3(mean(it["0.0"]["DeepAutoencoder"]["roc_auc"]))
mac["TXinterpaehi"] = f3(mean(it["0.6"]["DeepAutoencoder"]["roc_auc"]))
claim("interpolation raised the ROC-AUC of the autoencoder", mean(it["0.6"]["DeepAutoencoder"]["roc_auc"]) > mean(it["0.0"]["DeepAutoencoder"]["roc_auc"]))

# ---------------- extensions: reference, size controls ----------------
e10 = x["E10_reference"]
TABLE_RULES = ("DRIFT", "MinRank", "MaxRank", "All5MeanRank", "Gated_tau0.35")   # the rules of Table S-reference
diffs = [abs(mean(e10["in_sample"][g][r]["pr_auc"]) - mean(e10["calibration"][g][r]["pr_auc"]))
         for g in e10["in_sample"] for r in TABLE_RULES]
mac["TXrefmaxdiffall"] = f3(max(diffs))
exc = e10["exceedance"]
dev = [abs(100 * mean(exc[ref][m][q]) - 100 * (1 - float(q))) for ref in ("in_sample", "calibration")
       for m in ("IsolationForest", "DeepAutoencoder", "DRIFT") for q in ("0.9", "0.95", "0.99")]
mac["TXexcmaxdev"] = f"{max(dev):.1f}"
mt = x["E12_matched"]
vals = [mt[g][m] for g in mt for m in mt[g] if m not in ("median_distance", "k")]
vals = [mean(v) if isinstance(v, dict) else v for v in vals]
mac["TXmatchminall"], mac["TXmatchmaxall"] = f2(min(vals)), f2(max(vals))

# ---------------- qualitative statements about the external checks ----------------
o = load("ofac_external_sdn.json")["detectors"]
en = lambda m, k="enrich10_matched": o[m]["point"][k]
claim("OFAC: the matched enrichment was below one for DRIFT, the Isolation Forest, and the autoencoder",
      all(en(m) < 1 for m in ("DRIFT", "IsolationForest", "DeepAutoencoder")))
claim("OFAC: only LOF and the heuristic had a matched enrichment above one",
      {m for m in o if en(m) > 1} == {"LOF", "Heuristic"})
claim("OFAC: DRIFT ranked sanctioned addresses below controls more clearly at stricter cutoffs",
      en("DRIFT", "enrich1_matched") < en("DRIFT", "enrich5_matched") < en("DRIFT"))
claim("OFAC: the address-level interval of the DRIFT matched enrichment excludes one",
      o["DRIFT"]["ci95"]["enrich10_matched"][1] < 1)
pyx = load("pyod_external.json")["detectors"]
pyod_en = [v["ofac"]["point"]["enrich10_matched"] for v in pyx.values()]
claim("OFAC: the matched enrichment of all four PyOD detectors was below one", pyod_en and all(v < 1 for v in pyod_en))
cfx = load("ofac_crossfit_sdn.json")
cf = cfx["cross_fitted"]["detectors"]["DRIFT"]
mac["TXcfP"] = f3(cf["point"]["p_outrank_matched"])
# observation-time sensitivity: controls matched also on an empty balance (cross-fitted scores)
if "cross_fitted_zero_balance" in cfx:
    zb = cfx["cross_fitted_zero_balance"]["detectors"]
    zd = zb["DRIFT"]
    mac["TXzbS"], mac["TXzbC"] = pct1(zd["point"]["top10"]), pct1(zd["point"]["top10_ctrl"])
    mac["TXzbenrich"] = f2(zd["point"]["enrich10_matched"])
    mac["TXzbalo"], mac["TXzbahi"] = (f2(v) for v in zd["ci95_address"]["enrich10_matched"])
    mac["TXzbplo"], mac["TXzbphi"] = (f2(v) for v in zd["ci95_profile"]["enrich10_matched"])
    mac["TXzbP"] = f3(zd["point"]["p_outrank_matched"])
    mac["TXzbPlo"], mac["TXzbPhi"] = (f3(v) for v in zd["ci95_profile"]["p_outrank_matched"])
    mac["TXzbenrichIF"] = f2(zb["IsolationForest"]["point"]["enrich10_matched"])
    mac["TXzbenrichAE"] = f2(zb["DeepAutoencoder"]["point"]["enrich10_matched"])
    mac["TXzbinenrich"] = f2(cfx["in_sample_zero_balance"]["detectors"]["DRIFT"]["point"]["enrich10_matched"])
    es = cfx["empty_share"]
    mac["TXemptyS"], mac["TXemptyC"], mac["TXemptyCzb"] = (pct1(es[k]) for k in
                                                         ("sanctioned", "controls", "controls_zero_balance"))
    zlo, zhi = zd["ci95_profile"]["enrich10_matched"]
    claim("OFAC (balance matched, cross-fitted, profile interval) includes one: no evidence of a difference",
          zlo <= 1 <= zhi)
claim("OFAC: the cross-fitted profile-level interval of DRIFT excludes one", cf["ci95_profile"]["enrich10_matched"][1] < 1)
# observation time: the corpus holds wallets active at collection; the sanctioned addresses
# were last active long before their features were retrieved
import pandas as pd
sf = pd.read_csv(REV / "outputs" / "ofac_features_sdn_2026-11-12.csv")
listed = [a.strip() for a in (REV / "ofac" / "lists" / "sanctioned_addresses_XBT.txt").read_text().splitlines() if a.strip()]
valid = {a for a in listed if a.startswith(("1", "3")) or a.lower().startswith("bc1")}
sf = sf[sf["address"].astype(str).isin(valid)].drop_duplicates("address")
sf = sf[pd.to_numeric(sf["n_tx"], errors="coerce") > 0]
last = pd.to_datetime(sf["last_seen"], errors="coerce", utc=True)
mac["TXrecencyS"] = pct1(float((last < pd.Timestamp("2025-01-01", tz="UTC")).mean()))
# transfer of rank thresholds to other address sets (verified services of BABD-13)
hs0 = load("babd_external.json")["high_activity_services"]["detectors"]["DRIFT"]
mac["TXsrvexcfive"], mac["TXsrvexcten"] = pct1(hs0["top5"]), pct1(hs0["top10"])
mac["TXexchexcten"] = pct1(load("babd_external.json")["by_class"]["Centralized exchange"]["DRIFT"]["top10"])
bb = load("babd_external.json")
claim("BABD-13: every learned detector ranked illicit-associated addresses below service addresses",
      all(bb["detectors"][m]["auc_illicit_vs_service"] < 0.5
          for m in ("OneClassSVM", "LOF", "PCA", "IsolationForest", "DeepAutoencoder", "DRIFT")))
hs = bb["high_activity_services"]["detectors"]
claim("BABD-13: DRIFT separated injected wallets from verified services better than volume",
      hs["DRIFT"]["auc_injected_vs_service"] > hs["VolumeReceived"]["auc_injected_vs_service"])
claim("BABD-13: each component alone also separated injected wallets from verified services better than volume "
      "(the margin belongs to the learned detectors, not to fusion)",
      all(hs[m]["auc_injected_vs_service"] > hs["VolumeReceived"]["auc_injected_vs_service"]
          for m in ("IsolationForest", "DeepAutoencoder")))
mac["TXsancbigIF"] = f3(hs["IsolationForest"]["auc_sanctioned_ge100_vs_service"])
claim("BABD-13: for sanctioned addresses with at least 100 transactions against services, volume > Isolation Forest > DRIFT",
      hs["VolumeReceived"]["auc_sanctioned_ge100_vs_service"] > hs["IsolationForest"]["auc_sanctioned_ge100_vs_service"]
      > hs["DRIFT"]["auc_sanctioned_ge100_vs_service"])
gw_path = OUT / "variants" / "g2w_per_seed.json"
if gw_path.exists():
    from scipy.stats import wilcoxon
    gws = json.loads(gw_path.read_text())
    a = [r["G2L"]["eval"]["DRIFT"]["pr_auc"] for r in gws]
    b = [r["G2L"]["eval"]["IsolationForest"]["pr_auc"] for r in gws]
    p_g2w = 1.0 if all(abs(u - v) < 1e-15 for u, v in zip(a, b)) else float(wilcoxon(a, b).pvalue)
    below = sum(a) < sum(b) and p_g2w < 0.05
else:
    below = True
_better = {b: ("IsolationForest" if E(b, "IsolationForest", "pr_auc") > E(b, "DeepAutoencoder", "pr_auc") else "DeepAutoencoder")
           for b in ("E1_main", "E2_G3", "E2_H")}
claim("abstract 'matched ... on two and exceeded both on the third': no Holm-significant difference from the better "
      "component on G2 and G3, and Holm-significantly above both components on H",
      all(hp(b, _better[b]) >= 0.05 for b in ("E1_main", "E2_G3")) and both_holm["E2_H"])
claim("abstract: DRIFT matched the better of its components on G2 and G3 (never significantly below it) and exceeded "
      "both on H, with a clean reference (the generic anomaly types of Table III are the stated exception)",
      not below and all(not (P(bk, "DRIFT", c)["mean_diff"] < 0 and P(bk, "DRIFT", c)["wilcoxon_p"] < 0.05)
                        for bk in ("E1_main", "E2_G3", "E2_H") for c in ("IsolationForest", "DeepAutoencoder")))
vpath = OUT / "variants" / "variants.json"
if vpath.exists():
    vc = json.loads(vpath.read_text())["criteria"]
    claim("the two pre-specified variants failed their criteria (text reports failure)",
          not vc["C1"]["holds"] and not vc["variant2_succeeds"])

# ---------------- corpus: records whose transaction rate is window-based (Section III-B) ----------------
# BlockCypher records are those with nonzero value aggregates (the other services return zeros).
# The scraper computes the BlockCypher rate as the number of confirmed txrefs in the window divided by
# the span (blockchain-dataset/Wallet3.py), and BlockCypher returns one txref per input or output of the
# address. Its rate therefore departs from lifetime count / span for wallets with more than 50
# transactions (window) and for smaller wallets whose transactions touch the address more than once.
import glob as _glob
_csvs = sorted(_glob.glob(str(REV.parent / "blockchain-dataset" / "wallet_dataset_*.csv")))
if _csvs:
    _c = pd.concat([pd.read_csv(f_) for f_ in _csvs], ignore_index=True)
    for col_ in ("n_tx", "active_days", "tx_per_day", "avg_tx_value", "max_tx_value", "min_tx_value", "total_tx_value"):
        _c[col_] = pd.to_numeric(_c[col_], errors="coerce")
    _c = _c.dropna(subset=["address", "n_tx"])
    _c = _c[_c["n_tx"] >= 0].sort_values("n_tx", ascending=False).drop_duplicates("address", keep="first")
    _bc = (_c[["avg_tx_value", "max_tx_value", "min_tx_value", "total_tx_value"]].fillna(0) != 0).any(axis=1)
    _big = _c["n_tx"] > 50
    _differs = ~((_c["tx_per_day"] * _c["active_days"] - _c["n_tx"]).abs() <= 0.02 * _c["n_tx"])
    _timed = _c["active_days"] > 0
    mac["TXratewinpct"] = pct1(float((_bc & _differs).mean()))
    claim("corpus: 67,877 wallets after deduplication, 14% of them BlockCypher records", len(_c) == 67877
          and round(100 * float(_bc.mean())) == 14)
    claim("corpus: the rate departs from lifetime count / span for nearly all BlockCypher wallets with more than "
          "50 transactions and also for some with 50 or fewer (inputs and outputs counted separately)",
          float(_differs[_bc & _big].mean()) > 0.95 and float(_differs[_bc & ~_big].mean()) > 0.05)
    claim("corpus: outside BlockCypher, the rate equals lifetime count / span for every wallet with a confirmed "
          "timestamp, so the two definitions differ only in BlockCypher records",
          int((_differs & ~_bc & _timed).sum()) == 0)
else:
    mac["TXratewinpct"] = "??"
    claim("corpus files found for the window-rate share", False)

# ---------------- determinism of the run ----------------
dpath = OUT / "determinism_check.json"
if dpath.exists():
    det = json.loads(dpath.read_text())
    rr = det.get("rerun_repeat", {}).get("experiments_per_seed.json", {})
    xc = det.get("crosscheck_i7-6700K", {}).get("experiments_per_seed.json", {})
    mac["TXdetvalues"] = f"{rr.get('values_compared', 0):,}".replace(",", "{,}")
    mac["TXdetdiff"] = str(rr.get("values_different", -1))
    mac["TXdetcrossvalues"] = f"{xc.get('values_compared', 0):,}".replace(",", "{,}")
    mac["TXdetcrossdiff"] = str(xc.get("values_different", -1))
    claim("the verification run reproduced every value of the main comparison exactly",
          rr.get("values_different") == 0 and rr.get("missing_in_other") == 0)
    claim("the earlier run on the i7-6700K reproduced every value that both runs define exactly",
          xc.get("values_different") == 0 and xc.get("missing_in_other") == 0 and xc.get("values_compared", 0) > 0)
else:
    claim("determinism_check.json exists", False)

# ---------------- wording guards ----------------
# Typed (non-macro) statements that were true in an earlier run and can silently become false
# after a rerun. Each guard names a wording and the condition under which it may appear in the
# manuscript, the supplement, or the letter; a wording that appears without its condition fails.
import re
texts = {n: (REV / n).read_text(encoding="utf-8") for n in
         ("manuscript/main.tex", "response/response_to_reviewers.tex", "supplement/supplement.tex",
          "supplement/s_g2w.tex", "supplement/s_labeled.tex") if (REV / n).exists()}
# the letter reproduces the reviewers' comments verbatim; those are not statements of ours
texts["response/response_to_reviewers.tex"] = re.sub(
    r"\\begin\{comment\}.*?\\end\{comment\}", " ", texts.get("response/response_to_reviewers.tex", ""), flags=re.S)
guards = [
    (r"all remained significant|remained so after Holm", holm_sum["n_raw_below_0.05"] == holm_sum["n_holm_below_0.05"],
     "every raw-significant comparison survives the Holm correction"),
    (r"exceeded both[^.;]{0,90}independent(ly parameterized)? generator|exceeded both components on two others",
     both_holm["E2_G3"], "DRIFT exceeds both components on G3 after Holm correction"),
    (r"five detectors exceeded (DRIFT|it) on two benchmarks|five-detector mean rank was better on two",
     sum(t_["mean_diff"] < 0 for t_ in five) == 2, "the five-detector mean beats DRIFT on exactly two benchmarks"),
    (r"nothing reliably on G3 and H|no reliable difference on G3 and H",
     all(vsel[b]["pr_auc"]["wilcoxon_p"] >= 0.05 for b in ("E2_G3", "E2_H")),
     "the validation-selected rule does not differ from DRIFT on G3 and H"),
    (r"highest PR-AUC of all scores compared|strongest score on the held-out typologies|equally defensible",
     all(E("E2_H", r_, "pr_auc") <= E("E2_H", "DRIFT", "pr_auc") for r_ in rules if r_ != "DRIFT"),
     "no aggregation rule of Table V beats DRIFT on H"),
    (r"held within each script type", False, "the sanctioned-address result holds in every script type (P2WSH differs)"),
    (r"never significantly below (?:the better of its components|its better component) and exceeded",
     not adb_contradicts, "the never-below claim holds without restriction to the typology benchmarks (Table III contradicts it)"),
    (r"exceeded both(?: components| of its components)? on (?:the )?(?:four )?held-out typologies(?! combined)",
     all(R("E2_H", "DRIFT", t_) > max(R("E2_H", "IsolationForest", t_), R("E2_H", "DeepAutoencoder", t_))
         for t_ in ("hoard", "dust_fan", "structuring", "single_hop")),
     "DRIFT exceeds both components on every held-out typology, not only on the four combined"),
    (r"complementary detectors|combining complementary", not agreement,
     "the benefit is explained by complementary coverage (minimum and maximum of ranks contradict it)"),
    (r"fusion bought robustness", False, "fusion was the more robust choice (Sections VI-E and VI-F contradict it)"),
]
for pattern, cond, meaning in guards:
    hits = [n for n, s in texts.items() if re.search(pattern, s)]
    claim(f"wording guard ({meaning}): wording absent or condition true" + (f" [found in {', '.join(hits)}]" if hits else ""),
          cond or not hits)

# ---------------- write ----------------
(TAB / "numbers_text.tex").write_text("\n".join("\\newcommand{\\" + k + "}{" + v + "}" for k, v in mac.items()) + "\n",
                                      encoding="utf-8")
failed = [c for c, ok in claims if not ok]
(OUT / "text_claims_check.json").write_text(json.dumps({"claims": [{"claim": c, "holds": ok} for c, ok in claims]}, indent=1))
print(f"{len(mac)} macros written from {OUT}; {len(claims)} claims checked, {len(failed)} not supported")
for c in failed:
    print("  NOT SUPPORTED:", c)
if "--print" in sys.argv:
    for k, v in mac.items():
        print(f"  {k} = {v}")
