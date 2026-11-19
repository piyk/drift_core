"""make_tables_ext.py  |  Numeric macros and supplement tables of the added analyses.

Writes manuscript/tables/numbers_ext.tex (macros used in the manuscript, the
supplement and the response letter) and the table bodies tab_s_*.tex of the
supplement, from the stored result files:

  extensions.json            run_extensions.py
  paired_tests.json          stats_tests.py
  rerun_repeat/              a repeated run of run_experiments.py
  ofac_crossfit_<tag>.json   ofac_crossfit.py
  babd_external.json         babd_external.py

Run:  python revise1/src/make_tables_ext.py [ofac_tag]
"""
import json
import sys

import numpy as np

import drift_core as dc

TAB = dc.ROOT / "revise1" / "manuscript" / "tables"
OUT = dc.OUT_DIR
tag = sys.argv[1] if len(sys.argv) > 1 else "sdn"
g = lambda x: x["mean"]
f3 = lambda v: f"{v:.3f}"
f2 = lambda v: f"{v:.2f}"
pc = lambda v, n=1: f"{100 * v:.{n}f}"
WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
        9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}
mac = {}


def write(name, lines, rule=True):
    (TAB / name).write_text("\n".join(lines) + ("\n\\hline" if rule else "") + "\n", encoding="utf-8")
    print("wrote", name)


def load(name):
    p = OUT / name
    return json.load(open(p)) if p.exists() else None


# ---------------- repeated run ----------------
main = load("experiments.json")
rer = load("rerun_repeat/experiments.json")
if rer:
    for key, m in (("drift", "DRIFT"), ("ae", "DeepAutoencoder"), ("if", "IsolationForest")):
        mac["RR" + key + "roc"] = f3(g(rer["E1_main"][m]["roc_auc"]))
        mac["RR" + key + "pr"] = f3(g(rer["E1_main"][m]["pr_auc"]))
    rows = []
    for exp, lab in (("E1_main", "G2"), ("E2_G1", "G1"), ("E2_G3", "G3"), ("E2_H", "H")):
        for m, name in (("IsolationForest", "Isolation Forest"), ("DeepAutoencoder", "Autoencoder"),
                        ("DRIFT", "DRIFT"), ("MinRank", "Minimum of ranks")):
            a, b = main[exp][m], rer[exp][m]
            rows.append(f"{lab} & {name} & {f3(g(a['roc_auc']))} & {f3(g(b['roc_auc']))} & "
                        f"{f3(g(a['pr_auc']))} & {f3(g(b['pr_auc']))} \\\\")
    write("tab_s_rerun.tex", rows)
    per_a = load("experiments_per_seed.json")
    per_b = load("rerun_repeat/experiments_per_seed.json")
    by_seed = {r["seed"]: r for r in per_a}
    same_if = all(abs(r["E1_main"]["IsolationForest"]["pr_auc"]
                      - by_seed[r["seed"]]["E1_main"]["IsolationForest"]["pr_auc"]) < 1e-12 for r in per_b)
    d_ae = [abs(r["E1_main"]["DeepAutoencoder"]["roc_auc"]
                - by_seed[r["seed"]]["E1_main"]["DeepAutoencoder"]["roc_auc"]) for r in per_b]
    mac["RRaemaxdiff"] = f3(max(d_ae))
    from scipy.stats import wilcoxon

    def paired(exp, b):
        d = np.array([r[exp]["DRIFT"]["pr_auc"] - r[exp][b]["pr_auc"] for r in per_b])
        return float(d.mean()), int((d > 0).sum()), float(wilcoxon(d).pvalue)

    for exp, sfx in (("E1_main", "Gtwo"), ("E2_G3", "Gthree"), ("E2_H", "H")):
        for b, key in (("IsolationForest", "if"), ("DeepAutoencoder", "ae")):
            diff, wins, pv = paired(exp, b)
            mac["RRd" + key + sfx] = f"{diff:+.3f}".replace("+", "$+$").replace("-", "$-$")
            mac["RRp" + key + sfx] = f3(pv) if pv < 0.1 else f2(pv)
            mac["RRw" + key + sfx] = str(wins)
    mac["RRifidentical"] = "identical" if same_if else "not identical"

# ---------------- paired tests ----------------
pt = load("paired_tests.json")
if pt and "validation_selected_rule" in pt:
    v = pt["validation_selected_rule"]
    mac.update({"STvalpr": f3(v["E1_main"]["pr_auc"]["mean"]),
                "STvaldiff": f3(v["E1_main"]["pr_auc"]["mean_diff_vs_mean_rank"]),
                "STvalp": f3(v["E1_main"]["pr_auc"]["wilcoxon_p"]),
                "STvalprGthree": f3(v["E2_G3"]["pr_auc"]["mean"]),
                "STvalpGthree": f3(v["E2_G3"]["pr_auc"]["wilcoxon_p"]),
                "STvalprH": f3(v["E2_H"]["pr_auc"]["mean"]),
                "STvalpH": f2(v["E2_H"]["pr_auc"]["wilcoxon_p"])})
    h = pt["holm_summary"]
    mac.update({"STholmn": WORD.get(h["n_tests"], str(h["n_tests"])),
                "STholmsig": WORD.get(h["n_holm_below_0.05"], str(h["n_holm_below_0.05"])),
                "STholmraw": WORD.get(h["n_raw_below_0.05"], str(h["n_raw_below_0.05"])),
                "STholmmax": f3(h["max_holm_among_raw_significant"])})

# ---------------- extensions ----------------
x = load("extensions.json")
if x:
    e10 = x["E10_reference"]
    diffs = [abs(g(e10["calibration"][gen][r]["pr_auc"]) - g(e10["in_sample"][gen][r]["pr_auc"]))
             for gen in ("G2", "G3", "H") for r in ("DRIFT", "MinRank")]
    mac["EXTcalmaxdiff"] = f3(max(diffs))
    mac["EXTcalprin"] = f3(g(e10["in_sample"]["G2"]["DRIFT"]["pr_auc"]))
    mac["EXTcalprout"] = f3(g(e10["calibration"]["G2"]["DRIFT"]["pr_auc"]))
    ex = e10["exceedance"]
    for key, m in (("ae", "DeepAutoencoder"), ("if", "IsolationForest"), ("drift", "DRIFT")):
        mac["EXTexc" + key] = pc(g(ex["in_sample"][m]["0.95"]))
        mac["EXTexccal" + key] = pc(g(ex["calibration"][m]["0.95"]))
    write("tab_s_reference.tex", [
        f"{lab} & " + " & ".join(
            f"{f3(g(e10['in_sample'][gen][r]['pr_auc']))} & {f3(g(e10['calibration'][gen][r]['pr_auc']))}"
            for gen in ("G2", "G3", "H")) + " \\\\"
        for r, lab in (("DRIFT", "Mean of ranks (DRIFT)"), ("MinRank", "Minimum of ranks"),
                       ("MaxRank", "Maximum of ranks"), ("All5MeanRank", "Five-detector mean rank"),
                       ("Gated_tau0.35", "Gated, $\\tau=0.35$"))])
    write("tab_s_exceedance.tex", [
        f"{lab} & " + " & ".join(f"{pc(g(ex[ref][m][q]))}" for ref in ("in_sample", "calibration")
                                 for q in ("0.9", "0.95", "0.99")) + " \\\\"
        for m, lab in (("IsolationForest", "Isolation Forest"), ("DeepAutoencoder", "Autoencoder"),
                       ("LOF", "Local Outlier Factor"), ("DRIFT", "DRIFT"))])

    vol = x["E11_volume"]
    for gen, sfx in (("G2", ""), ("G3", "Gthree"), ("H", "H")):
        mac["VOLroc" + sfx] = f3(g(vol[gen]["VolumeReceived"]["roc_auc"]))
        mac["VOLpr" + sfx] = f3(g(vol[gen]["VolumeReceived"]["pr_auc"]))
    mac["VOLhnhundred"] = f3(g(vol["hard_negatives_G2"]["ntx_ge_100"]["VolumeReceived"]))
    mac["VOLhnthousand"] = f3(g(vol["hard_negatives_G2"]["ntx_ge_1000"]["VolumeReceived"]))
    mac["VOLrecall"] = f3(g(vol["G2"]["VolumeReceived"]["R@0.05"]))
    mac["VOLpone"] = pc(g(vol["G2"]["VolumeReceived"]["P@0.01"]), 0)
    # recall at fixed budgets on the held-out typologies H (Fig. 1 of the article)
    eh = main["E2_H"]
    for m, a in (("DRIFT", "drift"), ("OneClassSVM", "svm"), ("DeepAutoencoder", "ae"), ("Heuristic", "heur")):
        mac["HRfive" + a] = pc(g(eh[m]["R@0.05"]), 0)
        mac["HRten" + a] = pc(g(eh[m]["R@0.1"]), 0)
    mac["HRfivevol"] = pc(g(vol["H"]["VolumeReceived"]["R@0.05"]), 0)
    mac["HRtenvol"] = pc(g(vol["H"]["VolumeReceived"]["R@0.1"]), 0)
    mac["HRheurdust"] = pc(g(eh["Heuristic"]["R@0.05|dust_fan"]), 0)
    mac["HRheurstruct"] = pc(g(eh["Heuristic"]["R@0.05|structuring"]), 0)
    # noise-matched control of the identity-preserving (G2) noise at the widest setting
    cc = main["E4_noise"]["very_wide"]["ctrl_consistent_noise"]
    mac["CTRLconsmax"] = f3(max(g(cc[m]["auc_perturbed_vs_original"]) for m in ("DeepAutoencoder", "DRIFT")))

    mt = x["E12_matched"]
    core = ["IsolationForest", "DeepAutoencoder", "DRIFT"]
    vals = [g(mt[gen][m]) for gen in ("G2", "G3", "H") for m in core]
    mac.update({"MATCHdrift": f3(g(mt["G2"]["DRIFT"])), "MATCHif": f3(g(mt["G2"]["IsolationForest"])),
                "MATCHae": f3(g(mt["G2"]["DeepAutoencoder"])), "MATCHheur": f3(g(mt["G2"]["Heuristic"])),
                "MATCHlo": f2(min(vals)), "MATCHhi": f2(max(vals))})
    order = ["Heuristic", "OneClassSVM", "LOF", "PCA", "IsolationForest", "DeepAutoencoder", "DRIFT",
             "MinRank", "VolumeReceived"]
    label = {"Heuristic": "Rule-based heuristic", "OneClassSVM": "One-Class SVM", "LOF": "Local Outlier Factor",
             "PCA": "PCA reconstruction", "IsolationForest": "Isolation Forest",
             "DeepAutoencoder": "Deep autoencoder", "DRIFT": "DRIFT (mean rank)",
             "MinRank": "Minimum of ranks", "VolumeReceived": "Received value alone"}
    write("tab_s_matched.tex", [f"{label[m]} & " + " & ".join(f3(g(mt[gen][m])) for gen in ("G2", "G3", "H")) + " \\\\"
                                for m in order])

    if "E13_pyod" in x:
        py = x["E13_pyod"]
        for k in ("ECOD", "COPOD", "DIF", "DeepSVDD"):
            for gen, sfx in (("G2", ""), ("G3", "Gthree"), ("H", "H")):
                mac[k + "roc" + sfx] = f3(g(py[gen][k]["roc_auc"]))
                mac[k + "pr" + sfx] = f3(g(py[gen][k]["pr_auc"]))
        mac["DIFpone"] = pc(g(py["G2"]["DIF"]["P@0.01"]), 0)

    ad = x["E14_adbench"]
    kinds = [("global", "Global"), ("local", "Local"), ("cluster", "Clustered"), ("dependency", "Dependency")]
    mac["ADBviolreal"] = pc(g(ad["violations_real"]["any"]))
    viol = [g(ad["feature_space"][k]["violations"]["any"]) for k, _ in kinds]
    mac["ADBviollo"], mac["ADBviolhi"] = pc(min(viol), 0), pc(max(viol), 0)
    for k, _ in kinds:
        mac["ADBviol" + k] = pc(g(ad["feature_space"][k]["violations"]["any"]), 0)
        mac["ADBaefeat" + k] = f3(g(ad["feature_space"][k]["DeepAutoencoder"]))
        mac["ADBaeprim" + k] = f3(g(ad["primitive_space"][k]["DeepAutoencoder"]))
        mac["ADBdriftfeat" + k] = f3(g(ad["feature_space"][k]["DRIFT"]))
        mac["ADBdriftprim" + k] = f3(g(ad["primitive_space"][k]["DRIFT"]))
    aef = [g(ad["feature_space"][k]["DeepAutoencoder"]) for k, _ in kinds]
    aep = [g(ad["primitive_space"][k]["DeepAutoencoder"]) for k, _ in kinds]
    mac.update({"ADBaefeatlo": f3(min(aef)), "ADBaefeathi": f3(max(aef)),
                "ADBaeprimlo": f3(min(aep)), "ADBaeprimhi": f3(max(aep))})
    rows = []
    for k, lab in kinds:
        fs, ps = ad["feature_space"][k], ad["primitive_space"][k]
        rows.append(f"{lab} & {pc(g(fs['violations']['any']), 1)} & {f3(g(fs['IsolationForest']))} & "
                    f"{f3(g(fs['DeepAutoencoder']))} & {f3(g(fs['DRIFT']))} & "
                    f"{pc(g(ps['violations']['any']), 1)} & {f3(g(ps['IsolationForest']))} & "
                    f"{f3(g(ps['DeepAutoencoder']))} & {f3(g(ps['DRIFT']))} \\\\")
    write("tab_s_adbench.tex", rows)

# ---------------- evasion (full table for the supplement) ----------------
if main:
    ops = [("none", "No modification"), ("dilate_3", "Time dilation $\\times 3$"),
           ("dilate_10", "Time dilation $\\times 10$"), ("dilate_30", "Time dilation $\\times 30$"),
           ("retain_0.1", "Retention 10\\%"), ("retain_0.3", "Retention 30\\%"),
           ("retain_0.5", "Retention 50\\%"), ("split_3", "Splitting over 3 addresses"),
           ("split_10", "Splitting over 10 addresses"), ("split_30", "Splitting over 30 addresses"),
           ("combined", "Combined ($\\times 10$, 30\\%, 10)")]
    Ev = main["E6_evasion"]
    write("tab_s_evasion.tex", [
        f"{lab} & {f3(g(Ev[k]['IsolationForest']['roc_auc']))} & {f3(g(Ev[k]['DeepAutoencoder']['roc_auc']))} & "
        f"{f3(g(Ev[k]['DRIFT']['roc_auc']))} & {f3(g(Ev[k]['DRIFT']['R@0.05']))} & "
        f"{f3(g(Ev[k]['Heuristic']['R@0.05']))} \\\\" for k, lab in ops])

# ---------------- cross-fitted external check ----------------
cf = load(f"ofac_crossfit_{tag}.json")
if cf:
    D, I = cf["cross_fitted"]["detectors"], cf["in_sample"]["detectors"]
    dr = D["DRIFT"]
    mac.update({
        "CFnprofiles": str(cf["n_profiles"]), "CFlargest": str(cf["largest_profile_addresses"]),
        "CFtopfive": pc(cf["top5_profiles_share"], 0),
        "CFmatched": f2(dr["point"]["enrich10_matched"]),
        "CFplo": f2(dr["ci95_profile"]["enrich10_matched"][0]),
        "CFphi": f2(dr["ci95_profile"]["enrich10_matched"][1]),
        "CFalo": f2(dr["ci95_address"]["enrich10_matched"][0]),
        "CFahi": f2(dr["ci95_address"]["enrich10_matched"][1]),
        "CFtop": pc(dr["point"]["top10"]), "CFctrl": pc(dr["point"]["top10_ctrl"]),
        "CFout": f3(dr["point"]["p_outrank_matched"]),
        "CFoutlo": f3(dr["ci95_profile"]["p_outrank_matched"][0]),
        "CFouthi": f3(dr["ci95_profile"]["p_outrank_matched"][1]),
        "CFinmatched": f2(I["DRIFT"]["point"]["enrich10_matched"]),
        "CFinplo": f2(I["DRIFT"]["ci95_profile"]["enrich10_matched"][0]),
        "CFinphi": f2(I["DRIFT"]["ci95_profile"]["enrich10_matched"][1]),
        "CFmatchedIF": f2(D["IsolationForest"]["point"]["enrich10_matched"]),
        "CFmatchedAE": f2(D["DeepAutoencoder"]["point"]["enrich10_matched"]),
    })
    learned = ["OneClassSVM", "PCA", "IsolationForest", "DeepAutoencoder", "DRIFT"]
    mac["CFlearnedhi"] = f2(max(D[m]["ci95_profile"]["enrich10_matched"][1] for m in learned))
    name = {"Heuristic": "Rule-based heuristic", "OneClassSVM": "One-Class SVM", "LOF": "Local Outlier Factor",
            "PCA": "PCA reconstruction", "IsolationForest": "Isolation Forest",
            "DeepAutoencoder": "Deep autoencoder", "DRIFT": "DRIFT (mean rank)", "MinRank": "Minimum of ranks"}
    rows = []
    for m in ["Heuristic", "OneClassSVM", "LOF", "PCA", "IsolationForest", "DeepAutoencoder", "DRIFT", "MinRank"]:
        a, b = I[m], D[m]
        rows.append(f"{name[m]} & {pc(a['point']['top10'])} & {pc(a['point']['top10_ctrl'])} & "
                    f"{f2(a['point']['enrich10_matched'])} & {pc(b['point']['top10'])} & "
                    f"{pc(b['point']['top10_ctrl'])} & {f2(b['point']['enrich10_matched'])} "
                    f"({f2(b['ci95_address']['enrich10_matched'][0])}--{f2(b['ci95_address']['enrich10_matched'][1])}) & "
                    f"({f2(b['ci95_profile']['enrich10_matched'][0])}--{f2(b['ci95_profile']['enrich10_matched'][1])}) \\\\")
    write("tab_s_crossfit.tex", rows)
    rows, big = [], []
    for k, v in cf["strata"].items():
        d = v["detectors"]["DRIFT"]
        lab = k.replace("program: ", "").replace("party: ", "")
        lab = {"without largest profile": "Without the largest profile"}.get(lab, lab)
        ci = d["ci95_profile"]["enrich10_matched"]
        ci_txt = f"({f2(ci[0])}--{f2(ci[1])})" if v["n_addresses"] >= 20 and v["n_profiles"] >= 4 and np.isfinite(ci).all() else "n/a"
        rows.append(f"{lab} & {v['n_addresses']} & {v['n_profiles']} & {pc(d['point']['top10'])} & "
                    f"{pc(d['point']['top10_ctrl'])} & {f2(d['point']['enrich10_matched'])} & {ci_txt} \\\\")
        if k.startswith("program: ") and v["n_addresses"] >= 20:
            big.append(d["point"]["enrich10_matched"])
    write("tab_s_strata.tex", rows)
    mac.update({"CFstratalo": f2(min(big)), "CFstratahi": f2(max(big)),
                "CFstratan": WORD.get(len(big), str(len(big))),
                "CFnolargest": f2(cf["strata"]["without largest profile"]["detectors"]["DRIFT"]["point"]["enrich10_matched"]),
                "CFindividual": f2(cf["strata"]["party: Individual"]["detectors"]["DRIFT"]["point"]["enrich10_matched"]),
                "CFentity": f2(cf["strata"]["party: Entity"]["detectors"]["DRIFT"]["point"]["enrich10_matched"])})
    rb = cf["residual_balance"]
    flab = {"flow_ratio": "Flow ratio", "retention_ratio": "Retention ratio", "zero_balance_flag": "Zero-balance flag",
            "log_tx_per_day": "Log transaction rate", "burst_index": "Burst index",
            "log_recv_per_tx": "Log received value per transaction", "log_final_balance": "Log final balance"}
    write("tab_s_balance.tex", [f"{flab[k]} & {f3(v['sanctioned_mean'])} & {f3(v['controls_mean'])} & "
                                f"{v['smd_vs_controls']:+.2f} & {v['smd_vs_corpus']:+.2f} \\\\" for k, v in rb.items()])
    mac["CFsmdzero"] = f2(abs(rb["zero_balance_flag"]["smd_vs_controls"]))
    mac["CFsmdbalance"] = f2(abs(rb["log_final_balance"]["smd_vs_controls"]))
    lof = D["LOF"]
    mac.update({"CFmatchedLOF": f2(lof["point"]["enrich10_matched"]),
                "CFploLOF": f2(lof["ci95_profile"]["enrich10_matched"][0]),
                "CFphiLOF": f2(lof["ci95_profile"]["enrich10_matched"][1]),
                "CFmatchedHeur": f2(D["Heuristic"]["point"]["enrich10_matched"])})
    mac["CFsmdresidmax"] = f2(max(abs(v["smd_vs_controls"]) for v in rb.values()))

rk = load("ofac_recheck.json")
if rk:
    cal = rk["matching"]["5 nearest within caliper 0.25 SD"]
    mac["RCKcalipern"] = str(cal["addresses_matched"])
    mac["RCKcalipermatched"] = f2(cal["matched_enrichment"])
    rows = []
    for k, v in rk["matching"].items():
        rows.append(f"{k.replace(' (paper)', ' (main analysis)')} & {v['addresses_matched']} & "
                    f"{pc(v['sanctioned_top10'])} & {pc(v['controls_top10'])} & {f2(v['matched_enrichment'])} \\\\")
    write("tab_s_designs.tex", rows)
    mac["RCKvoltop"] = pc(rk["volume_only"]["top10"])

# ---------------- labeled external check (BABD-13 sample) ----------------
bb = load("babd_external.json")
if bb:
    B = bb["detectors"]
    learned = ["OneClassSVM", "LOF", "PCA", "IsolationForest", "DeepAutoencoder", "DRIFT"]
    aucs = [B[m]["auc_illicit_vs_service"] for m in learned]
    mac.update({
        "BBsampled": f"{bb['n_sampled']:,}".replace(",", "{,}"), "BBn": f"{bb['n_scored']:,}".replace(",", "{,}"),
        "BBnill": str(bb["n_illicit"]), "BBnsrv": str(bb["n_service"]),
        "BBaucdrift": f3(B["DRIFT"]["auc_illicit_vs_service"]),
        "BBauclo": f3(B["DRIFT"]["auc_ci95"][0]), "BBauchi": f3(B["DRIFT"]["auc_ci95"][1]),
        "BBaucif": f3(B["IsolationForest"]["auc_illicit_vs_service"]),
        "BBaucae": f3(B["DeepAutoencoder"]["auc_illicit_vs_service"]),
        "BBaucheur": f3(B["Heuristic"]["auc_illicit_vs_service"]),
        "BBaucvol": f3(B["VolumeReceived"]["auc_illicit_vs_service"]),
        "BBaucmin": f2(min(aucs)), "BBaucmax": f2(max(aucs)),
        "BBaucmatched": f3(B["DRIFT"]["auc_matched_service"]),
        "BBtopill": pc(B["DRIFT"]["top10_illicit"]), "BBtopsrv": pc(B["DRIFT"]["top10_service"]),
        "BBmedill": f"{B['DRIFT']['median_pct_illicit']:.1f}", "BBmedsrv": f"{B['DRIFT']['median_pct_service']:.1f}",
        "BBnbig": str(bb["n_illicit_ge100"] + bb["n_service_ge100"]),
        "BBnsrvbig": str(bb["n_service_ge100"]),
        "BBsmallshare": pc(bb["illicit_small_classes"]["share_of_illicit"], 0),
        "BBtopexch": pc(bb["by_class"]["Centralized exchange"]["DRIFT"]["top10"], 0),
        "BBmedblack": f"{bb['by_class']['Blackmail']['DRIFT']['median_pct']:.0f}",
        "BBmedponzi": f"{bb['by_class']['Ponzi scheme']['DRIFT']['median_pct']:.0f}",
        "BBnblackponzi": str(bb["by_class"]["Blackmail"]["n"] + bb["by_class"]["Ponzi scheme"]["n"]),
    })
    if "auc_within_script_type" in bb:
        wt = bb["auc_within_script_type"]
        mac.update({
            "BBtypeill": pc(bb["script_type_by_group"]["illicit"].get("P2SH", 0), 0),
            "BBtypesrv": pc(bb["script_type_by_group"]["service"].get("Legacy", 0), 0),
            "BBrhospan": f2(bb["drift_spearman"]["log_active_days"]),
            "BBrhocount": f2(bb["drift_spearman"]["log_n_tx"]),
            "BBrhovalue": f2(bb["drift_spearman"]["log_received"]),
        })
        for t, key in (("Legacy", "pkh"), ("P2SH", "sh")):
            if t in wt:
                v = wt[t]
                mac.update({
                    "BBn" + key + "ill": str(v["n_illicit"]), "BBn" + key + "srv": str(v["n_service"]),
                    "BBauc" + key: f3(v["DRIFT"]["auc"]),
                    "BBauc" + key + "lo": f3(v["DRIFT"]["ci95"][0]), "BBauc" + key + "hi": f3(v["DRIFT"]["ci95"][1]),
                    "BBvol" + key: f3(v["VolumeReceived"]["auc"])})
        tname = {"Legacy": "P2PKH", "P2SH": "P2SH", "P2WPKH": "P2WPKH"}
        sname = {"Heuristic": "Rule-based heuristic", "OneClassSVM": "One-Class SVM", "LOF": "Local Outlier Factor",
                 "PCA": "PCA reconstruction", "IsolationForest": "Isolation Forest",
                 "DeepAutoencoder": "Deep autoencoder", "DRIFT": "DRIFT (mean rank)", "MinRank": "Minimum of ranks",
                 "VolumeReceived": "Received value alone", "VolumeCount": "Transaction count alone"}
        types = [t for t in ("Legacy", "P2SH") if t in wt]
        write("tab_s_babd_type.tex", [
            f"{sname[m]} & " + " & ".join(
                f"{f3(wt[t][m]['auc'])} ({f3(wt[t][m]['ci95'][0])}--{f3(wt[t][m]['ci95'][1])})" for t in types) + " \\\\"
            for m in sname])
    sc = load("babd_source_check.json")
    if sc:
        mac["BBsrcn"] = str(sc["n_compared"])
        mac["BBsrcsame"] = "identical" if sc["all_identical"] else "not identical"
    hs = bb.get("high_activity_services")
    if hs:
        H = hs["detectors"]
        core = ["IsolationForest", "DeepAutoencoder", "DRIFT"]
        mac.update({
            "HSn": str(hs["n_scored"]), "HSsampled": str(hs["n_sampled"]),
            "HSmedtx": f"{hs['median_n_tx']:.0f}", "HSmedrecv": f"{hs['median_received_btc']:.1f}",
            "HSbigshare": pc(hs["share_ntx_ge_100"], 0),
            "HStop": pc(H["DRIFT"]["top10"]), "HSmedpct": f"{H['DRIFT']['median_pct']:.1f}",
            "HStopvol": pc(H["VolumeReceived"]["top10"]),
            "HSaucinj": f3(H["DRIFT"]["auc_injected_vs_service"]),
            "HSaucinjlo": f3(H["DRIFT"]["auc_injected_ci95"][0]),
            "HSaucinjhi": f3(H["DRIFT"]["auc_injected_ci95"][1]),
            "HSaucinjIF": f3(H["IsolationForest"]["auc_injected_vs_service"]),
            "HSaucinjAE": f3(H["DeepAutoencoder"]["auc_injected_vs_service"]),
            "HSaucinjheur": f3(H["Heuristic"]["auc_injected_vs_service"]),
            "HSaucinjvol": f3(H["VolumeReceived"]["auc_injected_vs_service"]),
            "HSaucsanc": f3(H["DRIFT"]["auc_sanctioned_vs_service"]),
            "HSaucsanclo": f3(H["DRIFT"]["auc_sanctioned_ci95"][0]),
            "HSaucsanchi": f3(H["DRIFT"]["auc_sanctioned_ci95"][1]),
            "HSaucsancbig": f3(H["DRIFT"]["auc_sanctioned_ge100_vs_service"]),
            "HSaucsancbigvol": f3(H["VolumeReceived"]["auc_sanctioned_ge100_vs_service"]),
            "HSnsancbig": str(hs["sanctioned_ge100"]),
            "HSaucsancmin": f2(min(H[m]["auc_sanctioned_vs_service"] for m in core)),
            "HSaucsancmax": f2(max(H[m]["auc_sanctioned_vs_service"] for m in core)),
            "HSaucinjmin": f2(min(H[m]["auc_injected_vs_service"] for m in core)),
            "HSaucinjmax": f2(max(H[m]["auc_injected_vs_service"] for m in core)),
        })
        hname = {"Heuristic": "Rule-based heuristic", "OneClassSVM": "One-Class SVM", "LOF": "Local Outlier Factor",
                 "PCA": "PCA reconstruction", "IsolationForest": "Isolation Forest",
                 "DeepAutoencoder": "Deep autoencoder", "DRIFT": "DRIFT (mean rank)", "MinRank": "Minimum of ranks",
                 "VolumeReceived": "Received value alone", "VolumeCount": "Transaction count alone"}
        write("tab_s_high.tex", [
            f"{hname[m]} & {v['median_pct']:.1f} & {pc(v['top10'])} & "
            f"{f3(v['auc_injected_vs_service'])} ({f3(v['auc_injected_ci95'][0])}--{f3(v['auc_injected_ci95'][1])}) & "
            f"{f3(v['auc_sanctioned_vs_service'])} ({f3(v['auc_sanctioned_ci95'][0])}--{f3(v['auc_sanctioned_ci95'][1])}) & "
            f"{f3(v['auc_sanctioned_ge100_vs_service'])} \\\\" for m, v in H.items()])
    name = {"Heuristic": "Rule-based heuristic", "OneClassSVM": "One-Class SVM", "LOF": "Local Outlier Factor",
            "PCA": "PCA reconstruction", "IsolationForest": "Isolation Forest",
            "DeepAutoencoder": "Deep autoencoder", "DRIFT": "DRIFT (mean rank)", "MinRank": "Minimum of ranks",
            "VolumeReceived": "Received value alone", "VolumeCount": "Transaction count alone"}
    write("tab_s_babd.tex", [
        f"{name[m]} & {f3(v['auc_illicit_vs_service'])} ({f3(v['auc_ci95'][0])}--{f3(v['auc_ci95'][1])}) & "
        f"{f3(v['auc_matched_service'])} & {v['median_pct_illicit']:.1f} & {v['median_pct_service']:.1f} & "
        f"{pc(v['top10_illicit'])} & {pc(v['top10_service'])} \\\\" for m, v in B.items()])
    rows = []
    for cls, v in sorted(bb["by_class"].items(), key=lambda kv: (kv[1]["group"], kv[0])):
        grp = "Illicit-associated" if v["group"] == "illicit" else "Service or user"
        rows.append(f"{cls} & {grp} & {v['n']} & {v['median_n_tx']:.0f} & {v['median_received_btc']:.2f} & "
                    f"{v['DRIFT']['median_pct']:.1f} & {pc(v['DRIFT']['top10'])} & "
                    f"{pc(v['VolumeReceived']['top10'])} \\\\")
    write("tab_s_babd_class.tex", rows)

# ---------------- hard negatives and per-typology recall on H (supplement) ----------------
LBL = {"Heuristic": "Rule-based heuristic", "OneClassSVM": "One-Class SVM", "LOF": "Local Outlier Factor",
       "PCA": "PCA reconstruction", "IsolationForest": "Isolation Forest", "DeepAutoencoder": "Deep autoencoder",
       "DRIFT": "DRIFT (mean rank)", "MaxRank": "Maximum of ranks", "MinRank": "Minimum of ranks",
       "VolumeReceived": "Received value alone", "ECOD": "ECOD", "COPOD": "COPOD",
       "DIF": "Deep Isolation Forest", "DeepSVDD": "Deep SVDD"}
pm = lambda v: f"{v['mean']:.3f}\\,$\\pm$\\,{v['std']:.3f}"
hn = main["E1_hard_negatives"]
fpc = hn["fp_composition_at_5pct"]
rows = []
for m in ("Heuristic", "OneClassSVM", "LOF", "PCA", "IsolationForest", "DeepAutoencoder", "DRIFT", "MaxRank"):
    share = pc(g(fpc[m]["share_high_activity"])) if m in fpc else "n/a"
    rows.append(f"{LBL[m]} & {pm(hn['ntx_ge_100'][m])} & {pm(hn['ntx_ge_1000'][m])} & {share} \\\\")
if x:
    v = x["E11_volume"]["hard_negatives_G2"]
    rows.append(f"{LBL['VolumeReceived']} & {pm(v['ntx_ge_100']['VolumeReceived'])} & "
                f"{pm(v['ntx_ge_1000']['VolumeReceived'])} & n/a \\\\")
write("tab_s_hardneg.tex", rows)
mac["HNnhundred"] = f"{g(hn['ntx_ge_100']['n_negatives']):,.0f}".replace(",", "{,}")
mac["HNnthousand"] = f"{g(hn['ntx_ge_1000']['n_negatives']):.0f}"

pcpm = lambda v: f"{100 * v['mean']:.1f}\\,$\\pm$\\,{100 * v['std']:.1f}"
HT = ("hoard", "dust_fan", "structuring", "single_hop")
rows = []
srcs = [(m, main["E2_H"][m]) for m in ("Heuristic", "OneClassSVM", "LOF", "PCA", "IsolationForest",
                                         "DeepAutoencoder", "DRIFT", "MaxRank", "MinRank")]
if x:
    srcs.append(("VolumeReceived", x["E11_volume"]["H"]["VolumeReceived"]))
    srcs += [(m, x["E13_pyod"]["H"][m]) for m in ("ECOD", "COPOD", "DIF", "DeepSVDD")]
for m, d in srcs:
    rows.append(f"{LBL[m]} & {pcpm(d['R@0.05'])} & " + " & ".join(pcpm(d["R@0.05|" + t]) for t in HT) + " \\\\")
write("tab_s_htyp.tex", rows)

# ---------------- external checks of the PyOD detectors (pyod_external.py) ----------------
pe = load("pyod_external.json")
if pe:
    rows = []
    for m in ("ECOD", "COPOD", "DIF", "DeepSVDD"):
        d = pe["detectors"][m]
        o_, b_, h_ = d["ofac"], d["babd"], d["high"]
        rows.append(
            f"{LBL[m]} & {pc(o_['point']['top10'])} & {pc(o_['point']['top10_ctrl'])} & "
            f"{o_['point']['enrich10_matched']:.2f} ({o_['ci95']['enrich10_matched'][0]:.2f}--{o_['ci95']['enrich10_matched'][1]:.2f}) & "
            f"{f3(o_['point']['p_outrank_matched'])} & "
            f"{f3(b_['auc_illicit_vs_service'])} ({f3(b_['auc_ci95'][0])}--{f3(b_['auc_ci95'][1])}) & "
            f"{pc(h_['top10'])} & {f3(h_['auc_injected_vs_high'])} \\\\")
    write("tab_s_pyod_ext.tex", rows)
    en = [pe["detectors"][m]["ofac"]["point"]["enrich10_matched"] for m in pe["detectors"]]
    po = [pe["detectors"][m]["ofac"]["point"]["p_outrank_matched"] for m in pe["detectors"]]
    ba = [pe["detectors"][m]["babd"]["auc_illicit_vs_service"] for m in pe["detectors"]]
    hi = [pe["detectors"][m]["high"]["auc_injected_vs_high"] for m in pe["detectors"]]
    mac.update({"PXenlo": f2(min(en)), "PXenhi": f2(max(en)), "PXoutlo": f3(min(po)), "PXouthi": f3(max(po)),
                "PXbabdlo": f2(min(ba)), "PXbabdhi": f2(max(ba)), "PXhighlo": f2(min(hi)), "PXhighhi": f2(max(hi)),
                "PXenecod": f2(pe["detectors"]["ECOD"]["ofac"]["point"]["enrich10_matched"])})

# ---------------- ADBench repair control (adbench_repair.py) ----------------
ar = load("adbench_repair.json")
if ar:
    kinds = (("global", "Global"), ("local", "Local"), ("cluster", "Clustered"), ("dependency", "Dependency"))
    rows = []
    for k, lab_ in kinds:
        a = ar[k]
        rows.append(f"{lab_} & {pc(a['invalid_feature_space'])} & {pc(a['invalid_repaired'])} & " + " & ".join(
            f"{f3(g(a['feature_space'][m]))} & {f3(g(a['repaired'][m]))}"
            for m in ("IsolationForest", "LOF", "DeepAutoencoder", "DRIFT")) + " \\\\")
    write("tab_s_repair.tex", rows)
    three = ("global", "cluster", "dependency")
    rep_ae = [g(ar[k]["repaired"]["DeepAutoencoder"]) for k in three]
    fs_ae = [g(ar[k]["feature_space"]["DeepAutoencoder"]) for k in three]
    rep_if = [g(ar[k]["repaired"]["IsolationForest"]) for k in three]
    fs_if = [g(ar[k]["feature_space"]["IsolationForest"]) for k in three]
    mac.update({"REPaelo": f3(min(rep_ae)), "REPaehi": f3(max(rep_ae)),
                "REPaefslo": f3(min(fs_ae)),
                "REPifdiffmax": f3(max(abs(a_ - b_) for a_, b_ in zip(fs_if, rep_if)))})

# ---------------- compact tables of the article (evidence for the external checks) ----------------
cf_ = load("ofac_crossfit_sdn.json")
if cf_:
    Dx = cf_["crossfit"]["detectors"] if "crossfit" in cf_ else None
if cf_ and Dx is None:
    # locate the cross-fitted block by its profile intervals
    Dx = next(v["detectors"] for v in cf_.values()
              if isinstance(v, dict) and "detectors" in v and "ci95_profile" in next(iter(v["detectors"].values())))
if cf_:
    short = {"Heuristic": "Heuristic", "OneClassSVM": "One-Class SVM", "LOF": "LOF", "PCA": "PCA",
             "IsolationForest": "Isolation Forest", "DeepAutoencoder": "Autoencoder", "DRIFT": "DRIFT"}
    write("tab_crossfit_main.tex", [
        f"{short[m]} & {pc(Dx[m]['point']['top10'])} & {pc(Dx[m]['point']['top10_ctrl'])} & "
        f"{f2(Dx[m]['point']['enrich10_matched'])} & "
        f"{f2(Dx[m]['ci95_profile']['enrich10_matched'][0])}--{f2(Dx[m]['ci95_profile']['enrich10_matched'][1])} \\\\"
        for m in short])
    rows = []
    dlab = {"1 nearest, same covariates": "1 nearest control", "10 nearest, same covariates": "10 nearest controls",
            "1:1 without replacement": "1:1 without replacement",
            "5 nearest within caliper 0.25 SD": "Caliper of 0.25 SD",
            "5 nearest, type + value only": "Type and value only",
            "5 nearest, type + value + count": "Type, value, and count",
            "5 nearest, type x empty + count + value + span": "Also matched on an empty balance"}
    if rk:
        for k, lab_ in dlab.items():
            v = rk["matching"][k]
            rows.append(f"{lab_} & {v['addresses_matched']} & {pc(v['sanctioned_top10'])} & "
                        f"{pc(v['controls_top10'])} & {f2(v['matched_enrichment'])} \\\\")
    # The strata (sanctions programs, party types, without the largest profile) are reported
    # in Table S-strata of the supplement (tab_s_strata.tex); the article keeps the designs only.
    write("tab_checks_main.tex", rows)
if pe:
    write("tab_ofac_pyod.tex", [
        f"{n_} & {pe['detectors'][m]['ofac']['point']['median_pct']:.1f} & "
        f"{pc(pe['detectors'][m]['ofac']['point']['top10'])} & {pc(pe['detectors'][m]['ofac']['top10_same_type'])} & "
        f"{pc(pe['detectors'][m]['ofac']['point']['top10_ctrl'])} & "
        f"{pe['detectors'][m]['ofac']['point']['enrich10_matched']:.2f} "
        f"({pe['detectors'][m]['ofac']['ci95']['enrich10_matched'][0]:.2f}--{pe['detectors'][m]['ofac']['ci95']['enrich10_matched'][1]:.2f}) & "
        f"{f3(pe['detectors'][m]['ofac']['point']['p_outrank_matched'])} \\\\"
        for m, n_ in (("ECOD", "ECOD"), ("COPOD", "COPOD"), ("DIF", "Deep IF"), ("DeepSVDD", "Deep SVDD"))])
if ar:
    write("tab_repair_main.tex", [
        f"{lab_} & {pc(ar[k]['invalid_feature_space'], 0)} & " + " & ".join(
            f"{f3(g(ar[k]['feature_space'][m]))} & {f3(g(ar[k]['repaired'][m]))}"
            for m in ("IsolationForest", "DeepAutoencoder", "DRIFT")) + " \\\\"
        for k, lab_ in (("global", "Global"), ("local", "Local"), ("cluster", "Clustered"), ("dependency", "Dependency"))])

write("numbers_ext.tex", ["\\newcommand{\\" + k + "}{" + v + "}" for k, v in mac.items()], rule=False)
print(len(mac), "macros")
