"""ofac_crossfit.py  |  External check with out-of-sample controls and clustered uncertainty.

ofac_external.py fits the detectors on the whole corpus, so the matched control
wallets are scored by models that were trained on them, whereas the sanctioned
addresses are unseen. This script removes that asymmetry and adds three checks.

  1. Cross-fitting: the corpus is divided into five folds. Each fold is scored
     by detectors fitted on the other four, and its out-of-sample scores are
     the reference distribution against which the fold itself and the
     sanctioned addresses are ranked. Sanctioned percentiles are averaged over
     the five fold models. The in-sample design of ofac_external.py is rerun in
     the same process, so the two designs are compared on the same machine.
  2. Uncertainty by designated profile: the listed addresses belong to a small
     number of designated persons and entities (ofac_programs.py). Bootstrap
     intervals are given both over addresses and over profiles.
  3. Strata: sanctions program group and party type, and the sample without its
     largest profile.
  4. Balance of the features that were not used for matching.

Run (from revise1/src, after ofac_programs.py):
    python ofac_crossfit.py ../outputs/ofac_features_sdn_2026-11-12.csv sdn ../ofac/lists/sanctioned_addresses_XBT.txt
"""
import json
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.model_selection import KFold

import drift_core as dc
import ofac_external as oe

K_FOLDS = 5
DETS = ["Heuristic", "OneClassSVM", "LOF", "PCA", "IsolationForest", "DeepAutoencoder",
        "DRIFT", "MinRank"]
BASE7 = dc.BASE + ["Cascade"]
KEYS = ("median_pct", "top10", "top10_ctrl", "enrich10_matched", "p_outrank_matched")
RESIDUAL = ["flow_ratio", "retention_ratio", "zero_balance_flag", "log_tx_per_day",
            "burst_index", "log_recv_per_tx", "log_final_balance"]


def pct(ref, v):
    rs = np.sort(ref)
    return np.searchsorted(rs, v, side="right") / len(rs) * 100.0


def crossfit_seed(seed, corpus, sanc):
    """Out-of-sample percentiles of every corpus wallet and of every sanctioned
    address for one seed."""
    torch.set_num_threads(2)
    feats_c = dc.engineer_features(corpus)
    Fc = feats_c.values.astype(float)
    feats_s = dc.engineer_features(sanc)
    Fs = feats_s.values.astype(float)
    Pc = {m: np.zeros(len(corpus)) for m in DETS}
    Ps = {m: np.zeros(len(sanc)) for m in DETS}
    for tr, ho in KFold(n_splits=K_FOLDS, shuffle=True, random_state=seed).split(Fc):
        pre = dc.Preprocessor().fit(Fc[tr])
        suite = dc.Suite(seed).fit(pre.transform(Fc[tr]), feats_c.iloc[tr])
        raw_h = suite.score(pre.transform(Fc[ho]), feats_c.iloc[ho])
        raw_s = suite.score(pre.transform(Fs), feats_s)
        ref = {m: raw_h[m] for m in BASE7}
        sh, ss = dict(raw_h), dict(raw_s)
        sh.update(dc.fusion_scores(dc.ecdf_ranks(ref, ref), dc.ref_zscores(ref, ref)))
        ss.update(dc.fusion_scores(dc.ecdf_ranks(ref, {m: raw_s[m] for m in BASE7}),
                                   dc.ref_zscores(ref, {m: raw_s[m] for m in BASE7})))
        for m in DETS:
            Pc[m][ho] = pct(sh[m], sh[m])
            Ps[m] += pct(sh[m], ss[m]) / K_FOLDS
    print(f"[cross-fit seed {seed}] done", flush=True)
    return {m: (Ps[m], Pc[m]) for m in DETS}


def insample_seed(seed, corpus, sanc):
    out = oe.fit_and_score(seed, corpus, sanc)
    return {m: out[m] for m in DETS}


def block(Ps, Pc, idx):
    b = oe.stats_block(Ps, Pc, None, None, idx)
    return {k: b[k] for k in KEYS}


def summarize(runs, ctrl, sel, profiles, rng, n_boot):
    """Point estimates with bootstrap intervals over addresses and over
    designated profiles, for the addresses selected by the boolean mask sel."""
    idx_all = np.where(sel)[0]
    prof = np.asarray(profiles)[idx_all]
    uniq = np.unique(prof)
    members = {p: idx_all[prof == p] for p in uniq}
    boot_a = [idx_all[rng.randint(0, len(idx_all), len(idx_all))] for _ in range(n_boot)]
    boot_p = [np.concatenate([members[p] for p in rng.choice(uniq, size=len(uniq))]) for _ in range(n_boot)]
    out = {"n_addresses": int(len(idx_all)), "n_profiles": int(len(uniq)), "detectors": {}}
    for m in DETS:
        Ps = np.stack([r[m][0] for r in runs])
        Pc = np.stack([r[m][1][ctrl] for r in runs])
        point = block(Ps, Pc, idx_all)
        ci = {}
        for name, boots in (("ci95_address", boot_a), ("ci95_profile", boot_p)):
            bs = [block(Ps, Pc, bi) for bi in boots]
            ci[name] = {k: [float(np.nanpercentile([b[k] for b in bs], 2.5)),
                            float(np.nanpercentile([b[k] for b in bs], 97.5))] for k in KEYS}
        out["detectors"][m] = {"point": point, **ci}
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    feat_path, tag = sys.argv[1], sys.argv[2]
    list_path = sys.argv[3] if len(sys.argv) > 3 else None
    n_boot = 200 if "--quick" in sys.argv else oe.N_BOOT
    seeds = oe.SEEDS[:1] if "--quick" in sys.argv else oe.SEEDS
    corpus = dc.load_data()
    sanc = pd.read_csv(feat_path)
    for c in dc.NUMERIC_COLS:
        sanc[c] = pd.to_numeric(sanc[c], errors="coerce")
    if list_path:
        listed = [a.strip() for a in open(list_path).read().splitlines() if a.strip()]
        valid = {a for a in listed if a.startswith(("1", "3")) or a.lower().startswith("bc1")}
        sanc = sanc[sanc["address"].astype(str).isin(valid)]
    sanc = sanc.drop_duplicates("address")
    sanc = sanc[~sanc["address"].astype(str).isin(set(corpus["address"].astype(str)))]
    sanc = sanc[sanc["n_tx"] > 0].reset_index(drop=True)

    prog = json.load(open(dc.OUT_DIR / "ofac_programs.json"))["addresses"]
    profiles = [prog[a]["profile_id"] for a in sanc["address"].astype(str)]
    groups = np.array([prog[a]["group"] for a in sanc["address"].astype(str)])
    ptype = np.array([prog[a]["party_type"] for a in sanc["address"].astype(str)])
    sizes = Counter(profiles)
    largest = sizes.most_common(1)[0]
    print(f"sanctioned scored: {len(sanc)} addresses of {len(sizes)} profiles; "
          f"largest profile holds {largest[1]}", flush=True)

    ctrl, dist, smd, st, ct = oe.match_controls(corpus, sanc)
    n_jobs = min(5, len(seeds))
    runs_cv = Parallel(n_jobs=n_jobs)(delayed(crossfit_seed)(s, corpus, sanc) for s in seeds)
    runs_in = Parallel(n_jobs=n_jobs)(delayed(insample_seed)(s, corpus, sanc) for s in seeds)

    everyone = np.ones(len(sanc), dtype=bool)
    res = {"tag": tag, "seeds": list(seeds), "folds": K_FOLDS, "n_boot": n_boot,
           "n_scored": int(len(sanc)), "n_profiles": int(len(sizes)),
           "largest_profile_addresses": int(largest[1]),
           "top5_profiles_share": float(sum(v for _, v in sizes.most_common(5)) / len(sanc)),
           "addresses_per_profile_median": float(np.median(list(sizes.values()))),
           "by_group": dict(Counter(groups)), "by_party_type": dict(Counter(ptype)),
           "profiles_by_group": {g: int(len({p for p, gg in zip(profiles, groups) if gg == g}))
                                 for g in set(groups)}}
    res["cross_fitted"] = summarize(runs_cv, ctrl, everyone, profiles, np.random.RandomState(0), n_boot)
    res["in_sample"] = summarize(runs_in, ctrl, everyone, profiles, np.random.RandomState(0), n_boot)
    # sensitivity check for the observation time: controls matched also on an empty balance
    ctrl_zb, _, smd_zb, _, _ = oe.match_controls(corpus, sanc, zero_balance=True)
    res["cross_fitted_zero_balance"] = summarize(runs_cv, ctrl_zb, everyone, profiles,
                                                 np.random.RandomState(0), n_boot)
    res["in_sample_zero_balance"] = summarize(runs_in, ctrl_zb, everyone, profiles, np.random.RandomState(0), n_boot)
    empty_c = corpus["final_balance"].values < 1e-8
    res["empty_share"] = {"sanctioned": float((sanc["final_balance"].values < 1e-8).mean()),
                          "controls": float(empty_c[ctrl].mean()),
                          "controls_zero_balance": float(empty_c[ctrl_zb].mean())}
    res["smd_matched_covariates_zero_balance"] = smd_zb

    # strata (cross-fitted scores)
    strata = {}
    for g in sorted(set(groups)):
        strata["program: " + g] = summarize(runs_cv, ctrl, groups == g, profiles, np.random.RandomState(1), n_boot)
    for t in sorted(set(ptype)):
        strata["party: " + t] = summarize(runs_cv, ctrl, ptype == t, profiles, np.random.RandomState(2), n_boot)
    not_largest = np.array([p != largest[0] for p in profiles])
    strata["without largest profile"] = summarize(runs_cv, ctrl, not_largest, profiles,
                                                  np.random.RandomState(3), n_boot)
    res["strata"] = strata

    # balance of features that were not matched on (standardised mean differences)
    fs = dc.engineer_features(sanc)
    fc = dc.engineer_features(corpus)
    resid = {}
    for f in RESIDUAL:
        s, c, m = fs[f].values, fc[f].values, fc[f].values[ctrl].ravel()
        pooled = np.sqrt(0.5 * (s.var() + c.var())) + dc.EPS
        resid[f] = {"sanctioned_mean": float(s.mean()), "controls_mean": float(m.mean()),
                    "corpus_mean": float(c.mean()),
                    "smd_vs_controls": float((s.mean() - m.mean()) / pooled),
                    "smd_vs_corpus": float((s.mean() - c.mean()) / pooled)}
    res["residual_balance"] = resid
    res["smd_matched_covariates"] = smd

    for name in ("in_sample", "cross_fitted"):
        print(f"== {name}")
        for m in DETS:
            d = res[name]["detectors"][m]
            p = d["point"]
            print(f"  {m:16s} top10 {p['top10']:.3f}  ctrl {p['top10_ctrl']:.3f}  "
                  f"matched enrichment {p['enrich10_matched']:.2f} "
                  f"address CI [{d['ci95_address']['enrich10_matched'][0]:.2f},{d['ci95_address']['enrich10_matched'][1]:.2f}] "
                  f"profile CI [{d['ci95_profile']['enrich10_matched'][0]:.2f},{d['ci95_profile']['enrich10_matched'][1]:.2f}]  "
                  f"P(outrank) {p['p_outrank_matched']:.3f}", flush=True)
    print("== strata (DRIFT, cross-fitted)")
    for k, v in strata.items():
        d = v["detectors"]["DRIFT"]
        print(f"  {k:36s} n={v['n_addresses']:3d} profiles={v['n_profiles']:2d}  top10 {d['point']['top10']:.3f} "
              f"ctrl {d['point']['top10_ctrl']:.3f}  enrichment {d['point']['enrich10_matched']:.2f} "
              f"[{d['ci95_profile']['enrich10_matched'][0]:.2f},{d['ci95_profile']['enrich10_matched'][1]:.2f}]")
    suffix = "_quick" if "--quick" in sys.argv else ""
    (dc.OUT_DIR / f"ofac_crossfit_{tag}{suffix}.json").write_text(json.dumps(res, indent=1))
    print(f"[save] {dc.OUT_DIR / f'ofac_crossfit_{tag}{suffix}.json'}", flush=True)


if __name__ == "__main__":
    main()
