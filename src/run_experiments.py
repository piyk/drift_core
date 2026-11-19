"""
run_experiments.py  |  Revision 1 experiments on the injected benchmark
========================================================================
Protocol (per seed):
  * the real corpus is split at random into a fit split (50%), a
    validation split (15%) and a test split (35%);
  * preprocessing statistics and all detectors are fitted on the fit split
    only, which contains no injected wallet;
  * validation and test splits receive independently generated anomalies;
  * the fusion rule is selected on the validation split; every reported
    number comes from the untouched test split.

Experiments:
  E1  main comparison, generator G2, 5% prevalence (all detectors, all
      fusion rules, fixed budgets, per-typology recall, rank analysis,
      hard negatives)
  E1v the same fusion rules on the validation split (selection only)
  E2  alternative generators: G1 (submission), G3 (independent), H
      (held-out typologies)
  E3  prevalence sweep
  E4  noise sweep for G1 and G2, with noise-matched real negatives
  E6  feasible evasion operators and feature-space interpolation
  E8  fit split contaminated with unlabeled anomalies
  E9  transductive protocol of the original submission (G1 and G2)

Run:  python revise1/src/run_experiments.py
"""

import json
import sys
import time

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

import drift_core as dc

NOISE_LEVELS = {"tight": (0.95, 1.05), "default": (0.8, 1.25),
                "wide": (0.6, 1.7), "very_wide": (0.4, 2.5)}
FUSION = ["DRIFT", "MaxRank", "MinRank", "GeoMeanRank", "WAvg_AE0.75", "WAvg_AE0.25",
          "ZMean", "ZMax", "All5MeanRank", "Gated_tau0.5", "Gated_tau0.35",
          "Gated_tau0.2", "Gated_tau0.1"]
MAIN = ["Heuristic", "OneClassSVM", "LOF", "PCA", "IsolationForest", "DeepAutoencoder",
        "Cascade", "DRIFT", "MaxRank"]
CORE = ["Heuristic", "IsolationForest", "DeepAutoencoder", "DRIFT", "MaxRank"]
ALLN = MAIN + FUSION[2:]
GRID = [round(x, 4) for x in np.arange(0.005, 0.2001, 0.005)]


def n_anom_for(n_real, c):
    return int(round(n_real * c / (1.0 - c)))


class Scorer:
    def __init__(self, pre, suite):
        self.pre, self.suite = pre, suite

    def __call__(self, df):
        feats = dc.engineer_features(df)
        X = self.pre.transform(feats.values.astype(float))
        raw = self.suite.score(X, feats)
        r = self.suite.ranks(raw)
        scores = dict(raw)
        scores.update(dc.fusion_scores(r, self.suite.zscores(raw)))
        return scores, r, X

    def from_X(self, X):
        """IF, AE and DRIFT from a preprocessed matrix (used for the
        feature-space interpolation only)."""
        s_if = -self.suite.iforest.score_samples(X)
        s_ae, _ = dc.ae_scores(self.suite.ae, X)
        r = self.suite.ranks({"IsolationForest": s_if, "DeepAutoencoder": s_ae})
        return {"IsolationForest": s_if, "DeepAutoencoder": s_ae,
                "DRIFT": 0.5 * (r["IsolationForest"] + r["DeepAutoencoder"]),
                "MaxRank": np.maximum(r["IsolationForest"], r["DeepAutoencoder"])}


def fit_scorer(df, seed):
    feats = dc.engineer_features(df)
    F = feats.values.astype(float)
    pre = dc.Preprocessor().fit(F)
    t = time.time()
    suite = dc.Suite(seed).fit(pre.transform(F), feats)
    return Scorer(pre, suite), time.time() - t


def joint_eval(S_real, S_an, typ_an, names, typ_names, rng, budgets=dc.BUDGETS):
    n_real = len(next(iter(S_real.values())))
    y = np.r_[np.zeros(n_real), np.ones(len(typ_an))].astype(int)
    typ = np.r_[np.array(["real"] * n_real, dtype=object), np.asarray(typ_an, dtype=object)]
    return {m: dc.evaluate(np.r_[S_real[m], S_an[m]], y, typ, typ_names, budgets, rng)
            for m in names}


def budget_curves(S_real, S_an, typ_an, names, typ_names, rng):
    n_real = len(next(iter(S_real.values())))
    y = np.r_[np.zeros(n_real), np.ones(len(typ_an))].astype(int)
    typ = np.r_[np.array(["real"] * n_real, dtype=object), np.asarray(typ_an, dtype=object)]
    out = {}
    for m in names:
        s = np.r_[S_real[m], S_an[m]]
        order = np.lexsort((rng.random_sample(len(s)), -s))
        ks = [max(int(round(b * len(s))), 1) for b in GRID]
        cum = np.cumsum(y[order])
        d = {"recall": [float(cum[k - 1] / y.sum()) for k in ks],
             "precision": [float(cum[k - 1] / k) for k in ks]}
        for t in typ_names:
            ct = np.cumsum((typ[order] == t).astype(int))
            d[t] = [float(ct[k - 1] / max((typ == t).sum(), 1)) for k in ks]
        out[m] = d
    return out


def run_seed(seed, raw):
    torch.set_num_threads(2)
    t_start = time.time()
    rng = np.random.RandomState(seed)
    n = len(raw)
    idx = rng.permutation(n)
    n_fit, n_val = int(0.50 * n), int(0.15 * n)
    fit_df = raw.iloc[idx[:n_fit]].reset_index(drop=True)
    val_df = raw.iloc[idx[n_fit:n_fit + n_val]].reset_index(drop=True)
    test_df = raw.iloc[idx[n_fit + n_val:]].reset_index(drop=True)
    stream = lambda k: np.random.RandomState(seed * 1000 + k)   # independent draws
    erng = np.random.RandomState(seed + 7)                      # tie breaking

    scorer, fit_seconds = fit_scorer(fit_df, seed)
    t = time.time()
    S_val, _, _ = scorer(val_df)
    S_test, R_test, X_test = scorer(test_df)
    score_seconds_per_10k = (time.time() - t) / (len(val_df) + len(test_df)) * 1e4

    res = {"seed": seed,
           "sizes": {"fit": len(fit_df), "val": len(val_df), "test": len(test_df)},
           "fit_seconds": fit_seconds, "score_seconds_per_10k": score_seconds_per_10k}
    n_test = n_anom_for(len(test_df), dc.CONTAMINATION)
    n_valn = n_anom_for(len(val_df), dc.CONTAMINATION)
    res["sizes"]["test_anomalies"] = n_test
    res["sizes"]["val_anomalies"] = n_valn

    # ---------------- E1: main comparison (G2, test split) ----------------
    # prim_life: the G2 draws with the span over the whole activity (G2L, audit only);
    # prim: the same draws measured as the scraper measures real wallets (G2, primary)
    prim_life = dc.prims_g2(n_test, stream(1), fit_df, window=False)
    prim = dc.observe(prim_life)
    an = dc.assemble(prim)
    S_an, R_an, X_an = scorer(an)
    typ_an = an["__label_typology"].values
    res["E1_main"] = joint_eval(S_test, S_an, typ_an, ALLN, dc.TYPOLOGIES, erng)
    res["E1_curves"] = budget_curves(S_test, S_an, typ_an, CORE, dc.TYPOLOGIES, erng)

    # rank analysis: where each typology sits in each component ranking
    ra = {}
    drift_real = S_test["DRIFT"]
    for tname in dc.TYPOLOGIES:
        m = typ_an == tname
        a, f = R_an["DeepAutoencoder"][m], R_an["IsolationForest"][m]
        ra[tname] = {"mean_r_ae": float(a.mean()), "mean_r_if": float(f.mean()),
                     "median_r_ae": float(np.median(a)), "median_r_if": float(np.median(f)),
                     "share_ae_top5": float((a >= 0.95).mean()),
                     "share_if_top5": float((f >= 0.95).mean()),
                     "share_both_top5": float(((a >= 0.95) & (f >= 0.95)).mean()),
                     "share_ae_only_top5": float(((a >= 0.95) & (f < 0.95)).mean()),
                     "share_if_only_top5": float(((a < 0.95) & (f >= 0.95)).mean())}
    res["E1_rank_analysis"] = ra
    res["E1_component_corr"] = {
        "spearman_real": float(pd.Series(R_test["DeepAutoencoder"]).corr(
            pd.Series(R_test["IsolationForest"]), method="spearman")),
        "spearman_ae_heur_real": float(pd.Series(R_test["DeepAutoencoder"]).corr(
            pd.Series(R_test["Heuristic"]), method="spearman"))}

    # hard negatives: only high-activity real wallets as the negative class
    hn = {}
    for label, thr in (("ntx_ge_100", 100), ("ntx_ge_1000", 1000)):
        mask = test_df["n_tx"].values >= thr
        y = np.r_[np.zeros(int(mask.sum())), np.ones(len(an))]
        hn[label] = {"n_negatives": int(mask.sum()), "base_rate": float(mask.mean())}
        for mname in MAIN:
            hn[label][mname] = float(roc_auc_score(y, np.r_[S_test[mname][mask], S_an[mname]]))
    # which real wallets consume the 5% budget
    fp = {}
    high = test_df["n_tx"].values >= 100
    for mname in CORE:
        s = np.r_[S_test[mname], S_an[mname]]
        k = int(round(0.05 * len(s)))
        top = np.argsort(-s)[:k]
        top_real = top[top < len(test_df)]
        fp[mname] = {"false_positives": int(len(top_real)),
                     "share_high_activity": float(high[top_real].mean()) if len(top_real) else 0.0}
    hn["fp_composition_at_5pct"] = fp
    res["E1_hard_negatives"] = hn

    # ---------------- E1v: fusion rules on the validation split ----------------
    anv = dc.assemble(dc.prims_g2(n_valn, stream(2), fit_df))
    S_anv, _, _ = scorer(anv)
    res["E1v_val"] = joint_eval(S_val, S_anv, anv["__label_typology"].values,
                                ["IsolationForest", "DeepAutoencoder", "Cascade"] + FUSION,
                                dc.TYPOLOGIES, erng)

    # ---------------- E2: alternative generators ----------------
    g1 = dc.gen_g1(n_test, stream(3))
    S_g1, _, _ = scorer(g1)
    res["E2_G1"] = joint_eval(S_test, S_g1, g1["__label_typology"].values, ALLN, dc.TYPOLOGIES, erng)
    g3 = dc.assemble(dc.prims_g3(n_test, stream(4), fit_df))
    S_g3, _, _ = scorer(g3)
    res["E2_G3"] = joint_eval(S_test, S_g3, g3["__label_typology"].values, ALLN, dc.TYPOLOGIES, erng)
    gh = dc.assemble(dc.prims_heldout(n_test, stream(5), fit_df))
    S_gh, _, _ = scorer(gh)
    res["E2_H"] = joint_eval(S_test, S_gh, gh["__label_typology"].values, ALLN, dc.HELDOUT, erng)

    # ---------------- E3: prevalence sweep (G2) ----------------
    res["E3_prevalence"] = {}
    for j, c in enumerate([0.005, 0.01, 0.02, 0.05, 0.10]):
        a = dc.assemble(dc.prims_g2(n_anom_for(len(test_df), c), stream(10 + j), fit_df))
        S_a, _, _ = scorer(a)
        res["E3_prevalence"][str(c)] = joint_eval(S_test, S_a, a["__label_typology"].values,
                                                  CORE, dc.TYPOLOGIES, erng)

    # ---------------- E4: noise sweep and noise-matched negatives ----------------
    res["E4_noise"] = {}
    for j, (lname, nz) in enumerate(NOISE_LEVELS.items()):
        d = {}
        a1 = dc.gen_g1(n_test, stream(20 + j), noise=nz)
        S_a1, _, _ = scorer(a1)
        d["G1"] = joint_eval(S_test, S_a1, a1["__label_typology"].values, CORE, dc.TYPOLOGIES, erng)
        a2 = dc.assemble(dc.prims_g2(n_test, stream(30 + j), fit_df, noise=nz))
        S_a2, _, _ = scorer(a2)
        d["G2"] = joint_eval(S_test, S_a2, a2["__label_typology"].values, CORE, dc.TYPOLOGIES, erng)
        # control: the same noise applied to real (negative) wallets
        for cname, fn in (("ctrl_G1_noise", dc.perturb_real_g1),
                          ("ctrl_consistent_noise", dc.perturb_real_consistent)):
            S_p, _, _ = scorer(fn(test_df, stream(40 + j), noise=nz))
            y = np.r_[np.zeros(len(test_df)), np.ones(len(test_df))]
            d[cname] = {}
            for mname in CORE:
                thr = np.quantile(S_test[mname], 0.95)
                d[cname][mname] = {
                    "auc_perturbed_vs_original": float(roc_auc_score(y, np.r_[S_test[mname], S_p[mname]])),
                    "share_above_real_p95": float((S_p[mname] > thr).mean())}
        res["E4_noise"][lname] = d

    # ---------------- E6: evasion ----------------
    ev = {}
    ops = {"dilate_3": lambda p: dc.op_dilate(p, 3), "dilate_10": lambda p: dc.op_dilate(p, 10),
           "dilate_30": lambda p: dc.op_dilate(p, 30),
           "retain_0.1": lambda p: dc.op_retain(p, 0.1), "retain_0.3": lambda p: dc.op_retain(p, 0.3),
           "retain_0.5": lambda p: dc.op_retain(p, 0.5),
           "split_3": lambda p: dc.op_split(p, 3), "split_10": lambda p: dc.op_split(p, 10),
           "split_30": lambda p: dc.op_split(p, 30),
           "combined": lambda p: dc.op_split(dc.op_retain(dc.op_dilate(p, 10), 0.3), 10)}
    ev["none"] = {m: res["E1_main"][m] for m in CORE}
    for oname, op in ops.items():
        # the operations change the behaviour (span, count, value); the scraper then observes it
        a = dc.assemble(dc.observe(op(prim_life)))
        S_a, _, _ = scorer(a)
        ev[oname] = joint_eval(S_test, S_a, a["__label_typology"].values, CORE, dc.TYPOLOGIES, erng)
    res["E6_evasion"] = ev
    # feature-space interpolation of the submission (sensitivity analysis only)
    brng = stream(50)
    real_X = {"IsolationForest": S_test["IsolationForest"], "DeepAutoencoder": S_test["DeepAutoencoder"],
              "DRIFT": S_test["DRIFT"], "MaxRank": S_test["MaxRank"]}
    res["E6_interp"] = {}
    for h in (0.0, 0.35, 0.6):
        partners = brng.choice(len(X_test), size=len(X_an))
        Xb = (1.0 - h) * X_an + h * X_test[partners]
        res["E6_interp"][str(h)] = joint_eval(real_X, scorer.from_X(Xb), typ_an,
                                              ["IsolationForest", "DeepAutoencoder", "DRIFT"],
                                              dc.TYPOLOGIES, erng)

    # ---------------- E8: contaminated fit split ----------------
    # The fit split itself contains unlabeled anomalies (independent draws);
    # the test wallets and test anomalies are those of E1.
    res["E8_contaminated_fit"] = {}
    for j, c in enumerate([0.005, 0.01, 0.02, 0.05]):
        cont = dc.assemble(dc.prims_g2(n_anom_for(len(fit_df), c), stream(60 + j), fit_df))
        fit_c = pd.concat([fit_df, cont.drop(columns="__label_typology")], ignore_index=True, sort=False)
        fit_c = fit_c.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        scorer_c, _ = fit_scorer(fit_c, seed)
        S_test_c, _, _ = scorer_c(test_df)
        S_an_c, _, _ = scorer_c(an)
        res["E8_contaminated_fit"][str(c)] = joint_eval(S_test_c, S_an_c, typ_an,
                                                        MAIN + ["MinRank"], dc.TYPOLOGIES, erng)

    # ---------------- E2_G2L: the G2 draws with the span over the whole activity ----------------
    g2l = dc.assemble(prim_life)
    S_g2l, _, _ = scorer(g2l)
    res["E2_G2L"] = joint_eval(S_test, S_g2l, g2l["__label_typology"].values, ALLN, dc.TYPOLOGIES, erng)

    res["seconds"] = time.time() - t_start
    print(f"[seed {seed}] done in {res['seconds']:.0f}s  "
          f"DRIFT roc={res['E1_main']['DRIFT']['roc_auc']:.4f} "
          f"pr={res['E1_main']['DRIFT']['pr_auc']:.4f}", flush=True)
    return res


def run_transductive(seed, raw, gen_name):
    """Protocol of the original submission: the injected wallets are part
    of the sample on which preprocessing and detectors are fitted, and the
    same sample is scored."""
    torch.set_num_threads(2)
    rng = np.random.RandomState(seed)
    n_anom = int(len(raw) * dc.CONTAMINATION / (1 - dc.CONTAMINATION))
    if gen_name == "G1":
        syn = dc.gen_g1(n_anom, rng)
    else:
        syn = dc.assemble(dc.prims_g2(n_anom, rng, raw, window=(gen_name == "G2")))
    real = raw.copy()
    real["__label_typology"] = "real"
    comb = pd.concat([real, syn], ignore_index=True, sort=False)
    comb = comb.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    typ = comb["__label_typology"].values
    y = (typ != "real").astype(int)
    feats = dc.engineer_features(comb)
    F = feats.values.astype(float)
    pre = dc.Preprocessor().fit(F)
    suite = dc.Suite(seed).fit(pre.transform(F), feats)
    raw_s = suite.ref_scores
    scores = dict(raw_s)
    scores.update(dc.fusion_scores(suite.ranks(raw_s), suite.zscores(raw_s)))
    erng = np.random.RandomState(seed + 7)
    out = {m: dc.evaluate(scores[m], y, typ, dc.TYPOLOGIES, dc.BUDGETS, erng)
           for m in MAIN + FUSION[2:]}
    print(f"[transductive {gen_name} seed {seed}] DRIFT roc={out['DRIFT']['roc_auc']:.4f} "
          f"pr={out['DRIFT']['pr_auc']:.4f}", flush=True)
    return out


def aggregate(dicts):
    """Mean and standard deviation of every numeric leaf across seeds."""
    first = dicts[0]
    if isinstance(first, dict):
        return {k: aggregate([d[k] for d in dicts]) for k in first}
    if isinstance(first, list):
        arr = np.array(dicts, dtype=float)
        return {"mean": arr.mean(axis=0).tolist(), "std": arr.std(axis=0).tolist()}
    arr = np.array(dicts, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "n": int(len(arr))}


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    quick = "--quick" in sys.argv
    raw = dc.load_data()
    print(f"corpus: {len(raw):,} wallets", flush=True)
    seeds = dc.SEEDS[:1] if quick else dc.SEEDS
    per_seed = Parallel(n_jobs=1 if quick else 5)(delayed(run_seed)(s, raw) for s in seeds)
    agg = aggregate([{k: v for k, v in r.items() if k != "seed"} for r in per_seed])
    agg["seeds"] = seeds
    agg["budget_grid"] = GRID
    tag = "_quick" if quick else ""
    (dc.OUT_DIR / f"experiments_per_seed{tag}.json").write_text(json.dumps(per_seed))
    (dc.OUT_DIR / f"experiments{tag}.json").write_text(json.dumps(agg, indent=1))
    print(f"[save] {dc.OUT_DIR / f'experiments{tag}.json'}", flush=True)
    if quick:
        return
    tseeds = dc.SEEDS[:5]
    trans = {}
    for g in ("G1", "G2L", "G2"):
        runs = Parallel(n_jobs=5)(delayed(run_transductive)(s, raw, g) for s in tseeds)
        trans[g] = aggregate(runs)
        trans[g + "_per_seed"] = runs
    trans["seeds"] = tseeds
    (dc.OUT_DIR / "transductive.json").write_text(json.dumps(trans, indent=1))
    print(f"[save] {dc.OUT_DIR / 'transductive.json'}", flush=True)


if __name__ == "__main__":
    main()
