"""separability.py  |  Supervised reference for the injected benchmarks.

A random forest is trained on the labeled validation split and evaluated on
the test split of the same seed (the splits and the injected wallets are
those of run_experiments.py). This bounds what a label-free detector can
reach and shows how much of the separability of G1 came from its artefacts.
The best single feature, with feature and direction selected on the
validation labels, is reported as a second reference.

Run:  python revise1/src/separability.py
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

import drift_core as dc
from run_experiments import n_anom_for

SEEDS = dc.SEEDS


def run(seed, raw):
    rng = np.random.RandomState(seed)
    n = len(raw)
    idx = rng.permutation(n)
    n_fit, n_val = int(0.50 * n), int(0.15 * n)
    fit_df = raw.iloc[idx[:n_fit]].reset_index(drop=True)
    val_df = raw.iloc[idx[n_fit:n_fit + n_val]].reset_index(drop=True)
    test_df = raw.iloc[idx[n_fit + n_val:]].reset_index(drop=True)
    stream = lambda k: np.random.RandomState(seed * 1000 + k)
    pre = dc.Preprocessor().fit(dc.engineer_features(fit_df).values.astype(float))
    X = lambda df: pre.transform(dc.engineer_features(df).values.astype(float))
    Xv, Xt = X(val_df), X(test_df)
    nv, nt = n_anom_for(len(val_df), 0.05), n_anom_for(len(test_df), 0.05)
    gens = {
        "G1": (dc.gen_g1(nv, stream(102)), dc.gen_g1(nt, stream(3))),
        "G2": (dc.assemble(dc.prims_g2(nv, stream(2), fit_df)), dc.assemble(dc.prims_g2(nt, stream(1), fit_df))),
        "G3": (dc.assemble(dc.prims_g3(nv, stream(104), fit_df)), dc.assemble(dc.prims_g3(nt, stream(4), fit_df))),
        "H": (dc.assemble(dc.prims_heldout(nv, stream(105), fit_df)), dc.assemble(dc.prims_heldout(nt, stream(5), fit_df))),
    }
    out = {}
    for g, (av, at) in gens.items():
        Xtr = np.r_[Xv, X(av)]
        ytr = np.r_[np.zeros(len(Xv)), np.ones(len(av))]
        Xte = np.r_[Xt, X(at)]
        yte = np.r_[np.zeros(len(Xt)), np.ones(len(at))]
        rf = RandomForestClassifier(n_estimators=200, n_jobs=4, random_state=seed).fit(Xtr, ytr)
        p = rf.predict_proba(Xte)[:, 1]
        # best single feature, chosen on validation labels
        best_j, best_sign, best_auc = 0, 1.0, 0.0
        for j in range(Xtr.shape[1]):
            a = roc_auc_score(ytr, Xtr[:, j])
            if max(a, 1 - a) > best_auc:
                best_j, best_sign, best_auc = j, (1.0 if a >= 0.5 else -1.0), max(a, 1 - a)
        out[g] = {"rf_roc_auc": float(roc_auc_score(yte, p)),
                  "rf_pr_auc": float(average_precision_score(yte, p)),
                  "single_feature_roc_auc": float(roc_auc_score(yte, best_sign * Xte[:, best_j])),
                  "single_feature_index": int(best_j)}
    print(seed, {g: (round(v["rf_roc_auc"], 4), round(v["rf_pr_auc"], 3),
                     round(v["single_feature_roc_auc"], 3), v["single_feature_index"])
                 for g, v in out.items()}, flush=True)
    return out


if __name__ == "__main__":
    raw = dc.load_data()
    runs = [run(s, raw) for s in SEEDS]
    names = list(dc.engineer_features(raw.head(5)).columns)
    agg = {}
    for g in runs[0]:
        agg[g] = {k: {"mean": float(np.mean([r[g][k] for r in runs])),
                      "std": float(np.std([r[g][k] for r in runs]))}
                  for k in ("rf_roc_auc", "rf_pr_auc", "single_feature_roc_auc")}
        feats = [names[r[g]["single_feature_index"]] for r in runs]
        agg[g]["single_feature_selected"] = {f: feats.count(f) for f in set(feats)}
    (dc.OUT_DIR / "separability.json").write_text(json.dumps(agg, indent=1))
    print(json.dumps(agg, indent=1))
