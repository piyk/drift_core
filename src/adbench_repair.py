"""adbench_repair.py  |  Control that isolates record validity in experiment E14.

E14 compares generic anomaly types generated in the 21-dimensional feature
space with the same types generated on four primitive quantities. The two
versions differ in validity and also in the number of perturbed dimensions.
Here every feature-space anomaly of E14 (same seeds, same draws) is repaired:
its transaction count, active span, received value, sent fraction, script
type and unconfirmed count are read from the sampled vector, and the wallet
record is reassembled by dc.assemble, so that every derived field is
recomputed and all identities hold. The repaired record keeps the sampled
primitive values and loses only the inconsistent part of the vector.

Reported per anomaly type: ROC-AUC against the real test wallets before and
after the repair, and the share of invalid records.

Run:  python revise1/src/adbench_repair.py
"""
import json
import sys

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

import drift_core as dc
import run_experiments as rx
from run_extensions import ADB_TYPES, adbench_sample, fit_gmm, identity_violations, score_from_X

NAMES = ["OneClassSVM", "LOF", "PCA", "IsolationForest", "DeepAutoencoder", "DRIFT"]


def repair(F, cols, pool, kind):
    c = {n: i for i, n in enumerate(cols)}
    n_tx = np.clip(np.expm1(F[:, c["log_n_tx"]]), 1.0, 5e6)
    days = np.clip(np.expm1(F[:, c["log_active_days"]]), 1.0, 6500.0)
    rec = np.clip(np.expm1(F[:, c["log_total_received"]]) / 1e8, 1e-8, 2.1e7)
    sf = np.clip(F[:, c["flow_ratio"]], 0.0, 1.0)
    canon = pool["address_type"].map(dc.canonical_type)
    rep = {cat: pool["address_type"][canon == cat].iloc[0] for cat in dc.TYPE_CATEGORIES if (canon == cat).any()}
    tcols = [c[f"type_{t}"] for t in dc.TYPE_CATEGORIES]
    avail = [i for i, t in enumerate(dc.TYPE_CATEGORIES) if t in rep]
    best = np.array(avail)[np.argmax(F[:, [tcols[i] for i in avail]], axis=1)]
    types = np.array([rep[dc.TYPE_CATEGORIES[i]] for i in best], dtype=object)
    unc = (F[:, c["unconfirmed_ratio"]] * n_tx >= 0.5).astype(int)
    return dc.assemble(dict(n_tx=n_tx, days=days, rec=rec, sf=sf, type=types, unc=unc,
                            typ=np.array([f"adbfix_{kind}"] * len(F), dtype=object)))


def run_seed(seed, raw):
    torch.set_num_threads(2)
    rng = np.random.RandomState(seed)
    n = len(raw)
    idx = rng.permutation(n)
    n_fit, n_val = int(0.50 * n), int(0.15 * n)
    fit_df = raw.iloc[idx[:n_fit]].reset_index(drop=True)
    test_df = raw.iloc[idx[n_fit + n_val:]].reset_index(drop=True)
    scorer, _ = rx.fit_scorer(fit_df, seed)
    _, _, X_fit = scorer(fit_df)
    S_test, _, X_test = scorer(test_df)
    n_test = rx.n_anom_for(len(test_df), dc.CONTAMINATION)
    cols = list(dc.engineer_features(test_df.iloc[:5]).columns)
    y = np.r_[np.zeros(len(X_test)), np.ones(n_test)].astype(int)
    gm_x = fit_gmm(X_fit, seed)
    out = {"seed": seed}
    for j, kind in enumerate(ADB_TYPES):
        Xa = adbench_sample(X_fit, kind, n_test, seed * 1000 + 100 + j, gm_x)
        Sa = score_from_X(scorer, Xa)
        Fa = scorer.pre.scaler.inverse_transform(Xa)
        rec = repair(Fa, cols, fit_df, kind)
        Sr, _, Xr = scorer(rec)
        out[kind] = {
            "feature_space": {m: float(roc_auc_score(y, np.r_[S_test[m], Sa[m]])) for m in NAMES},
            "repaired": {m: float(roc_auc_score(y, np.r_[S_test[m], Sr[m]])) for m in NAMES},
            "invalid_feature_space": identity_violations(Fa, cols)["any"],
            "invalid_repaired": identity_violations(scorer.pre.scaler.inverse_transform(Xr), cols)["any"]}
    print(f"[seed {seed}] " + "  ".join(
        f"{k}: AE {out[k]['feature_space']['DeepAutoencoder']:.3f}->{out[k]['repaired']['DeepAutoencoder']:.3f}"
        for k in ADB_TYPES), flush=True)
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    raw = dc.load_data()
    per_seed = Parallel(n_jobs=5)(delayed(run_seed)(s, raw) for s in dc.SEEDS)
    agg = {"seeds": list(dc.SEEDS)}
    for kind in ADB_TYPES:
        agg[kind] = {}
        for ver in ("feature_space", "repaired"):
            agg[kind][ver] = {m: {"mean": float(np.mean([r[kind][ver][m] for r in per_seed])),
                                  "std": float(np.std([r[kind][ver][m] for r in per_seed]))} for m in NAMES}
        for k in ("invalid_feature_space", "invalid_repaired"):
            agg[kind][k] = float(np.mean([r[kind][k] for r in per_seed]))
    (dc.OUT_DIR / "adbench_repair.json").write_text(json.dumps(agg, indent=1))
    (dc.OUT_DIR / "adbench_repair_per_seed.json").write_text(json.dumps(per_seed))
    for kind in ADB_TYPES:
        a = agg[kind]
        print(kind, "invalid %.3f -> %.3f | " % (a["invalid_feature_space"], a["invalid_repaired"]) + "  ".join(
            f"{m} {a['feature_space'][m]['mean']:.3f}->{a['repaired'][m]['mean']:.3f}" for m in NAMES), flush=True)


if __name__ == "__main__":
    main()
