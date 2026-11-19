"""
run_g2w.py  |  The observation model of the injected benchmark
================================================================
The scraper measures the span of a real wallet over its last 50 confirmed transactions
(Section III-B of the manuscript). G2, the primary benchmark, measures synthetic
wallets in the same way. G2L uses exactly the same draws (same seeds and anomaly stream
as run_experiments.py, E1) with the span over the whole activity of the wallet, as the
first draft of the corrected generator did. All detectors and aggregation rules are
scored on both, with the same fitted detectors, so that every difference is due to the
observation model.

Run:  python revise1/src/run_g2w.py   (writes revise1/outputs/variants/g2w*.json)
"""
import json
import sys
import time

import numpy as np
import torch
from joblib import Parallel, delayed

import drift_core as dc
import run_experiments as rx
import run_extensions as rex

OUT = dc.OUT_DIR / "variants"
NAMES = rx.ALLN + ["VolumeReceived"]


def run_seed(seed, raw):
    torch.set_num_threads(2)
    t0 = time.time()
    rng = np.random.RandomState(seed)
    n = len(raw)
    idx = rng.permutation(n)
    n_fit, n_val = int(0.50 * n), int(0.15 * n)
    fit_df = raw.iloc[idx[:n_fit]].reset_index(drop=True)
    test_df = raw.iloc[idx[n_fit + n_val:]].reset_index(drop=True)
    stream = lambda k: np.random.RandomState(seed * 1000 + k)
    ev = lambda: np.random.RandomState(seed + 7)
    n_test = rx.n_anom_for(len(test_df), dc.CONTAMINATION)

    scorer, _ = rx.fit_scorer(fit_df, seed)
    S_test, _, _ = scorer(test_df)
    S_test["VolumeReceived"] = dc.volume_scores(dc.engineer_features(test_df))["VolumeReceived"]
    p2l = dc.prims_g2(n_test, stream(1), fit_df, window=False)
    p2 = dc.observe(p2l)
    res = {"seed": seed}
    cov_test = rex.covariates(test_df)
    for g, prim in (("G2L", p2l), ("G2", p2)):
        an = dc.assemble(prim)
        S_an, R_an, _ = scorer(an)
        S_an["VolumeReceived"] = dc.volume_scores(dc.engineer_features(an))["VolumeReceived"]
        typ = an["__label_typology"].values
        e = rx.joint_eval(S_test, S_an, typ, NAMES, dc.TYPOLOGIES, ev())
        res[g] = {"eval": e,
                  "matched_auc": rex.matched_auc(S_test, S_an, cov_test, rex.covariates(an),
                                                 ["Heuristic", "IsolationForest", "DeepAutoencoder", "DRIFT",
                                                  "VolumeReceived"]),
                  "median_rank": {t: {m: float(np.median(R_an[m][typ == t]))
                                      for m in ("IsolationForest", "DeepAutoencoder")} for t in dc.TYPOLOGIES},
                  "median_active_days": {t: float(np.median(an["active_days"].values[typ == t]))
                                         for t in dc.TYPOLOGIES}}
    res["seconds"] = time.time() - t0
    print(f"[g2w seed {seed}] {res['seconds']:.0f}s  DRIFT PR-AUC G2L {res['G2L']['eval']['DRIFT']['pr_auc']:.4f} "
          f"G2 {res['G2']['eval']['DRIFT']['pr_auc']:.4f}", flush=True)
    return res


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    OUT.mkdir(parents=True, exist_ok=True)
    raw = dc.load_data()
    per_seed = Parallel(n_jobs=5)(delayed(run_seed)(s, raw) for s in dc.SEEDS)
    (OUT / "g2w_per_seed.json").write_text(json.dumps(per_seed))
    agg = rx.aggregate([{k: v for k, v in r.items() if k != "seed"} for r in per_seed])
    agg["seeds"] = dc.SEEDS
    (OUT / "g2w.json").write_text(json.dumps(agg, indent=1))
    print(f"[save] {OUT / 'g2w.json'}", flush=True)


if __name__ == "__main__":
    main()
