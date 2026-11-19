"""
run_features.py  |  Pre-specified feature revision on the injected benchmarks
=============================================================================
Specification: revise1/features_prespecification.md (frozen before this code).

For every seed and every feature set (F21, W21, B, BR, BR-u), with the split, the
anomaly streams and the detector settings of run_experiments.py:
  * all detectors, the volume ranking and the aggregation rules on G2, G3, H and H'
    (ROC-AUC, PR-AUC, budgets, recall per typology);
  * the matched AUC against the five nearest real test wallets in count, value, span;
  * hard negatives (real test wallets with >= 100 and >= 1,000 transactions) and the
    composition of the false positives at the 5% budget;
  * the noise-matched negative control (G1 and identity-preserving noise paths);
  * the validity control (ADBench types in the feature space of the set, and the same
    samples reassembled into valid records);
  * the share of unseen real wallets above the 95th percentile of the reference.
Criteria S1, S3 and S4 of the specification are evaluated at the end (S2 is evaluated
by features_external.py).

Run:  python revise1/src/run_features.py --check     (F21 reproduction, smoke test)
      python revise1/src/run_features.py [--jobs N]  (full run, executed once)
"""

import json
import sys
import time

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

import drift_core as dc
import features_v2 as fv
import run_experiments as rx
import run_extensions as rex

OUT = dc.OUT_DIR / "features"
PARTS = OUT / "parts"
DETS = ["Heuristic", "OneClassSVM", "LOF", "PCA", "IsolationForest", "DeepAutoencoder",
        "ECOD", "COPOD", "DIF", "DeepSVDD"]
RULES = ["DRIFT", "MinRank", "MaxRank", "All5MeanRank", "GeoMeanRank", "Gated_tau0.35", "ZMean", "Cascade"]
VOL = ["VolumeReceived"]
CTRL = ["Heuristic", "OneClassSVM", "LOF", "PCA", "IsolationForest", "DeepAutoencoder", "DRIFT", "MinRank"]
FPN = ["Heuristic", "IsolationForest", "DeepAutoencoder", "DRIFT", "VolumeReceived"]
S4_FAMILY = ["OneClassSVM", "LOF", "PCA", "IsolationForest", "DeepAutoencoder", "ECOD", "COPOD", "DIF", "DeepSVDD"]
BENCH = ["G2", "G3", "H", "Hp"]
TNAMES = {"G2": dc.TYPOLOGIES, "G3": dc.TYPOLOGIES, "H": dc.HELDOUT, "Hp": fv.HPRIME}
NAMES14 = ["OneClassSVM", "LOF", "PCA", "IsolationForest", "DeepAutoencoder", "DRIFT"]


def names_for(name):
    return DETS + RULES + VOL + (["DRIFT-S"] if name == "BR" else [])


def slim(ev):
    return {m: {k: v for k, v in d.items()
                if k in ("roc_auc", "pr_auc", "oracle_f1") or k.startswith(("P@", "R@0.05", "R@0.1"))}
            for m, d in ev.items()}


def split(raw, seed):
    rng = np.random.RandomState(seed)
    n = len(raw)
    idx = rng.permutation(n)
    n_fit, n_val = int(0.50 * n), int(0.15 * n)
    return (raw.iloc[idx[:n_fit]].reset_index(drop=True),
            raw.iloc[idx[n_fit + n_val:]].reset_index(drop=True))


def benchmarks(seed, fit_df, n_test):
    stream = lambda k: np.random.RandomState(seed * 1000 + k)
    return {"G2": dc.assemble(dc.prims_g2(n_test, stream(1), fit_df)),
            "G3": dc.assemble(dc.prims_g3(n_test, stream(4), fit_df)),
            "H": dc.assemble(dc.prims_heldout(n_test, stream(5), fit_df)),
            "Hp": dc.assemble(fv.prims_hprime(n_test, stream(90), fit_df))}


def score_all(sc, pyod, df):
    S, _, X = sc(df)
    S["VolumeReceived"] = dc.volume_scores(dc.engineer_features(df))["VolumeReceived"]
    for p, clf in pyod.items():
        S[p] = np.asarray(clf.decision_function(X), dtype=float)
    if sc.name == "BR":
        S["DRIFT-S"] = sc.drift_s(S, fv.canonical_types(df))
    return S, X


# ----------------------------------------------------------------------
# one job: one seed, one feature set
# ----------------------------------------------------------------------
def run_job(seed, name, raw, with_pyod=True):
    torch.set_num_threads(2)
    t0 = time.time()
    fit_df, test_df = split(raw, seed)
    stream = lambda k: np.random.RandomState(seed * 1000 + k)
    ev = lambda: np.random.RandomState(seed + 7)
    n_test = rx.n_anom_for(len(test_df), dc.CONTAMINATION)
    gens = benchmarks(seed, fit_df, n_test)
    names = names_for(name) if with_pyod else [m for m in names_for(name) if m not in ("ECOD", "COPOD", "DIF", "DeepSVDD")]

    sc = fv.fit_scorer(fit_df, seed, name)
    X_fit = sc.matrix(fit_df)
    t_pyod = time.time()
    pyod = fv.fit_pyod(X_fit, seed) if with_pyod else {}
    t_pyod = time.time() - t_pyod
    S_test, X_test = score_all(sc, pyod, test_df)
    S_gen = {g: score_all(sc, pyod, df)[0] for g, df in gens.items()}
    res = {"seed": seed, "set": name, "d": int(X_fit.shape[1]), "k_pca": sc.suite.kp, "k_ae": sc.suite.ka,
           "ae_final_loss": float(sc.suite.ae_hist[-1]), "pyod_seconds": t_pyod,
           "sizes": {"fit": len(fit_df), "test": len(test_df), "test_anomalies": n_test}}
    if name in ("BR", "BR-u"):
        res["count_bins"] = len(sc.fmap.edges)
        res["min_bin_size"] = int(min(sc.fmap.bin_sizes))

    # detection, matched AUC, hard negatives, false-positive composition
    cov_test = rex.covariates(test_df)
    high = test_df["n_tx"].values >= 100
    res["main"], res["matched_auc"], res["hard_negatives"], res["fp_at_5pct"] = {}, {}, {}, {}
    for g, df in gens.items():
        typ = df["__label_typology"].values
        res["main"][g] = slim(rx.joint_eval(S_test, S_gen[g], typ, names, TNAMES[g], ev()))
        res["matched_auc"][g] = rex.matched_auc(S_test, S_gen[g], cov_test, rex.covariates(df), names)
        hn = {}
        for label, thr in (("ntx_ge_100", 100), ("ntx_ge_1000", 1000)):
            mask = test_df["n_tx"].values >= thr
            y = np.r_[np.zeros(int(mask.sum())), np.ones(len(df))]
            hn[label] = {m: float(roc_auc_score(y, np.r_[S_test[m][mask], S_gen[g][m]])) for m in names}
        res["hard_negatives"][g] = hn
        fp = {}
        for m in FPN:
            s = np.r_[S_test[m], S_gen[g][m]]
            top = np.argsort(-s, kind="stable")[:int(round(0.05 * len(s)))]
            top_real = top[top < len(test_df)]
            fp[m] = float(high[top_real].mean()) if len(top_real) else 0.0
        res["fp_at_5pct"][g] = fp

    # noise-matched negative control (default width, stream of run_experiments.py)
    ctrl = {}
    for cname, fn in (("ctrl_G1_noise", dc.perturb_real_g1), ("ctrl_consistent_noise", dc.perturb_real_consistent)):
        S_p, _, _ = sc(fn(test_df, stream(41), noise=(0.8, 1.25)))
        y = np.r_[np.zeros(len(test_df)), np.ones(len(test_df))]
        ctrl[cname] = {m: float(roc_auc_score(y, np.r_[S_test[m], S_p[m]])) for m in CTRL}
    res["noise_control"] = ctrl

    # calibration of the rank threshold for unseen real wallets
    ref = sc.reference()
    res["exceedance_p95"] = {m: float((S_test[m] > np.quantile(ref[m], 0.95)).mean())
                             for m in ("IsolationForest", "DeepAutoencoder", "DRIFT", "MinRank")}

    # validity control: ADBench types in the feature space of the set, and repaired
    gm = rex.fit_gmm(X_fit, seed)
    val = {"violations_real": fv.identity_violations(sc.pre.natural(X_test), sc),
           "gmm_components": int(gm.n_components)}
    y14 = np.r_[np.zeros(len(X_test)), np.ones(n_test)].astype(int)
    for j, kind in enumerate(rex.ADB_TYPES):
        Xa = rex.adbench_sample(X_fit, kind, n_test, seed * 1000 + 100 + j, gm)
        Sa = rex.score_from_X(sc, Xa)
        Fa = sc.pre.natural(Xa)
        rec = fv.repair(Fa, sc, fit_df, kind, stream(130 + j))
        Sr, _, Xr = sc(rec)
        val[kind] = {"feature_space": {m: float(roc_auc_score(y14, np.r_[S_test[m], Sa[m]])) for m in NAMES14},
                     "repaired": {m: float(roc_auc_score(y14, np.r_[S_test[m], Sr[m]])) for m in NAMES14},
                     "invalid_feature_space": fv.identity_violations(Fa, sc)["any"],
                     "invalid_repaired": fv.identity_violations(sc.pre.natural(Xr), sc)["any"]}
    res["validity"] = val
    res["seconds"] = time.time() - t0
    return res


def job_file(seed, name):
    return PARTS / f"{name}_{seed}.json"


def run_and_save(seed, name, raw):
    f = job_file(seed, name)
    if f.exists():
        return
    res = run_job(seed, name, raw)
    res["spec_sha256"] = fv.spec_digest()
    f.write_text(json.dumps(res))
    print(f"[features {name} seed {seed}] {res['seconds']:.0f}s", flush=True)


# ----------------------------------------------------------------------
# pre-specified criteria S1, S3, S4
# ----------------------------------------------------------------------
def paired(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = a - b
    p = float("nan") if np.allclose(d, 0) else float(wilcoxon(a, b).pvalue)
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()), "mean_diff": float(d.mean()),
            "wins": int((d > 0).sum()), "p": p}


def holm(ps):
    """Holm-adjusted P values in the order given."""
    ps = np.asarray(ps, dtype=float)
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[k]))
        adj[k] = run
    return adj.tolist()


def criteria(per):
    """per[set] is the list of per-seed results in seed order."""
    br, f21 = per["BR"], per["F21"]
    out = {}
    s1 = {g: paired([r["matched_auc"][g]["DRIFT"] for r in br], [r["matched_auc"][g]["DRIFT"] for r in f21])
          for g in ("G3", "H", "Hp")}
    adj = holm([s1[g]["p"] for g in ("G3", "H", "Hp")])
    for g, a in zip(("G3", "H", "Hp"), adj):
        s1[g]["p_holm"] = a
        s1[g]["holds"] = bool(s1[g]["mean_diff"] >= 0.05 and a < 0.05)
    s1["holds"] = all(s1[g]["holds"] for g in ("G3", "H", "Hp"))
    out["S1"] = s1
    s3 = paired([r["hard_negatives"]["G2"]["ntx_ge_100"]["DRIFT"] for r in br],
                [r["hard_negatives"]["G2"]["ntx_ge_100"]["DRIFT"] for r in f21])
    s3["holds"] = bool(s3["mean_diff"] > 0 and s3["p"] < 0.05)
    out["S3"] = s3
    s4 = {m: paired([r["main"]["Hp"]["DRIFT"]["pr_auc"] for r in br], [r["main"]["Hp"][m]["pr_auc"] for r in br])
          for m in S4_FAMILY}
    adj = holm([s4[m]["p"] for m in S4_FAMILY])
    for m, a in zip(S4_FAMILY, adj):
        s4[m]["p_holm"] = a
        s4[m]["holds"] = bool(s4[m]["mean_diff"] > 0 and a < 0.05)
    s4["holds"] = all(s4[m]["holds"] for m in S4_FAMILY)
    out["S4"] = s4
    return out


def collect():
    per = {}
    for name in fv.SETS:
        runs = [json.loads(job_file(s, name).read_text()) for s in dc.SEEDS]
        per[name] = runs
    return per


# ----------------------------------------------------------------------
# check mode (specification, section Protocol)
# ----------------------------------------------------------------------
def check(raw):
    """(1) F21 on seed 11 must reproduce the stored values of the manuscript exactly;
    (2) the revised sets run on a subsample without missing values (no metric printed)."""
    seed = dc.SEEDS[0]
    stored = json.loads((dc.OUT_DIR / "experiments_per_seed.json").read_text())
    st = next(r for r in stored if r["seed"] == seed)
    ext = json.loads((dc.OUT_DIR / "extensions_per_seed.json").read_text())
    se = next(r for r in ext if r["seed"] == seed)
    rep = json.loads((dc.OUT_DIR / "adbench_repair_per_seed.json").read_text())
    sr = next(r for r in rep if r["seed"] == seed)
    t = time.time()
    r = run_job(seed, "F21", raw)
    diffs = []
    add = lambda label, a, b: diffs.append((label, abs(a - b)))
    for g, key in (("G2", "E1_main"), ("G3", "E2_G3"), ("H", "E2_H")):
        for m in ("Heuristic", "OneClassSVM", "LOF", "PCA", "IsolationForest", "DeepAutoencoder", "DRIFT",
                  "MinRank", "All5MeanRank", "Cascade"):
            for k in ("roc_auc", "pr_auc"):
                add(f"main {g} {m} {k}", r["main"][g][m][k], st[key][m][k])
        for m in ("ECOD", "COPOD", "DIF", "DeepSVDD"):
            for k in ("roc_auc", "pr_auc"):
                add(f"pyod {g} {m} {k}", r["main"][g][m][k], se["E13_pyod"][g][m][k])
        for m in ("IsolationForest", "DeepAutoencoder", "DRIFT"):
            add(f"matched {g} {m}", r["matched_auc"][g][m], se["E12_matched"][g][m])
    for m in ("IsolationForest", "DeepAutoencoder", "DRIFT"):
        add(f"hardneg {m}", r["hard_negatives"]["G2"]["ntx_ge_100"][m], st["E1_hard_negatives"]["ntx_ge_100"][m])
        for c in ("ctrl_G1_noise", "ctrl_consistent_noise"):
            add(f"{c} {m}", r["noise_control"][c][m], st["E4_noise"]["default"][c][m]["auc_perturbed_vs_original"])
    for kind in rex.ADB_TYPES:
        for v in ("feature_space", "repaired"):
            for m in NAMES14:
                add(f"validity {kind} {v} {m}", r["validity"][kind][v][m], sr[kind][v][m])
    bad = [(lab, d) for lab, d in diffs if d > 0]
    print(f"[check] F21 seed {seed}: {len(diffs)} stored values compared, {len(bad)} differ, max abs difference "
          f"{max(d for _, d in diffs):.3g} ({time.time() - t:.0f}s)", flush=True)
    for lab, d in bad:
        print(f"[check]   differs: {lab}  {d:.3g}", flush=True)
    ok = not bad

    # smoke test of the revised sets on a 3,000-wallet subsample: shapes and finiteness only
    sub = raw.sample(n=3000, random_state=0).reset_index(drop=True)
    fit_df, test_df = sub.iloc[:2000].reset_index(drop=True), sub.iloc[2000:].reset_index(drop=True)
    gens = benchmarks(seed, fit_df, 50)
    for name in fv.SETS[1:]:
        sc = fv.fit_scorer(fit_df, seed, name)
        X_fit = sc.matrix(fit_df)
        pyod = fv.fit_pyod(X_fit, seed)
        S, X = score_all(sc, pyod, test_df)
        finite = all(np.isfinite(v).all() for v in S.values()) and np.isfinite(X).all()
        for g, df in gens.items():
            Sg, Xg = score_all(sc, pyod, df)
            finite &= all(np.isfinite(v).all() for v in Sg.values()) and np.isfinite(Xg).all()
        Fn = sc.pre.natural(X)
        iv = fv.identity_violations(Fn, sc)
        gm = rex.fit_gmm(X_fit, seed)
        for j, kind in enumerate(rex.ADB_TYPES):
            Xa = rex.adbench_sample(X_fit, kind, 30, seed * 1000 + 100 + j, gm)
            Sa = rex.score_from_X(sc, Xa)
            rec = fv.repair(sc.pre.natural(Xa), sc, fit_df, kind, np.random.RandomState(j))
            Sr, _, _ = sc(rec)
            finite &= all(np.isfinite(v).all() for v in Sa.values()) and all(np.isfinite(v).all() for v in Sr.values())
        finite &= bool(np.isfinite(iv["any"]))
        print(f"[check] {name}: d={X.shape[1]} (PCA {sc.suite.kp}, AE bottleneck {sc.suite.ka}), "
              f"score keys {len(S)}, all finite: {bool(finite)}", flush=True)
        ok &= bool(finite)
    print(f"[check] passed: {ok}", flush=True)
    return ok


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    raw = dc.load_data()
    print(f"corpus: {len(raw):,} wallets; spec sha256 {fv.spec_digest()}", flush=True)
    if "--check" in sys.argv:
        sys.exit(0 if check(raw) else 1)
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 5
    OUT.mkdir(parents=True, exist_ok=True)
    PARTS.mkdir(parents=True, exist_ok=True)
    todo = [(s, n) for n in ("BR", "F21", "B", "W21", "BR-u") for s in dc.SEEDS]   # heaviest first
    t = time.time()
    Parallel(n_jobs=jobs)(delayed(run_and_save)(s, n, raw) for s, n in todo)
    per = collect()
    agg = {name: rx.aggregate([{k: v for k, v in r.items() if k not in ("seed", "set", "spec_sha256")}
                               for r in runs]) for name, runs in per.items()}
    out = {"spec_sha256": fv.spec_digest(), "seeds": dc.SEEDS, "sets": fv.SETS,
           "software": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "numpy": np.__version__, "pandas": pd.__version__},
           "seconds": time.time() - t, "results": agg, "criteria": criteria(per)}
    (OUT / "features.json").write_text(json.dumps(out, indent=1))
    print(f"[save] {OUT / 'features.json'} ({time.time() - t:.0f}s)", flush=True)
    c = out["criteria"]
    print("criteria:", {k: v["holds"] for k, v in c.items()}, flush=True)


if __name__ == "__main__":
    main()
