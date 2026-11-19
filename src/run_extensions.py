"""
run_extensions.py  |  Further controls on the injected benchmark
=================================================================
Every seed uses the split, the detectors and the injected wallets of
run_experiments.py (same seeds, same random streams), so the results below
are paired with the stored main results.

  E10  out-of-sample rank reference: the reference distribution of each
       detector is taken from real wallets that were not used for fitting
       (the real wallets of the validation split) instead of the fit split;
       calibration of the ranks of unseen real wallets under both references
  E11  label-free volume baselines (received value, transaction count)
  E12  matched negatives: each injected wallet against the real test wallets
       nearest in transaction count, received value and active span
  E13  further detectors: ECOD, COPOD, Deep Isolation Forest, Deep SVDD
       (PyOD, default hyperparameters). PyOD estimates the tail probabilities of
       ECOD and COPOD on the fit data together with the batch that is scored, which
       would let the injected test wallets enter the model. HeldOutTail computes the
       same scores with tail probabilities from the fit split only, so that every
       wallet is scored independently of the others, as for every other detector.
  E14  external generator audit: the four anomaly types of ADBench generated
       in feature space, as published, and on the primitive quantities of a
       wallet record, so that every accounting identity holds

Run:  python revise1/src/run_extensions.py [--quick] [--no-pyod]
"""

import contextlib
import io
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

import drift_core as dc
import run_experiments as rx

warnings.filterwarnings("ignore")

BASE7 = dc.BASE + ["Cascade"]
RULES = ["DRIFT", "MinRank", "MaxRank", "GeoMeanRank", "WAvg_AE0.75", "WAvg_AE0.25",
         "All5MeanRank", "Gated_tau0.35", "ZMean"]
LEARNED = ["OneClassSVM", "LOF", "PCA", "IsolationForest", "DeepAutoencoder"]
MAIN7 = ["Heuristic"] + LEARNED + ["DRIFT"]
QS = (0.90, 0.95, 0.99)
ADB_TYPES = ["global", "local", "cluster", "dependency"]
ADB_ALPHA = {"global": 1.1, "local": 5.0, "cluster": 5.0}
PRIM_COLS = ["log_n_tx", "log_days", "log_rec", "logit_sf"]


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def covariates(df):
    """The three matching covariates of the external check."""
    return np.c_[np.log1p(np.maximum(df["n_tx"].values.astype(float), 0)),
                 np.log1p(np.maximum(df["total_received"].values.astype(float), 0) * 1e8),
                 np.log1p(np.maximum(df["active_days"].values.astype(float), 0))]


def matched_auc(S_real, S_an, cov_real, cov_an, names, k=5):
    """Probability that an injected wallet outranks one of its k matched real
    wallets (nearest in the standardised covariates), ties counted as 0.5."""
    mu, sd = cov_real.mean(0), cov_real.std(0) + dc.EPS
    nn = NearestNeighbors(n_neighbors=k).fit((cov_real - mu) / sd)
    dist, ind = nn.kneighbors((cov_an - mu) / sd)
    out = {"median_distance": float(np.median(dist)), "k": k}
    for m in names:
        sa, sc = S_an[m][:, None], S_real[m][ind]
        out[m] = float((sa > sc).mean() + 0.5 * (sa == sc).mean())
    return out


def score_from_X(scorer, X):
    """Learned detectors and their fusion from a preprocessed matrix (for
    synthetic vectors that exist in feature space only)."""
    su = scorer.suite
    raw = {"OneClassSVM": -su.ocsvm.decision_function(X),
           "LOF": -su.lof.score_samples(X),
           "PCA": np.mean((su.pca.inverse_transform(su.pca.transform(X)) - X) ** 2, axis=1),
           "IsolationForest": -su.iforest.score_samples(X)}
    err, _ = dc.ae_scores(su.ae, X)
    raw["DeepAutoencoder"] = err
    r = su.ranks(raw)
    out = dict(raw)
    out["DRIFT"] = 0.5 * (r["DeepAutoencoder"] + r["IsolationForest"])
    return out


def identity_violations(F, cols):
    """Share of records, given in the winsorised feature space, that break an
    encoding or accounting identity which a wallet record satisfies by
    construction. Tolerances allow for winsorisation and rounding."""
    c = {n: i for i, n in enumerate(cols)}
    types = F[:, [c[f"type_{t}"] for t in dc.TYPE_CATEGORIES]]
    onehot = (np.abs(types - np.round(types)).max(axis=1) > 0.05) | (np.abs(types.sum(axis=1) - 1.0) > 0.05)
    zb = F[:, c["zero_balance_flag"]]
    flag = np.abs(zb - np.round(zb)) > 0.05
    fr, rr, nf = F[:, c["flow_ratio"]], F[:, c["retention_ratio"]], F[:, c["net_flow_norm"]]
    balance = np.abs(fr + rr - 1.0) > 0.05                      # balance = received - sent
    netflow = np.abs(nf - (1.0 - fr) / (1.0 + fr)) > 0.05       # net flow follows from the flow ratio
    anyv = onehot | flag | balance | netflow
    return {"any": float(anyv.mean()), "one_hot": float(onehot.mean()), "flag": float(flag.mean()),
            "balance": float(balance.mean()), "net_flow": float(netflow.mean())}


def fit_gmm(Z, seed):
    """Gaussian mixture with the number of components (1 to 9) chosen by BIC,
    as in the ADBench generator."""
    best, best_bic = None, np.inf
    for k in range(1, 10):
        gm = GaussianMixture(n_components=k, random_state=seed).fit(Z)
        b = gm.bic(Z)
        if b < best_bic:
            best, best_bic = gm, b
    return best


def adbench_sample(Z, kind, n, seed, gm=None):
    """n synthetic anomalies of one ADBench type around the data matrix Z."""
    rng = np.random.RandomState(seed)
    if kind == "global":
        a = ADB_ALPHA["global"]                      # both bounds are multiplied by 1.1, as in ADBench
        lo, hi = a * Z.min(axis=0), a * Z.max(axis=0)
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        return rng.uniform(lo, hi, size=(n, Z.shape[1]))
    if kind == "dependency":
        out = np.zeros((n, Z.shape[1]))
        for j in range(Z.shape[1]):
            col = Z[:, j]
            if np.std(col) < 1e-9:
                out[:, j] = col[0]
                continue
            sub = col[rng.choice(len(col), size=min(5000, len(col)), replace=False)]
            try:
                out[:, j] = gaussian_kde(sub).resample(n, seed=rng)[0]
            except Exception:
                out[:, j] = rng.choice(col, size=n)
        return out
    comp = rng.choice(gm.n_components, size=n, p=gm.weights_)
    out = np.zeros((n, Z.shape[1]))
    for k in range(gm.n_components):
        m = comp == k
        if not m.any():
            continue
        mean, cov = gm.means_[k], gm.covariances_[k]
        if kind == "local":
            cov = ADB_ALPHA["local"] * cov
        else:
            mean = ADB_ALPHA["cluster"] * mean
        out[m] = rng.multivariate_normal(mean, cov, size=int(m.sum()), check_valid="ignore")
    return out


def primitives_of(df):
    """Primitive quantities of real records: count, span, received value and
    sent fraction, on scales on which a Gaussian mixture is meaningful."""
    n_tx = np.maximum(df["n_tx"].values.astype(float), 1.0)
    days = np.maximum(df["active_days"].values.astype(float), 1.0)
    rec = np.maximum(df["total_received"].values.astype(float), 1e-8)
    sf = np.clip(np.maximum(df["total_sent"].values.astype(float), 0.0) / rec, 1e-4, 1.0 - 1e-4)
    return np.c_[np.log(n_tx), np.log(days), np.log(rec), np.log(sf / (1.0 - sf))]


def records_from_primitives(P, rng, pool, kind):
    """Valid wallet records from primitive quantities: every derived field is
    recomputed by dc.assemble, so that all identities hold."""
    n_tx = np.clip(np.exp(P[:, 0]), 1.0, 5e6)
    days = np.clip(np.exp(P[:, 1]), 1.0, 6500.0)
    rec = np.clip(np.exp(P[:, 2]), 1e-8, 2.1e7)
    sf = 1.0 / (1.0 + np.exp(-np.clip(P[:, 3], -30, 30)))
    types, unc = dc._nuisance(len(P), rng, pool)
    return dc.assemble(dict(n_tx=n_tx, days=days, rec=rec, sf=sf, type=types, unc=unc,
                            typ=np.array([f"adb_{kind}"] * len(P), dtype=object)))


class HeldOutTail:
    """ECOD (Li et al., 2023) or COPOD (Li et al., 2020) with the formulas of PyOD 2.1.0,
    but with the empirical tail probabilities and the skewness estimated on the fit data
    only. A scored wallet is ranked among the fit wallets plus itself, so its score does
    not depend on the other wallets that are scored with it. reference_scores() gives
    the in-sample scores of the fit wallets, as for the other detectors."""

    def __init__(self, kind):
        self.kind = kind

    def fit(self, X):
        from pyod.utils.stat_models import column_ecdf
        from scipy.stats import skew
        X = np.asarray(X, dtype=float)
        self.sorted, self.n = np.sort(X, axis=0), len(X)
        self.sign = np.sign(np.nan_to_num(skew(X, axis=0)))
        self.ref = self._combine(-np.log(column_ecdf(X)), -np.log(column_ecdf(-X)))
        return self

    def _combine(self, ul, ur):
        s = self.sign
        uskew = ul * -np.sign(s - 1) + ur * np.sign(s + 1)
        o = np.maximum(uskew, np.maximum(ul, ur)) if self.kind == "ECOD" else np.maximum(uskew, (ul + ur) / 2)
        return o.sum(axis=1)

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        ul, ur = np.empty_like(X), np.empty_like(X)
        for j in range(X.shape[1]):
            col = self.sorted[:, j]
            ul[:, j] = -np.log((np.searchsorted(col, X[:, j], side="right") + 1) / (self.n + 1))
            ur[:, j] = -np.log((self.n - np.searchsorted(col, X[:, j], side="left") + 1) / (self.n + 1))
        return self._combine(ul, ur)

    def reference_scores(self):
        return self.ref


def pyod_models(seed):
    from pyod.models.deep_svdd import DeepSVDD
    from pyod.models.dif import DIF
    return {"ECOD": lambda: HeldOutTail("ECOD"),
            "COPOD": lambda: HeldOutTail("COPOD"),
            "DIF": lambda: DIF(random_state=seed, device="cpu"),
            "DeepSVDD": lambda: DeepSVDD(n_features=len(dc.FEATURE_NAMES) + len(dc.TYPE_CATEGORIES),
                                         batch_size=256, verbose=0, random_state=seed)}


# ----------------------------------------------------------------------
# one seed
# ----------------------------------------------------------------------
def run_seed(seed, raw, use_pyod=True):
    torch.set_num_threads(2)
    t0 = time.time()
    rng = np.random.RandomState(seed)
    n = len(raw)
    idx = rng.permutation(n)
    n_fit, n_val = int(0.50 * n), int(0.15 * n)
    fit_df = raw.iloc[idx[:n_fit]].reset_index(drop=True)
    val_df = raw.iloc[idx[n_fit:n_fit + n_val]].reset_index(drop=True)
    test_df = raw.iloc[idx[n_fit + n_val:]].reset_index(drop=True)
    stream = lambda k: np.random.RandomState(seed * 1000 + k)
    ev_rng = lambda: np.random.RandomState(seed + 7)

    scorer, _ = rx.fit_scorer(fit_df, seed)
    S_fit, _, X_fit = scorer(fit_df)
    S_val, _, _ = scorer(val_df)
    S_test, R_test, X_test = scorer(test_df)
    n_test = rx.n_anom_for(len(test_df), dc.CONTAMINATION)
    cols = list(dc.engineer_features(test_df.iloc[:5]).columns)

    gens = {"G2": dc.assemble(dc.prims_g2(n_test, stream(1), fit_df)),
            "G3": dc.assemble(dc.prims_g3(n_test, stream(4), fit_df)),
            "H": dc.assemble(dc.prims_heldout(n_test, stream(5), fit_df))}
    typ_names = {"G2": dc.TYPOLOGIES, "G3": dc.TYPOLOGIES, "H": dc.HELDOUT}
    S_gen = {g: scorer(df)[0] for g, df in gens.items()}
    res = {"seed": seed, "sizes": {"fit": len(fit_df), "val": len(val_df), "test": len(test_df),
                                   "test_anomalies": n_test}}

    # ---------------- E10: out-of-sample rank reference ----------------
    ref_cal = {m: S_val[m] for m in BASE7}

    def fused_cal(S):
        raw_s = {m: S[m] for m in BASE7}
        out = dict(raw_s)
        out.update(dc.fusion_scores(dc.ecdf_ranks(ref_cal, raw_s), dc.ref_zscores(ref_cal, raw_s)))
        return out

    C_val, C_test = fused_cal(S_val), fused_cal(S_test)
    e10 = {"in_sample": {}, "calibration": {}}
    for g, df in gens.items():
        typ = df["__label_typology"].values
        e10["in_sample"][g] = rx.joint_eval(S_test, S_gen[g], typ, RULES, typ_names[g], ev_rng())
        e10["calibration"][g] = rx.joint_eval(C_test, fused_cal(S_gen[g]), typ, RULES, typ_names[g], ev_rng())
    # share of unseen real wallets above the q-quantile of the reference
    cal = {"in_sample": {}, "calibration": {}}
    for m in ["IsolationForest", "DeepAutoencoder", "OneClassSVM", "PCA", "LOF", "DRIFT", "MinRank"]:
        cal["in_sample"][m] = {str(q): float((S_test[m] > np.quantile(S_fit[m], q)).mean()) for q in QS}
        cal["calibration"][m] = {str(q): float((C_test[m] > np.quantile(C_val[m], q)).mean()) for q in QS}
    r_cal_test = dc.ecdf_ranks(ref_cal, {m: S_test[m] for m in ("DeepAutoencoder", "IsolationForest")})
    cal["mean_rank_unseen_real"] = {
        "in_sample": {m: float(R_test[m].mean()) for m in ("DeepAutoencoder", "IsolationForest")},
        "calibration": {m: float(r_cal_test[m].mean()) for m in ("DeepAutoencoder", "IsolationForest")}}
    e10["exceedance"] = cal
    res["E10_reference"] = e10

    # ---------------- E11: label-free volume baselines ----------------
    V_test = dc.volume_scores(dc.engineer_features(test_df))
    V_gen = {g: dc.volume_scores(dc.engineer_features(df)) for g, df in gens.items()}
    vnames = ["VolumeReceived", "VolumeCount"]
    e11 = {g: rx.joint_eval(V_test, V_gen[g], gens[g]["__label_typology"].values, vnames,
                            typ_names[g], ev_rng()) for g in gens}
    hn = {}
    for label, thr in (("ntx_ge_100", 100), ("ntx_ge_1000", 1000)):
        mask = test_df["n_tx"].values >= thr
        y = np.r_[np.zeros(int(mask.sum())), np.ones(n_test)]
        hn[label] = {v: float(roc_auc_score(y, np.r_[V_test[v][mask], V_gen["G2"][v]])) for v in vnames}
    e11["hard_negatives_G2"] = hn
    res["E11_volume"] = e11

    # ---------------- E12: matched negatives ----------------
    cov_test = covariates(test_df)
    allS_test = dict(S_test, **V_test)
    res["E12_matched"] = {
        g: matched_auc(allS_test, dict(S_gen[g], **V_gen[g]), cov_test, covariates(gens[g]),
                       MAIN7 + ["MinRank"] + vnames) for g in gens}

    # ---------------- E13: further detectors (PyOD) ----------------
    if use_pyod:
        e13 = {g: {} for g in gens}
        secs = {}
        X_gen = {g: scorer(df)[2] for g, df in gens.items()}
        for name, make in pyod_models(seed).items():
            t = time.time()
            dc.set_seed(seed)
            with contextlib.redirect_stdout(io.StringIO()):
                clf = make().fit(X_fit)
            # every wallet is scored on its own: no scored wallet enters the model of another
            s_real = np.asarray(clf.decision_function(X_test), dtype=float)
            for g, df in gens.items():
                s = np.r_[s_real, np.asarray(clf.decision_function(X_gen[g]), dtype=float)]
                y = np.r_[np.zeros(len(X_test)), np.ones(len(X_gen[g]))].astype(int)
                typ = np.r_[np.array(["real"] * len(X_test), dtype=object),
                            df["__label_typology"].values.astype(object)]
                e13[g][name] = dc.evaluate(s, y, typ, typ_names[g], dc.BUDGETS, ev_rng())
            secs[name] = time.time() - t
        e13["seconds"] = secs
        res["E13_pyod"] = e13

    # ---------------- E14: ADBench anomaly types ----------------
    names14 = LEARNED + ["DRIFT"]
    real14 = {m: S_test[m] for m in names14}
    y14 = np.r_[np.zeros(len(X_test)), np.ones(n_test)].astype(int)
    F_test = scorer.pre.scaler.inverse_transform(X_test)
    e14 = {"violations_real": identity_violations(F_test, cols), "feature_space": {}, "primitive_space": {}}
    gm_x = fit_gmm(X_fit, seed)
    P_fit = primitives_of(fit_df)
    p_mu, p_sd = P_fit.mean(0), P_fit.std(0) + dc.EPS
    Zp = (P_fit - p_mu) / p_sd
    gm_p = fit_gmm(Zp, seed)
    e14["gmm_components"] = {"feature_space": int(gm_x.n_components), "primitive_space": int(gm_p.n_components)}
    for j, kind in enumerate(ADB_TYPES):
        # (a) in feature space, as published
        Xa = adbench_sample(X_fit, kind, n_test, seed * 1000 + 100 + j, gm_x)
        Sa = score_from_X(scorer, Xa)
        d = {m: float(roc_auc_score(y14, np.r_[real14[m], Sa[m]])) for m in names14}
        d["violations"] = identity_violations(scorer.pre.scaler.inverse_transform(Xa), cols)
        e14["feature_space"][kind] = d
        # (b) on the primitive quantities, records assembled with all identities
        Pa = adbench_sample(Zp, kind, n_test, seed * 1000 + 110 + j, gm_p) * p_sd + p_mu
        rec = records_from_primitives(Pa, stream(120 + j), fit_df, kind)
        Sp, _, Xp = scorer(rec)
        d = {m: float(roc_auc_score(y14, np.r_[S_test[m], Sp[m]])) for m in ["Heuristic"] + names14}
        d["violations"] = identity_violations(scorer.pre.scaler.inverse_transform(Xp), cols)
        e14["primitive_space"][kind] = d
    res["E14_adbench"] = e14

    res["seconds"] = time.time() - t0
    print(f"[ext seed {seed}] done in {res['seconds']:.0f}s  "
          f"DRIFT pr in-sample={e10['in_sample']['G2']['DRIFT']['pr_auc']:.4f} "
          f"calibration={e10['calibration']['G2']['DRIFT']['pr_auc']:.4f}", flush=True)
    return res


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    quick = "--quick" in sys.argv
    use_pyod = "--no-pyod" not in sys.argv
    raw = dc.load_data()
    print(f"corpus: {len(raw):,} wallets", flush=True)
    seeds = dc.SEEDS[:1] if quick else dc.SEEDS
    per_seed = Parallel(n_jobs=1 if quick else 5)(delayed(run_seed)(s, raw, use_pyod) for s in seeds)
    agg = rx.aggregate([{k: v for k, v in r.items() if k != "seed"} for r in per_seed])
    agg["seeds"] = seeds
    tag = "_quick" if quick else ""
    (dc.OUT_DIR / f"extensions_per_seed{tag}.json").write_text(json.dumps(per_seed))
    (dc.OUT_DIR / f"extensions{tag}.json").write_text(json.dumps(agg, indent=1))
    print(f"[save] {dc.OUT_DIR / f'extensions{tag}.json'}", flush=True)


if __name__ == "__main__":
    main()
