"""
run_variants.py  |  Two pre-specified variants of DRIFT on the injected benchmarks
==================================================================================
Specification: revise1/variants_prespecification.md (written before this code).

  DRIFT-C   mean of size-conditional ranks: each detector score is ranked against the
            K nearest reference wallets in log count, log received value and log
            active span, within the same script type (K = 500; 200 and 1000 as
            sensitivity)
  DRIFT-UC  mean of the two unconditional and the two conditional ranks
  DRIFT-T   trimmed reference: the top q of the fit split by DRIFT score is removed,
            the autoencoder and the Isolation Forest are refitted on the rest, and
            the rest is the reference (q = 0.02; 0.01 and 0.05 as sensitivity)

Benchmarks: G2 (primary, span measured over the last 50 transactions as the scraper
measures real wallets), G2L (the same draws with the span over the whole activity, a
check only), G3 and H, with the seeds, splits and anomaly
streams of run_experiments.py. Contamination of the fit split at 0.5% to 5% with
independent draws of each benchmark's own generator.

Nothing in revise1/outputs is overwritten; results go to revise1/outputs/variants/.

Run:  python revise1/src/run_variants.py [--quick]
"""

import json
import sys
import time

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

import drift_core as dc
import run_experiments as rx
import run_extensions as rex

OUT = dc.OUT_DIR / "variants"
K_MAIN, K_SENS = 500, (200, 1000)
Q_MAIN, Q_SENS = 0.02, (0.01, 0.05)
CONT = [0.005, 0.01, 0.02, 0.05]
COMP = ["IsolationForest", "DeepAutoencoder"]
NAMES = ["IsolationForest", "DeepAutoencoder", "DRIFT", "MinRank", "IF-C", "AE-C",
         "DRIFT-C", "DRIFT-UC", "IF-T", "AE-T", "DRIFT-T", "VolumeReceived"]
CONT_NAMES = ["IsolationForest", "DeepAutoencoder", "DRIFT", "DRIFT-C", "DRIFT-UC",
              "IF-T", "AE-T", "DRIFT-T"]
GEN_STREAM = {"G2": (dc.prims_g2, 60), "G3": (dc.prims_g3, 70), "H": (dc.prims_heldout, 80)}


# ----------------------------------------------------------------------
# the two components with the settings of dc.Suite
# ----------------------------------------------------------------------
class Pair:
    """Preprocessing, Isolation Forest and autoencoder fitted on one reference set,
    with the hyperparameters and seeds of dc.Suite, so that the scores equal those
    of the manuscript for the same seed and machine."""

    def fit(self, df, seed, pre=None):
        df = df.reset_index(drop=True)
        F = dc.engineer_features(df).values.astype(float)
        self.pre = pre if pre is not None else dc.Preprocessor().fit(F)
        X = self.pre.transform(F)
        self.iforest = IsolationForest(n_estimators=300, random_state=seed, n_jobs=dc.NJ).fit(X)
        self.ae, _ = dc.train_autoencoder(X, seed)
        self.ref_df = df
        self.ref = self.raw_X(X)
        self.ref_sorted = {m: np.sort(v) for m, v in self.ref.items()}
        return self

    def raw_X(self, X):
        err, _ = dc.ae_scores(self.ae, X)
        return {"IsolationForest": -self.iforest.score_samples(X), "DeepAutoencoder": err}

    def raw(self, df):
        return self.raw_X(self.pre.transform(dc.engineer_features(df).values.astype(float)))

    def ranks(self, raw):
        return {m: np.searchsorted(self.ref_sorted[m], raw[m], side="right") / len(self.ref_sorted[m])
                for m in COMP}

    def fused_ref(self):
        r = self.ranks(self.ref)
        return 0.5 * (r["IsolationForest"] + r["DeepAutoencoder"])


def trimmed(pair, seed, q):
    """DRIFT-T: remove the top q of the reference by its in-sample DRIFT score and
    refit both components on the rest, with the preprocessing unchanged."""
    fused = pair.fused_ref()
    n_drop = int(round(q * len(fused)))
    drop = np.argsort(-fused, kind="stable")[:n_drop]
    keep = np.setdiff1d(np.arange(len(fused)), drop)
    return Pair().fit(pair.ref_df.iloc[keep], seed, pre=pair.pre)


# ----------------------------------------------------------------------
# size-conditional ranks
# ----------------------------------------------------------------------
class CondRanker:
    """Rank of a score among the K reference wallets nearest in standardized log
    count, log received value and log span, within the same canonical script type
    (all reference wallets when the type holds too few)."""

    def __init__(self, ref_df, ref_raw, k):
        C = rex.covariates(ref_df)
        self.mu, self.sd = C.mean(0), C.std(0) + dc.EPS
        Cz = (C - self.mu) / self.sd
        self.types = ref_df["address_type"].map(dc.canonical_type).values
        self.ref_raw, self.k = ref_raw, k
        self.nn = {}
        for t in np.unique(self.types):
            idx = np.where(self.types == t)[0]
            if len(idx) >= k + 1:
                self.nn[t] = (idx, NearestNeighbors().fit(Cz[idx]))
        self.nn_all = (np.arange(len(Cz)), NearestNeighbors().fit(Cz))

    def neighbors(self, df, self_idx=None):
        """Reference indices [n, k] of the neighborhood of each wallet. self_idx gives
        the reference index of each scored wallet when the scored wallets are the
        reference itself; the wallet is then left out of its own neighborhood."""
        Cz = (rex.covariates(df) - self.mu) / self.sd
        t = df["address_type"].map(dc.canonical_type).values
        out = np.zeros((len(df), self.k), dtype=np.int32)
        extra = 0 if self_idx is None else 1
        for tt in np.unique(t):
            rows = np.where(t == tt)[0]
            idx, nn = self.nn.get(tt, self.nn_all)
            _, ind = nn.kneighbors(Cz[rows], n_neighbors=self.k + extra)
            ind = idx[ind]
            if extra:
                keep = ind != self_idx[rows][:, None]
                keep[~(~keep).any(axis=1), -1] = False      # self not found: drop the farthest
                ind = ind[keep].reshape(len(rows), self.k)
            out[rows] = ind
        return out

    def ranks(self, raw, nbr):
        return {m: (self.ref_raw[m][nbr] <= raw[m][:, None]).mean(axis=1) for m in COMP}


def scoreset(pair, cond, raw, nbr):
    r = pair.ranks(raw)
    c = cond.ranks(raw, nbr)
    return {"IsolationForest": raw["IsolationForest"], "DeepAutoencoder": raw["DeepAutoencoder"],
            "DRIFT": 0.5 * (r["IsolationForest"] + r["DeepAutoencoder"]),
            "MinRank": np.minimum(r["IsolationForest"], r["DeepAutoencoder"]),
            "IF-C": c["IsolationForest"], "AE-C": c["DeepAutoencoder"],
            "DRIFT-C": 0.5 * (c["IsolationForest"] + c["DeepAutoencoder"]),
            "DRIFT-UC": 0.25 * (r["IsolationForest"] + r["DeepAutoencoder"]
                                + c["IsolationForest"] + c["DeepAutoencoder"])}


def add_trimmed(S, pt, raw_t):
    r = pt.ranks(raw_t)
    S["IF-T"] = raw_t["IsolationForest"]
    S["AE-T"] = raw_t["DeepAutoencoder"]
    S["DRIFT-T"] = 0.5 * (r["IsolationForest"] + r["DeepAutoencoder"])
    return S


def add_volume(S, df):
    S["VolumeReceived"] = dc.volume_scores(dc.engineer_features(df))["VolumeReceived"]
    return S


def slim(ev):
    """Keep the headline metrics and the per-typology recall at 5% and 10%."""
    return {m: {k: v for k, v in d.items()
                if k in ("roc_auc", "pr_auc", "oracle_f1") or k.startswith(("P@", "R@0.05", "R@0.1"))}
            for m, d in ev.items()}


# ----------------------------------------------------------------------
# one seed
# ----------------------------------------------------------------------
def run_seed(seed, raw, quick=False):
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

    p2l = dc.prims_g2(n_test, stream(1), fit_df, window=False)
    gens = {"G2": dc.assemble(dc.observe(p2l)), "G2L": dc.assemble(p2l),
            "G3": dc.assemble(dc.prims_g3(n_test, stream(4), fit_df)),
            "H": dc.assemble(dc.prims_heldout(n_test, stream(5), fit_df))}
    tnames = {"G2": dc.TYPOLOGIES, "G2L": dc.TYPOLOGIES, "G3": dc.TYPOLOGIES, "H": dc.HELDOUT}
    res = {"seed": seed, "sizes": {"fit": len(fit_df), "test": len(test_df), "test_anomalies": n_test}}

    # ---------------- clean fit ----------------
    pair = Pair().fit(fit_df, seed)
    cond = CondRanker(pair.ref_df, pair.ref, K_MAIN)
    pt = trimmed(pair, seed, Q_MAIN)
    raw_test, raw_gen = pair.raw(test_df), {g: pair.raw(df) for g, df in gens.items()}
    S_test = add_volume(add_trimmed(scoreset(pair, cond, raw_test, cond.neighbors(test_df)),
                                    pt, pt.raw(test_df)), test_df)
    S_gen = {g: add_volume(add_trimmed(scoreset(pair, cond, raw_gen[g], cond.neighbors(df)),
                                       pt, pt.raw(df)), df) for g, df in gens.items()}
    cov_test = rex.covariates(test_df)
    res["main"], res["matched_auc"] = {}, {}
    for g, df in gens.items():
        typ = df["__label_typology"].values
        res["main"][g] = slim(rx.joint_eval(S_test, S_gen[g], typ, NAMES, tnames[g], ev()))
        res["matched_auc"][g] = rex.matched_auc(S_test, S_gen[g], cov_test, rex.covariates(df), NAMES)

    # hard negatives on G2: real test wallets with at least 100 / 1000 transactions
    hn = {}
    for label, thr in (("ntx_ge_100", 100), ("ntx_ge_1000", 1000)):
        mask = test_df["n_tx"].values >= thr
        y = np.r_[np.zeros(int(mask.sum())), np.ones(n_test)]
        hn[label] = {m: float(roc_auc_score(y, np.r_[S_test[m][mask], S_gen["G2"][m]])) for m in NAMES}
    res["hard_negatives_G2"] = hn

    # calibration: share of unseen real test wallets above a quantile of the reference
    ref_fused = {"DRIFT": pair.fused_ref(), "DRIFT-T": pt.fused_ref()}
    rc = cond.ranks(pair.ref, cond.neighbors(pair.ref_df, self_idx=np.arange(len(pair.ref_df))))
    ref_fused["DRIFT-C"] = 0.5 * (rc["IsolationForest"] + rc["DeepAutoencoder"])
    res["exceedance"] = {m: {str(q): float((S_test[m] > np.quantile(ref_fused[m], q)).mean())
                             for q in (0.90, 0.95, 0.99)} for m in ref_fused}
    res["trim"] = {"q": Q_MAIN, "n_dropped": int(len(pair.ref_df) - len(pt.ref_df)),
                   "dropped_median_n_tx": float(np.median(pair.ref_df["n_tx"].values[
                       np.argsort(-pair.fused_ref(), kind="stable")[:int(round(Q_MAIN * len(pair.ref_df)))]])),
                   "reference_median_n_tx": float(np.median(pair.ref_df["n_tx"].values))}

    # ---------------- sensitivity: K and q ----------------
    sens = {"K": {}, "q": {}}
    for k in K_SENS:
        ck = CondRanker(pair.ref_df, pair.ref, k)
        St = ck.ranks(raw_test, ck.neighbors(test_df))
        St = {"DRIFT-C": 0.5 * (St["IsolationForest"] + St["DeepAutoencoder"])}
        d = {}
        for g in ("G2", "G3", "H"):
            Sg = ck.ranks(raw_gen[g], ck.neighbors(gens[g]))
            Sg = {"DRIFT-C": 0.5 * (Sg["IsolationForest"] + Sg["DeepAutoencoder"])}
            e = rx.joint_eval(St, Sg, gens[g]["__label_typology"].values, ["DRIFT-C"], tnames[g], ev())
            d[g] = {"roc_auc": e["DRIFT-C"]["roc_auc"], "pr_auc": e["DRIFT-C"]["pr_auc"],
                    "matched_auc": rex.matched_auc(St, Sg, cov_test, rex.covariates(gens[g]), ["DRIFT-C"])["DRIFT-C"]}
        sens["K"][str(k)] = d
    for q in Q_SENS:
        pq = trimmed(pair, seed, q)
        St = add_trimmed({}, pq, pq.raw(test_df))
        Sg = add_trimmed({}, pq, pq.raw(gens["G2"]))
        e = rx.joint_eval(St, Sg, gens["G2"]["__label_typology"].values, ["DRIFT-T"], dc.TYPOLOGIES, ev())
        sens["q"][f"{q}|clean|G2"] = {"roc_auc": e["DRIFT-T"]["roc_auc"], "pr_auc": e["DRIFT-T"]["pr_auc"]}
    res["sensitivity"] = sens
    if quick:
        res["seconds"] = time.time() - t0
        return res

    # ---------------- contaminated fit split ----------------
    res["contaminated"] = {g: {} for g in GEN_STREAM}
    for j, c in enumerate(CONT):
        for g, (gen, k0) in GEN_STREAM.items():
            cont = dc.assemble(gen(rx.n_anom_for(len(fit_df), c), stream(k0 + j), fit_df))
            fit_c = pd.concat([fit_df, cont.drop(columns="__label_typology")], ignore_index=True, sort=False)
            fit_c = fit_c.sample(frac=1.0, random_state=seed).reset_index(drop=True)
            pc = Pair().fit(fit_c, seed)
            cc = CondRanker(pc.ref_df, pc.ref, K_MAIN)
            ptc = trimmed(pc, seed, Q_MAIN)
            Sr = add_trimmed(scoreset(pc, cc, pc.raw(test_df), cc.neighbors(test_df)), ptc, ptc.raw(test_df))
            Sa = add_trimmed(scoreset(pc, cc, pc.raw(gens[g]), cc.neighbors(gens[g])), ptc, ptc.raw(gens[g]))
            e = rx.joint_eval(Sr, Sa, gens[g]["__label_typology"].values, CONT_NAMES, tnames[g], ev())
            res["contaminated"][g][str(c)] = {m: {"roc_auc": e[m]["roc_auc"], "pr_auc": e[m]["pr_auc"],
                                                  "P@0.01": e[m]["P@0.01"], "R@0.05": e[m]["R@0.05"]}
                                              for m in CONT_NAMES}
            if g == "G2" and c == 0.01:
                for q in Q_SENS:
                    pq = trimmed(pc, seed, q)
                    St = add_trimmed({}, pq, pq.raw(test_df))
                    Sg = add_trimmed({}, pq, pq.raw(gens[g]))
                    eq = rx.joint_eval(St, Sg, gens[g]["__label_typology"].values, ["DRIFT-T"], tnames[g], ev())
                    res["sensitivity"]["q"][f"{q}|0.01|G2"] = {"roc_auc": eq["DRIFT-T"]["roc_auc"],
                                                               "pr_auc": eq["DRIFT-T"]["pr_auc"]}
    res["seconds"] = time.time() - t0
    print(f"[variants seed {seed}] {res['seconds']:.0f}s  G2 PR-AUC DRIFT "
          f"{res['main']['G2']['DRIFT']['pr_auc']:.4f} DRIFT-C {res['main']['G2']['DRIFT-C']['pr_auc']:.4f} "
          f"DRIFT-T {res['main']['G2']['DRIFT-T']['pr_auc']:.4f}", flush=True)
    return res


# ----------------------------------------------------------------------
# pre-specified criteria
# ----------------------------------------------------------------------
def paired(per_seed, get_a, get_b):
    a = np.array([get_a(r) for r in per_seed])
    b = np.array([get_b(r) for r in per_seed])
    d = a - b
    p = float("nan") if np.allclose(d, 0) else float(wilcoxon(a, b).pvalue)
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()), "mean_diff": float(d.mean()),
            "wins": int((d > 0).sum()), "p": p}


def criteria(per_seed):
    out = {"C1": {}, "T1": {}, "T2": {}}
    for g in ("G3", "H"):
        t = paired(per_seed, lambda r: r["matched_auc"][g]["DRIFT-C"], lambda r: r["matched_auc"][g]["DRIFT"])
        t["holds"] = bool(t["mean_a"] >= 0.60 and t["mean_diff"] >= 0.05 and t["p"] <= 0.01)
        out["C1"][g] = t
    out["C1"]["holds"] = any(out["C1"][g]["holds"] for g in ("G3", "H"))
    t1 = True
    for g in GEN_STREAM:
        for c in CONT:
            t = paired(per_seed, lambda r: r["contaminated"][g][str(c)]["DRIFT-T"]["pr_auc"],
                       lambda r: r["contaminated"][g][str(c)]["IsolationForest"]["pr_auc"])
            t["holds"] = bool(t["mean_diff"] >= 0 or t["p"] >= 0.05)
            t1 &= t["holds"]
            out["T1"][f"{g}|{c}"] = t
    out["T1"]["holds"] = t1
    t2 = True
    for g in ("G2", "G3", "H"):
        t = paired(per_seed, lambda r: r["main"][g]["DRIFT-T"]["pr_auc"], lambda r: r["main"][g]["DRIFT"]["pr_auc"])
        t["holds"] = bool(t["mean_diff"] >= -0.01)
        t2 &= t["holds"]
        out["T2"][g] = t
    out["T2"]["holds"] = t2
    out["variant2_succeeds"] = bool(t1 and t2)
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    quick = "--quick" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    raw = dc.load_data()
    print(f"corpus: {len(raw):,} wallets", flush=True)
    seeds = dc.SEEDS[:1] if quick else dc.SEEDS
    t = time.time()
    per_seed = Parallel(n_jobs=1 if quick else 5)(delayed(run_seed)(s, raw, quick) for s in seeds)
    tag = "_quick" if quick else ""
    (OUT / f"variants_per_seed{tag}.json").write_text(json.dumps(per_seed))
    agg = rx.aggregate([{k: v for k, v in r.items() if k != "seed"} for r in per_seed])
    agg["seeds"] = seeds
    agg["software"] = {"python": sys.version.split()[0], "torch": torch.__version__,
                       "numpy": np.__version__, "pandas": pd.__version__}
    if not quick:
        agg["criteria"] = criteria(per_seed)
    (OUT / f"variants{tag}.json").write_text(json.dumps(agg, indent=1))
    print(f"[save] {OUT / f'variants{tag}.json'}  ({time.time() - t:.0f}s)", flush=True)
    m = agg["main"]
    for g in ("G2", "G2L", "G3", "H"):
        print(g, "  ".join(f"{k} {m[g][k]['pr_auc']['mean']:.3f}" for k in NAMES), flush=True)
    for g in ("G2", "G2L", "G3", "H"):
        print("matched", g, "  ".join(f"{k} {agg['matched_auc'][g][k]['mean']:.3f}" for k in NAMES), flush=True)
    if not quick:
        print(json.dumps({k: v.get("holds", v) if isinstance(v, dict) else v
                          for k, v in agg["criteria"].items()}), flush=True)


if __name__ == "__main__":
    main()
