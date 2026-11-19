"""
features_v2.py  |  Pre-specified feature revision (behavioral basis)
====================================================================
Specification: revise1/features_prespecification.md (written and frozen before this
code; its SHA-256 digest is recorded in every result file).

Feature sets, all computed from the stored count, received value, sent value and span
(never from the stored rate, which depends on the source of the record):

  F21    the 21 features of the manuscript, unchanged
  W21    F21 with the rate and the burst index computed from the window count
         w = min(n, 50) instead of the lifetime count (change 1)
  B      log count, log received value, sent fraction, log window span, log window
         rate (changes 1 and 2)
  BR     B plus the received value and the span relative to the reference median at
         the same count (changes 1, 2 and 3; the primary revised set)
  BR-u   BR without winsorization (sensitivity analysis)

Detectors receive the same set. PCA components and the autoencoder bottleneck follow
the fixed sizing rule max(2, round(8d/21)) and max(2, round(6d/21)); for d = 21 the
detectors are those of drift_core.Suite and give identical scores.
"""

import contextlib
import hashlib
import io

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

import drift_core as dc
from run_extensions import HeldOutTail

# Every stored result of the article was computed in joblib workers of a 5-process pool on a
# 16-thread processor, where joblib limits the BLAS and OpenMP pools to 16 // 5 = 3 threads.
# The thread count changes the order of floating-point sums, so it is pinned here as well;
# then the results do not depend on the number of workers or on the process that runs them.
THREAD_VARS = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]
N_THREADS = "3"


def check_threads():
    import os
    wrong = {v: os.environ.get(v) for v in THREAD_VARS if os.environ.get(v) != N_THREADS}
    if wrong:
        raise RuntimeError(f"Set {', '.join(THREAD_VARS)} to {N_THREADS} before starting Python "
                           f"(found {wrong}); the stored results were computed with 3 threads per process.")


check_threads()

SPEC = dc.ROOT / "revise1" / "features_prespecification.md"
SETS = ["F21", "W21", "B", "BR", "BR-u"]
WINDOW = 50
B_COLS = ["log_n_tx", "log_total_received", "sent_fraction", "log_window_span", "log_window_rate"]
BR_COLS = B_COLS + ["rel_received", "rel_span"]
MIN_BIN = 50            # minimum number of reference wallets in a count bin
MIN_TYPE = 200          # DRIFT-S: smaller script types are ranked against the whole reference
TOL = 0.05              # tolerance of the identity checks, in feature units


def spec_digest() -> str:
    return hashlib.sha256(SPEC.read_bytes()).hexdigest()


# ----------------------------------------------------------------------
# raw quantities and feature sets
# ----------------------------------------------------------------------
def _num(df, col):
    return np.nan_to_num(df[col].values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)


def raw_quantities(df):
    """n (confirmed count), r and s (received and sent value, satoshi), a (span over the
    window, days), w (window count, min(n, 50), the same for every source)."""
    n = np.maximum(_num(df, "n_tx"), 0.0)
    r = np.maximum(_num(df, "total_received"), 0.0) * 1e8
    s = np.maximum(_num(df, "total_sent"), 0.0) * 1e8
    a = np.maximum(_num(df, "active_days"), 0.0)
    return n, r, s, a, np.minimum(n, WINDOW)


def sent_fraction(r, s):
    q = np.zeros_like(r)
    pos = r > 0
    q[pos] = np.minimum(s[pos] / r[pos], 1.0)
    return q


def window_rate(w, a):
    return np.log1p(w / np.maximum(a, 1.0))


def window_burst(w, a):
    return np.log1p(w / np.sqrt(a + 1.0))


def count_edges(Ln_ref):
    """Bin edges on log(1 + n): the 2nd to 98th percentiles of the reference (observed
    values, duplicates removed); bins with fewer than MIN_BIN reference wallets are merged
    with the bin above (the top bin with the bin below), repeatedly."""
    e = np.unique(np.quantile(Ln_ref, np.arange(1, 50) / 50.0, method="lower"))
    while len(e) > 1:
        cnt = np.bincount(bin_of(e, Ln_ref), minlength=len(e))
        small = np.where(cnt < MIN_BIN)[0]
        if len(small) == 0:
            break
        k = int(small[0])
        e = np.delete(e, k + 1 if k < len(e) - 1 else k)
    return e


def bin_of(edges, Ln):
    """Bin k holds [edges[k], edges[k+1]); values below the first edge join bin 0."""
    return np.clip(np.searchsorted(edges, Ln, side="right") - 1, 0, len(edges) - 1)


class FeatureMap:
    """Maps raw wallet records to one feature set. BR and BR-u are fitted on the
    reference (count bins and bin medians); the other sets need no fitting."""

    def __init__(self, name):
        assert name in SETS
        self.name = name

    def fit(self, ref_df):
        if self.name in ("BR", "BR-u"):
            n, r, _, a, _ = raw_quantities(ref_df)
            Ln, Lr, La = np.log1p(n), np.log1p(r), np.log1p(a)
            self.edges = count_edges(Ln)
            b = bin_of(self.edges, Ln)
            self.med_r = np.array([np.median(Lr[b == k]) for k in range(len(self.edges))])
            self.med_a = np.array([np.median(La[b == k]) for k in range(len(self.edges))])
            self.bin_sizes = np.bincount(b, minlength=len(self.edges)).tolist()
        return self

    def frame(self, df) -> pd.DataFrame:
        if self.name == "F21":
            return dc.engineer_features(df)
        n, r, s, a, w = raw_quantities(df)
        if self.name == "W21":
            f = dc.engineer_features(df).copy()
            f["log_tx_per_day"] = window_rate(w, a)
            f["burst_index"] = window_burst(w, a)
            return f
        f = pd.DataFrame(index=df.index)
        f["log_n_tx"] = np.log1p(n)
        f["log_total_received"] = np.log1p(r)
        f["sent_fraction"] = sent_fraction(r, s)
        f["log_window_span"] = np.log1p(a)
        f["log_window_rate"] = window_rate(w, a)
        if self.name in ("BR", "BR-u"):
            b = bin_of(self.edges, f["log_n_tx"].values)
            f["rel_received"] = f["log_total_received"].values - self.med_r[b]
            f["rel_span"] = f["log_window_span"].values - self.med_a[b]
        return f

    def transform(self, df) -> np.ndarray:
        return self.frame(df).values.astype(float)

    def columns(self):
        if self.name in ("F21", "W21"):
            return dc.FEATURE_NAMES + [f"type_{t}" for t in dc.TYPE_CATEGORIES]
        return BR_COLS if self.name in ("BR", "BR-u") else B_COLS


class Prep(dc.Preprocessor):
    """Winsorization (1st/99th percentiles) and standardization fitted on the reference;
    with winsorize=False standardization only (BR-u)."""

    def __init__(self, winsorize=True):
        self.winsorize = winsorize

    def fit(self, F):
        if self.winsorize:
            return super().fit(F)
        self.scaler = StandardScaler().fit(F)
        return self

    def transform(self, F):
        return super().transform(F) if self.winsorize else self.scaler.transform(F)

    def natural(self, X):
        """Back to (winsorized) feature units."""
        return self.scaler.inverse_transform(X)


# ----------------------------------------------------------------------
# detectors with the sizing rule
# ----------------------------------------------------------------------
def k_pca(d):
    return max(2, int(round(8 * d / 21)))


def k_ae(d):
    return max(2, int(round(6 * d / 21)))


class AE(nn.Module):
    """The autoencoder of drift_core with a bottleneck of k units; for k = 6 the layers
    are built in the same order and receive the same initial weights."""

    def __init__(self, d_in: int, k: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d_in, 32), nn.ReLU(),
                                     nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, k))
        self.decoder = nn.Sequential(nn.Linear(k, 16), nn.ReLU(),
                                     nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, d_in))

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_ae(X, seed, k):
    """dc.train_autoencoder with a k-unit bottleneck (identical steps)."""
    dc.set_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    model = AE(X.shape[1], k)
    opt = torch.optim.Adam(model.parameters(), lr=dc.AE_LR, weight_decay=dc.AE_WD)
    loss_fn = nn.MSELoss()
    n = len(Xt)
    history = []
    for _ in range(dc.AE_EPOCHS):
        model.train()
        perm = torch.randperm(n)
        ep = 0.0
        for i in range(0, n, dc.AE_BATCH):
            idx = perm[i:i + dc.AE_BATCH]
            batch = Xt[idx]
            opt.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            opt.step()
            ep += loss.item() * len(idx)
        history.append(ep / n)
    model.eval()
    return model, history


class SuiteV(dc.Suite):
    """dc.Suite with the dimension-dependent sizes of the specification. The heuristic
    always receives the 21 features of the manuscript (its flags are fixed)."""

    def __init__(self, seed, d):
        super().__init__(seed)
        self.d, self.kp, self.ka = d, k_pca(d), k_ae(d)

    def fit(self, X, feats21):
        seed = self.seed
        rng = np.random.RandomState(seed)
        self.heur = dc.Heuristic().fit(feats21)
        sub = rng.choice(len(X), size=min(5000, len(X)), replace=False)
        self.ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1).fit(X[sub])
        self.lof = LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=dc.NJ).fit(X)
        self.pca = PCA(n_components=self.kp, random_state=seed).fit(X)
        self.iforest = IsolationForest(n_estimators=300, random_state=seed, n_jobs=dc.NJ).fit(X)
        self.ae, self.ae_hist = train_ae(X, seed, self.ka)
        err, z = dc.ae_scores(self.ae, X)
        self.casc = IsolationForest(n_estimators=300, random_state=seed, n_jobs=dc.NJ).fit(
            np.c_[z, np.log1p(err)])
        ref = self.score(X, feats21)
        ref["LOF"] = -self.lof.negative_outlier_factor_
        self.ref_sorted = {m: np.sort(s) for m, s in ref.items()}
        self.ref_mu = {m: float(s.mean()) for m, s in ref.items()}
        self.ref_sd = {m: float(s.std() + dc.EPS) for m, s in ref.items()}
        self.ref_scores = ref
        return self


class ScorerV:
    """Feature map, preprocessing and suite fitted on one reference set."""

    def __init__(self, name, fmap, pre, suite, ref_types):
        self.name, self.fmap, self.pre, self.suite, self.ref_types = name, fmap, pre, suite, ref_types

    def matrix(self, df):
        return self.pre.transform(self.fmap.transform(df))

    def __call__(self, df):
        X = self.matrix(df)
        raw = self.suite.score(X, dc.engineer_features(df))
        r = self.suite.ranks(raw)
        scores = dict(raw)
        scores.update(dc.fusion_scores(r, self.suite.zscores(raw)))
        return scores, r, X

    def reference(self):
        """In-sample reference scores of the fit wallets and their fused scores."""
        raw = self.suite.ref_scores
        out = dict(raw)
        out.update(dc.fusion_scores(self.suite.ranks(raw), self.suite.zscores(raw)))
        return out

    def drift_s(self, raw, types):
        """DRIFT-S: autoencoder and Isolation Forest ranked against the reference wallets
        of the same script type (the whole reference for types with fewer than MIN_TYPE)."""
        ranks = []
        for m in ("DeepAutoencoder", "IsolationForest"):
            out = np.empty(len(types))
            for t in np.unique(types):
                rows = types == t
                ref_t = self.suite.ref_scores[m][self.ref_types == t]
                rs = np.sort(ref_t) if len(ref_t) >= MIN_TYPE else self.suite.ref_sorted[m]
                out[rows] = np.searchsorted(rs, raw[m][rows], side="right") / len(rs)
            ranks.append(out)
        return 0.5 * (ranks[0] + ranks[1])


def canonical_types(df):
    return df["address_type"].map(dc.canonical_type).values.astype(str)


def fit_scorer(df, seed, name):
    df = df.reset_index(drop=True)
    fmap = FeatureMap(name).fit(df)
    F = fmap.transform(df)
    pre = Prep(winsorize=(name != "BR-u")).fit(F)
    X = pre.transform(F)
    suite = SuiteV(seed, F.shape[1]).fit(X, dc.engineer_features(df))
    return ScorerV(name, fmap, pre, suite, canonical_types(df))


def pyod_models(seed, d):
    from pyod.models.deep_svdd import DeepSVDD
    from pyod.models.dif import DIF
    return {"ECOD": lambda: HeldOutTail("ECOD"),
            "COPOD": lambda: HeldOutTail("COPOD"),
            "DIF": lambda: DIF(random_state=seed, device="cpu"),
            "DeepSVDD": lambda: DeepSVDD(n_features=d, batch_size=256, verbose=0, random_state=seed)}


def fit_pyod(X_fit, seed):
    """The four PyOD detectors fitted on a reference matrix, as in run_extensions.py."""
    out = {}
    for name, make in pyod_models(seed, X_fit.shape[1]).items():
        dc.set_seed(seed)
        with contextlib.redirect_stdout(io.StringIO()):
            out[name] = make().fit(X_fit)
    return out


def pyod_reference(clf, X_fit):
    return (np.asarray(clf.reference_scores(), dtype=float) if hasattr(clf, "reference_scores")
            else np.asarray(clf.decision_function(X_fit), dtype=float))


# ----------------------------------------------------------------------
# H': four typologies written down in the specification before any run
# ----------------------------------------------------------------------
HPRIME = ["ransom_collection", "donation_drive", "escrow_float", "otc_settlement"]


def prims_hprime(n, rng, pool) -> dict:
    c = dc._split_counts(n, 4)
    ntx = np.concatenate([rng.randint(3, 13, c[0]), rng.randint(50, 2001, c[1]),
                          rng.randint(60, 801, c[2]), rng.randint(6, 41, c[3])]).astype(float)
    span = np.concatenate([rng.randint(1, 31, c[0]), rng.randint(60, 901, c[1]),
                           rng.randint(20, 366, c[2]), rng.randint(90, 721, c[3])]).astype(float)
    i1 = slice(c[0], c[0] + c[1])
    rec = np.concatenate([10 ** rng.uniform(-0.5, 1.5, c[0]),
                          ntx[i1] * 10 ** rng.uniform(-3.5, -2.0, c[1]),
                          10 ** rng.uniform(0.5, 2.5, c[2]), 10 ** rng.uniform(2.0, 3.5, c[3])])
    sf = np.concatenate([np.ones(c[0]), rng.uniform(0.0, 0.5, c[1]),
                         rng.uniform(0.6, 0.9, c[2]), rng.uniform(0.97, 1.0, c[3])])
    typ = np.repeat(HPRIME, c)
    types, unc = dc._nuisance(n, rng, pool)
    return dict(n_tx=ntx, days=dc._window(span, ntx), rec=rec, sf=sf, type=types, unc=unc, typ=typ)


# ----------------------------------------------------------------------
# validity: identities of each feature set and repair into valid records
# ----------------------------------------------------------------------
def identity_violations(F, scorer):
    """Share of records (in winsorized feature units) that break an identity which a
    wallet record satisfies by construction under the given feature set."""
    from run_extensions import identity_violations as iv21
    name = scorer.name
    cols = scorer.fmap.columns()
    c = {k: i for i, k in enumerate(cols)}
    checks = {}
    if name in ("F21", "W21"):
        base = iv21(F, cols)
        anyv = np.zeros(len(F), dtype=bool)
        checks = {k: v for k, v in base.items() if k != "any"}
        # recompute the per-record mask of the 21-feature identities
        types = F[:, [c[f"type_{t}"] for t in dc.TYPE_CATEGORIES]]
        anyv |= (np.abs(types - np.round(types)).max(axis=1) > TOL) | (np.abs(types.sum(axis=1) - 1.0) > TOL)
        zb = F[:, c["zero_balance_flag"]]
        anyv |= np.abs(zb - np.round(zb)) > TOL
        fr, rr, nf = F[:, c["flow_ratio"]], F[:, c["retention_ratio"]], F[:, c["net_flow_norm"]]
        anyv |= np.abs(fr + rr - 1.0) > TOL
        anyv |= np.abs(nf - (1.0 - fr) / (1.0 + fr)) > TOL
        if name == "W21":
            n = np.maximum(np.expm1(F[:, c["log_n_tx"]]), 0.0)
            a = np.maximum(np.expm1(F[:, c["log_active_days"]]), 0.0)
            w = np.minimum(n, WINDOW)
            rate = np.abs(F[:, c["log_tx_per_day"]] - window_rate(w, a)) > TOL
            burst = np.abs(F[:, c["burst_index"]] - window_burst(w, a)) > TOL
            checks["rate"], checks["burst"] = float(rate.mean()), float(burst.mean())
            anyv |= rate | burst
        checks["any"] = float(anyv.mean())
        return checks
    n = np.maximum(np.expm1(F[:, c["log_n_tx"]]), 0.0)
    a = np.maximum(np.expm1(F[:, c["log_window_span"]]), 0.0)
    q = F[:, c["sent_fraction"]]
    frac = (q < -TOL) | (q > 1.0 + TOL)
    rate = np.abs(F[:, c["log_window_rate"]] - window_rate(np.minimum(n, WINDOW), a)) > TOL
    anyv = frac | rate
    checks = {"sent_fraction": float(frac.mean()), "rate": float(rate.mean())}
    if name in ("BR", "BR-u"):
        b = bin_of(scorer.fmap.edges, F[:, c["log_n_tx"]])
        relr = np.abs(F[:, c["rel_received"]] - (F[:, c["log_total_received"]] - scorer.fmap.med_r[b])) > TOL
        rela = np.abs(F[:, c["rel_span"]] - (F[:, c["log_window_span"]] - scorer.fmap.med_a[b])) > TOL
        checks["rel_received"], checks["rel_span"] = float(relr.mean()), float(rela.mean())
        anyv |= relr | rela
    checks["any"] = float(anyv.mean())
    return checks


def repair(F, scorer, pool, kind, rng):
    """Reassemble feature-space samples into valid wallet records: count, span, received
    value and sent fraction are read from the sampled vector, every derived field is
    recomputed by dc.assemble. Under F21 and W21 the script type and unconfirmed count are
    read from the vector, as in adbench_repair.py; under the other sets, which do not
    contain them, they follow the corpus."""
    from adbench_repair import repair as repair21
    cols = scorer.fmap.columns()
    if scorer.name in ("F21", "W21"):
        return repair21(F, cols, pool, kind)
    c = {k: i for i, k in enumerate(cols)}
    n_tx = np.clip(np.expm1(F[:, c["log_n_tx"]]), 1.0, 5e6)
    days = np.clip(np.expm1(F[:, c["log_window_span"]]), 1.0, 6500.0)
    rec = np.clip(np.expm1(F[:, c["log_total_received"]]) / 1e8, 1e-8, 2.1e7)
    sf = np.clip(F[:, c["sent_fraction"]], 0.0, 1.0)
    types, unc = dc._nuisance(len(F), rng, pool)
    return dc.assemble(dict(n_tx=n_tx, days=days, rec=rec, sf=sf, type=types, unc=unc,
                            typ=np.array([f"adbfix_{kind}"] * len(F), dtype=object)))
