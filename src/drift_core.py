"""
drift_core.py  |  Shared components for the Revision 1 experiments
===================================================================
Manuscript T-IFS-27099-2026 (major revision).

This module keeps the data loading, the 21 behavioural features, the
autoencoder and the hyperparameters of the original submission unchanged,
and adds what the reviewers asked for:

  * an inductive (held-out) protocol: every preprocessing statistic and
    every detector is fitted on a fit split and applied to wallets that
    were never seen during fitting;
  * rank normalisation of unseen wallets through the empirical CDF of the
    reference (fit-split) scores;
  * four anomaly generators: G1 (the generator of the original
    submission, kept verbatim), G2 (same typology ranges, artefacts
    removed), G3 (independent parameterisation) and H (four held-out
    typologies);
  * established score-aggregation rules to compare against DRIFT;
  * fixed-budget metrics.
"""

import os
import glob
import warnings
from pathlib import Path

# ----------------------------------------------------------------------
# 0. Deterministic floating-point arithmetic
# ----------------------------------------------------------------------
# Training the autoencoder amplifies differences in the floating-point kernels
# that PyTorch's math libraries choose for a processor: the same code and seeds
# gave a ten-seed PR-AUC of DRIFT on G2 of 0.272 on a processor with AVX-512 and
# 0.259 on one without it. The kernels are therefore pinned to the AVX2 code
# path, which every processor with AVX2 executes identically. The variables must
# be set before torch and numpy are loaded, so run_all.py sets them for every
# script; here they are set as defaults and then verified.
PINNED_ENV = {"MKL_CBWR": "AVX2", "ATEN_CPU_CAPABILITY": "avx2", "OPENBLAS_CORETYPE": "Haswell"}
for _k, _v in PINNED_ENV.items():
    os.environ.setdefault(_k, _v)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

warnings.filterwarnings("ignore")


def check_determinism():
    """Stop unless the pinned kernels are in effect in this process."""
    wrong = {k: os.environ.get(k) for k, v in PINNED_ENV.items() if os.environ.get(k) != v}
    cap = torch.backends.cpu.get_cpu_capability()
    from threadpoolctl import threadpool_info
    blas = {i.get("architecture") for i in threadpool_info() if i.get("internal_api") == "openblas"}
    if wrong or cap != "AVX2" or not blas <= {"Haswell"}:
        raise RuntimeError(
            f"Non-deterministic setup (environment {wrong}, ATen capability {cap}, OpenBLAS {blas}). Start Python with "
            "MKL_CBWR=AVX2, ATEN_CPU_CAPABILITY=avx2 and OPENBLAS_CORETYPE=Haswell, or run src/run_all.py.")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)


check_determinism()

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "blockchain-dataset"
OUT_DIR = Path(os.environ.get("DRIFT_OUT_DIR", ROOT / "revise1" / "outputs"))   # overridden only for the verification run
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [11, 23, 42, 77, 101, 131, 151, 181, 211, 251]
CONTAMINATION = 0.05
AE_EPOCHS, AE_BATCH, AE_LR, AE_WD = 80, 256, 1e-3, 1e-5
EPS = 1e-9
BUDGETS = [0.01, 0.02, 0.05, 0.10]
NJ = int(os.environ.get("DRIFT_NJOBS", "2"))

TYPOLOGIES = ["peeling_chain", "mixer_passthrough", "rapid_cashout", "dormant_burst"]
HELDOUT = ["hoard", "dust_fan", "structuring", "single_hop"]


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------------------------------------------------
# 1. Data loading and consolidation (unchanged from the submission)
# ----------------------------------------------------------------------
NUMERIC_COLS = ["n_tx", "unconfirmed_n_tx", "final_n_tx", "total_received",
                "total_sent", "balance", "unconfirmed_balance", "final_balance",
                "active_days", "tx_per_day", "avg_tx_value", "max_tx_value",
                "min_tx_value", "total_tx_value"]


def load_data() -> pd.DataFrame:
    files = sorted(glob.glob(str(DATA_DIR / "wallet_dataset_*.csv")))
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["__source"] = os.path.basename(f)
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    for c in NUMERIC_COLS:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw.dropna(subset=["address", "n_tx"])
    raw = raw[raw["n_tx"] >= 0]
    raw = raw.sort_values("n_tx", ascending=False).drop_duplicates("address", keep="first")
    raw = raw.reset_index(drop=True)
    raw["address_type"] = raw["address_type"].fillna("Unknown")
    return raw


def canonical_type(t: str) -> str:
    t = str(t).lower()
    if "p2wpkh" in t:
        return "P2WPKH"
    if "p2wsh" in t:
        return "P2WSH"
    if "taproot" in t or "p2tr" in t:
        return "Taproot"
    if "p2sh" in t:
        return "P2SH"
    if "legacy" in t or "p2pkh" in t:
        return "Legacy"
    if "bech32" in t or "segwit" in t:
        return "Bech32_other"
    return "Unknown"


# ----------------------------------------------------------------------
# 2. Feature engineering (unchanged from the submission)
# ----------------------------------------------------------------------
FEATURE_NAMES = [
    "log_n_tx", "log_total_received", "log_total_sent", "log_final_balance",
    "flow_ratio", "retention_ratio", "log_active_days", "log_tx_per_day",
    "net_flow_norm", "log_recv_per_tx", "log_sent_per_tx", "burst_index",
    "unconfirmed_ratio", "zero_balance_flag",
]
TYPE_CATEGORIES = ["P2WPKH", "P2WSH", "Taproot", "P2SH", "Legacy", "Bech32_other", "Unknown"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    sat = 1e8
    rec = np.maximum(df["total_received"].values.astype(float), 0)
    snt = np.maximum(df["total_sent"].values.astype(float), 0)
    bal = np.maximum(df["final_balance"].values.astype(float), 0)
    ntx = np.maximum(df["n_tx"].values.astype(float), 0)
    act = np.maximum(df["active_days"].values.astype(float), 0)
    tpd = np.maximum(df["tx_per_day"].values.astype(float), 0)
    unc = np.maximum(df["unconfirmed_n_tx"].values.astype(float), 0)

    feats = pd.DataFrame(index=df.index)
    feats["log_n_tx"] = np.log1p(ntx)
    feats["log_total_received"] = np.log1p(rec * sat)
    feats["log_total_sent"] = np.log1p(snt * sat)
    feats["log_final_balance"] = np.log1p(bal * sat)
    feats["flow_ratio"] = np.clip(snt / (rec + EPS), 0, 8)
    feats["retention_ratio"] = np.clip(bal / (rec + EPS), 0, 3)
    feats["log_active_days"] = np.log1p(act)
    feats["log_tx_per_day"] = np.log1p(tpd)
    feats["net_flow_norm"] = (rec - snt) / (rec + snt + EPS)
    feats["log_recv_per_tx"] = np.log1p((rec * sat) / (ntx + EPS))
    feats["log_sent_per_tx"] = np.log1p((snt * sat) / (ntx + EPS))
    feats["burst_index"] = np.log1p(ntx / np.sqrt(act + 1.0))
    feats["unconfirmed_ratio"] = np.clip(unc / (ntx + EPS), 0, 1)
    feats["zero_balance_flag"] = (bal < 1e-8).astype(float)

    ctype = df["address_type"].map(canonical_type)
    for cat in TYPE_CATEGORIES:
        feats[f"type_{cat}"] = (ctype == cat).astype(float)
    return feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)


class Preprocessor:
    """Winsorisation (1st/99th percentile) and standardisation whose
    statistics are estimated on the fit split only."""

    def fit(self, F: np.ndarray):
        self.lo = np.percentile(F, 1.0, axis=0)
        self.hi = np.percentile(F, 99.0, axis=0)
        self.clip = self.hi > self.lo
        self.scaler = StandardScaler().fit(self._win(F))
        return self

    def _win(self, F):
        out = np.array(F, dtype=float, copy=True)
        out[:, self.clip] = np.clip(out[:, self.clip], self.lo[self.clip], self.hi[self.clip])
        return out

    def transform(self, F):
        return self.scaler.transform(self._win(F))


# ----------------------------------------------------------------------
# 3. Anomaly generators
# ----------------------------------------------------------------------
_ZERO_COLS = ["avg_tx_value", "max_tx_value", "min_tx_value", "total_tx_value"]


def gen_g1(n: int, rng: np.random.RandomState, noise=(0.8, 1.25)) -> pd.DataFrame:
    """G1: generator of the original submission, reproduced verbatim."""
    per = n // len(TYPOLOGIES)
    rows = []

    def add(typ, n_tx, active_days, total_received, sent_frac, addr_type):
        total_received = max(total_received, 1e-6)
        total_sent = total_received * sent_frac
        final_balance = max(total_received - total_sent, 0.0)
        active_days = max(int(active_days), 1)
        rows.append(dict(
            address=f"SYN_{typ}_{len(rows)}", address_type=addr_type,
            n_tx=int(n_tx), unconfirmed_n_tx=int(rng.choice([0, 0, 0, 1])),
            final_n_tx=int(n_tx), total_received=total_received,
            total_sent=total_sent, balance=final_balance, unconfirmed_balance=0.0,
            final_balance=final_balance, active_days=active_days,
            tx_per_day=n_tx / active_days, avg_tx_value=0.0, max_tx_value=0.0,
            min_tx_value=0.0, total_tx_value=0.0, __label_typology=typ))

    for _ in range(per):
        add("peeling_chain", rng.randint(120, 1600), rng.randint(2, 45),
            10 ** rng.uniform(-0.6, 1.4), rng.uniform(0.965, 0.999),
            rng.choice(["P2WPKH", "Legacy", "P2SH"]))
    for _ in range(per):
        add("mixer_passthrough", rng.randint(400, 5000), rng.randint(1, 8),
            10 ** rng.uniform(0.3, 2.6), rng.uniform(0.985, 1.0),
            rng.choice(["P2WPKH", "P2SH", "Bech32_other"]))
    for _ in range(per):
        add("rapid_cashout", rng.randint(8, 90), rng.randint(1, 3),
            10 ** rng.uniform(1.0, 3.0), rng.uniform(0.97, 1.0),
            rng.choice(["P2WPKH", "Taproot", "P2SH"]))
    for _ in range(n - 3 * per):
        add("dormant_burst", rng.randint(3, 22), rng.randint(280, 1800),
            10 ** rng.uniform(-0.4, 1.8), rng.uniform(0.90, 1.0),
            rng.choice(["Legacy", "P2SH", "P2WPKH"]))

    df = pd.DataFrame(rows)
    for c in ["n_tx", "total_received", "total_sent", "final_balance", "tx_per_day"]:
        df[c] = df[c] * rng.uniform(noise[0], noise[1], len(df))
    df["n_tx"] = df["n_tx"].clip(lower=1).round().astype(int)
    df["final_n_tx"] = df["n_tx"]
    return df


def assemble(prim: dict) -> pd.DataFrame:
    """Turn primitive behavioural quantities into a raw wallet record that
    satisfies every accounting identity of a real record:
    sent <= received, balance = received - sent, rate = count / span."""
    n_tx = np.maximum(np.round(prim["n_tx"]), 1).astype(int)
    days = np.maximum(np.round(prim["days"]), 1).astype(int)
    rec = np.round(np.maximum(prim["rec"], 1e-6), 8)
    sent = np.minimum(np.round(rec * np.clip(prim["sf"], 0.0, 1.0), 8), rec)
    bal = np.round(rec - sent, 8)
    unc = np.asarray(prim["unc"]).astype(int)
    n = len(n_tx)
    df = pd.DataFrame(dict(
        address=[f"SYN_{t}_{i}" for i, t in enumerate(prim["typ"])],
        address_type=prim["type"], n_tx=n_tx, unconfirmed_n_tx=unc,
        final_n_tx=n_tx + unc, total_received=rec, total_sent=sent, balance=bal,
        unconfirmed_balance=np.zeros(n), final_balance=bal, active_days=days,
        tx_per_day=n_tx / days))
    for c in _ZERO_COLS:
        df[c] = 0.0
    df["__label_typology"] = prim["typ"]
    return df


def _nuisance(n, rng, pool):
    """Script type and unconfirmed count follow the real corpus, so that
    neither field carries label information."""
    types = pool["address_type"].values[rng.randint(0, len(pool), n)]
    p_unc = float((pool["unconfirmed_n_tx"].values > 0).mean())
    unc = (rng.uniform(size=n) < p_unc).astype(int)
    return types, unc


def _split_counts(n, k):
    per = n // k
    return [per] * (k - 1) + [n - per * (k - 1)]


def prims_g2(n, rng, pool, noise=(0.8, 1.25), window=True) -> dict:
    """G2: the typology ranges of the submission with the generator
    artefacts removed. Noise acts on the primitives, every derived field is
    recomputed, and the nuisance fields follow the corpus. The active span is
    measured over the last 50 transactions, as the scraper measures real wallets
    (observe). With window=False the span covers the whole activity of the wallet:
    this is G2L, the step of the audit between G1 and G2."""
    c = _split_counts(n, 4)
    ntx = np.concatenate([rng.randint(120, 1600, c[0]), rng.randint(400, 5000, c[1]),
                          rng.randint(8, 90, c[2]), rng.randint(3, 22, c[3])]).astype(float)
    days = np.concatenate([rng.randint(2, 45, c[0]), rng.randint(1, 8, c[1]),
                           rng.randint(1, 3, c[2]), rng.randint(280, 1800, c[3])]).astype(float)
    rec = np.concatenate([10 ** rng.uniform(-0.6, 1.4, c[0]), 10 ** rng.uniform(0.3, 2.6, c[1]),
                          10 ** rng.uniform(1.0, 3.0, c[2]), 10 ** rng.uniform(-0.4, 1.8, c[3])])
    sf = np.concatenate([rng.uniform(0.965, 0.999, c[0]), rng.uniform(0.985, 1.0, c[1]),
                         rng.uniform(0.97, 1.0, c[2]), rng.uniform(0.90, 1.0, c[3])])
    typ = np.repeat(TYPOLOGIES, c)
    ntx = ntx * rng.uniform(noise[0], noise[1], n)
    rec = rec * rng.uniform(noise[0], noise[1], n)
    retained = np.clip((1.0 - sf) * rng.uniform(noise[0], noise[1], n), 0.0, 1.0)
    types, unc = _nuisance(n, rng, pool)
    prim = dict(n_tx=ntx, days=days, rec=rec, sf=1.0 - retained, type=types, unc=unc, typ=typ)
    return observe(prim) if window else prim


def observe(prim) -> dict:
    """Apply the observation model of the scraper to primitive quantities whose
    span covers the whole activity of a wallet (see _window)."""
    out = dict(prim)
    out["days"] = _window(prim["days"], prim["n_tx"])
    return out


def _ln(rng, med, sig, size, lo, hi):
    return np.clip(np.exp(rng.normal(np.log(med), sig, size)), lo, hi)


def _window(span, ntx, w=50):
    """Observation model of the scraper: the active span is measured over
    the most recent w transactions only."""
    return np.maximum(np.round(span * np.minimum(1.0, w / np.maximum(ntx, 1.0))), 1.0)


def prims_g3(n, rng, pool) -> dict:
    """G3: an independently parameterised generator for the same four
    typologies. Log-normal and beta laws replace the uniform ranges, a share
    of wallets is swept to an exactly zero balance, and the active span
    follows the 50-transaction observation window of the scraper."""
    c = _split_counts(n, 4)
    ntx = np.concatenate([_ln(rng, 300, 0.8, c[0], 40, 5000), _ln(rng, 1500, 0.8, c[1], 150, 20000),
                          _ln(rng, 25, 0.7, c[2], 4, 200), _ln(rng, 8, 0.5, c[3], 3, 40)])
    span = np.concatenate([_ln(rng, 12, 0.8, c[0], 1, 120), _ln(rng, 3, 0.6, c[1], 1, 20),
                           rng.randint(1, 5, c[2]).astype(float), _ln(rng, 700, 0.5, c[3], 180, 4000)])
    rec = np.concatenate([_ln(rng, 3, 1.2, c[0], 0.01, 5000), _ln(rng, 40, 1.3, c[1], 0.1, 50000),
                          _ln(rng, 80, 1.2, c[2], 1, 50000), _ln(rng, 3, 1.3, c[3], 0.01, 5000)])
    p_swept = np.repeat([0.5, 0.7, 0.8, 0.0], c)
    beta_b = np.repeat([60.0, 100.0, 50.0, 15.0], c)
    retained = rng.beta(1.0, beta_b)
    retained[rng.uniform(size=n) < p_swept] = 0.0
    typ = np.repeat(TYPOLOGIES, c)
    types, unc = _nuisance(n, rng, pool)
    return dict(n_tx=ntx, days=_window(span, ntx), rec=rec, sf=1.0 - retained,
                type=types, unc=unc, typ=typ)


def prims_heldout(n, rng, pool) -> dict:
    """H: four typologies that were not used anywhere in the submission
    and are never used for model or fusion selection."""
    c = _split_counts(n, 4)
    ntx = np.concatenate([rng.randint(2, 31, c[0]), rng.randint(300, 5001, c[1]),
                          rng.randint(30, 301, c[2]), np.full(c[3], 2)]).astype(float)
    span = np.concatenate([rng.randint(1, 61, c[0]), rng.randint(1, 21, c[1]),
                           rng.randint(10, 91, c[2]), np.ones(c[3])]).astype(float)
    rec = np.concatenate([10 ** rng.uniform(1.0, 3.3, c[0]), 10 ** rng.uniform(-3.0, -1.0, c[1]),
                          ntx[c[0] + c[1]:c[0] + c[1] + c[2]] * rng.uniform(0.05, 0.3, c[2]),
                          10 ** rng.uniform(0.7, 2.7, c[3])])
    sf = np.concatenate([rng.uniform(0.0, 0.05, c[0]), rng.uniform(0.9, 1.0, c[1]),
                         rng.uniform(0.95, 1.0, c[2]), np.ones(c[3])])
    typ = np.repeat(HELDOUT, c)
    types, unc = _nuisance(n, rng, pool)
    return dict(n_tx=ntx, days=_window(span, ntx), rec=rec, sf=sf, type=types, unc=unc, typ=typ)


# --- feasible, identity-preserving evasion operators (raw-field level) ---
def op_dilate(prim, f):
    """Spread the same activity over f times as many days."""
    out = dict(prim)
    out["days"] = prim["days"] * f
    return out


def op_retain(prim, rho):
    """Leave at least a fraction rho of the received value unspent."""
    out = dict(prim)
    out["sf"] = np.minimum(prim["sf"], 1.0 - rho)
    return out


def op_split(prim, m):
    """Spread the activity evenly over m addresses; one of them is scored."""
    out = dict(prim)
    out["n_tx"] = np.maximum(prim["n_tx"] / m, 1.0)
    out["rec"] = prim["rec"] / m
    return out


# --- noise-matched negatives: real wallets through the generator noise ---
def perturb_real_g1(real: pd.DataFrame, rng, noise=(0.8, 1.25)) -> pd.DataFrame:
    """Real wallets passed through the G1 noise path (independent noise on
    five fields and the G1 unconfirmed-count rule)."""
    df = real.copy()
    for c in ["n_tx", "total_received", "total_sent", "final_balance", "tx_per_day"]:
        df[c] = df[c].values * rng.uniform(noise[0], noise[1], len(df))
    df["n_tx"] = df["n_tx"].clip(lower=1).round().astype(int)
    df["unconfirmed_n_tx"] = rng.choice([0, 0, 0, 1], size=len(df))
    return df


def perturb_real_consistent(real: pd.DataFrame, rng, noise=(0.8, 1.25)) -> pd.DataFrame:
    """Real wallets perturbed with the same noise magnitudes but with every
    accounting identity preserved (the G2 noise path)."""
    df = real.copy()
    n = len(df)
    ntx0 = np.maximum(df["n_tx"].values.astype(float), 1.0)
    ntx = np.maximum(np.round(ntx0 * rng.uniform(noise[0], noise[1], n)), 1.0)
    rec0 = np.maximum(df["total_received"].values.astype(float), 0.0)
    ret0 = np.clip(np.maximum(df["final_balance"].values, 0.0) / (rec0 + EPS), 0.0, 1.0)
    rec = np.round(rec0 * rng.uniform(noise[0], noise[1], n), 8)
    ret = np.clip(ret0 * rng.uniform(noise[0], noise[1], n), 0.0, 1.0)
    bal = np.round(rec * ret, 8)
    df["tx_per_day"] = df["tx_per_day"].values * (ntx / ntx0)
    df["n_tx"] = ntx.astype(int)
    df["total_received"] = rec
    df["final_balance"] = bal
    df["balance"] = bal
    df["total_sent"] = np.round(rec - bal, 8)
    return df


# ----------------------------------------------------------------------
# 4. Detectors
# ----------------------------------------------------------------------
class DeepAutoencoder(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d_in, 32), nn.ReLU(),
                                     nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 6))
        self.decoder = nn.Sequential(nn.Linear(6, 16), nn.ReLU(),
                                     nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, d_in))

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder(X: np.ndarray, seed: int):
    set_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    model = DeepAutoencoder(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=AE_LR, weight_decay=AE_WD)
    loss_fn = nn.MSELoss()
    n = len(Xt)
    history = []
    for _ in range(AE_EPOCHS):
        model.train()
        perm = torch.randperm(n)
        ep = 0.0
        for i in range(0, n, AE_BATCH):
            idx = perm[i:i + AE_BATCH]
            batch = Xt[idx]
            opt.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            opt.step()
            ep += loss.item() * len(idx)
        history.append(ep / n)
    model.eval()
    return model, history


def ae_scores(model, X):
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32)
        z = model.encoder(Xt)
        recon = model.decoder(z).numpy()
    return np.mean((recon - X) ** 2, axis=1), z.numpy()


HEUR_W = dict(passthrough=0.30, low_retain=0.25, velocity=0.20, burst=0.15, high_vol=0.10)


def _heur_components(feats: pd.DataFrame) -> dict:
    return dict(passthrough=1.0 - np.abs(feats["flow_ratio"].values - 1.0),
                low_retain=-feats["retention_ratio"].values,
                velocity=feats["log_tx_per_day"].values,
                burst=feats["burst_index"].values,
                high_vol=feats["log_total_received"].values)


class Heuristic:
    """Rule-based red-flag score; the z-scores use fit-split statistics."""

    def fit(self, feats):
        comp = _heur_components(feats)
        self.mu = {k: v.mean() for k, v in comp.items()}
        self.sd = {k: v.std() + EPS for k, v in comp.items()}
        return self

    def score(self, feats):
        comp = _heur_components(feats)
        return sum(HEUR_W[k] * (comp[k] - self.mu[k]) / self.sd[k] for k in HEUR_W)


BASE = ["Heuristic", "OneClassSVM", "LOF", "PCA", "IsolationForest", "DeepAutoencoder"]


class Suite:
    """All base detectors, fitted on one reference set."""

    def __init__(self, seed: int):
        self.seed = seed

    def fit(self, X: np.ndarray, feats: pd.DataFrame):
        seed = self.seed
        rng = np.random.RandomState(seed)
        self.heur = Heuristic().fit(feats)
        sub = rng.choice(len(X), size=min(5000, len(X)), replace=False)
        self.ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1).fit(X[sub])
        self.lof = LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=NJ).fit(X)
        self.pca = PCA(n_components=8, random_state=seed).fit(X)
        self.iforest = IsolationForest(n_estimators=300, random_state=seed, n_jobs=NJ).fit(X)
        self.ae, self.ae_hist = train_autoencoder(X, seed)
        err, z = ae_scores(self.ae, X)
        self.casc = IsolationForest(n_estimators=300, random_state=seed, n_jobs=NJ).fit(
            np.c_[z, np.log1p(err)])
        ref = self.score(X, feats)
        ref["LOF"] = -self.lof.negative_outlier_factor_   # leave-one-out on the fit set
        self.ref_sorted = {m: np.sort(s) for m, s in ref.items()}
        self.ref_mu = {m: float(s.mean()) for m, s in ref.items()}
        self.ref_sd = {m: float(s.std() + EPS) for m, s in ref.items()}
        self.ref_scores = ref
        return self

    def score(self, X: np.ndarray, feats: pd.DataFrame) -> dict:
        """Raw anomaly scores, higher means more anomalous."""
        s = {}
        s["Heuristic"] = np.asarray(self.heur.score(feats), dtype=float)
        s["OneClassSVM"] = -self.ocsvm.decision_function(X)
        s["LOF"] = -self.lof.score_samples(X)
        rec = self.pca.inverse_transform(self.pca.transform(X))
        s["PCA"] = np.mean((rec - X) ** 2, axis=1)
        s["IsolationForest"] = -self.iforest.score_samples(X)
        err, z = ae_scores(self.ae, X)
        s["DeepAutoencoder"] = err
        s["Cascade"] = -self.casc.score_samples(np.c_[z, np.log1p(err)])
        return s

    def ranks(self, raw: dict) -> dict:
        """Rank normalisation of (possibly unseen) wallets: the empirical CDF
        of the reference scores evaluated at each wallet's score."""
        return {m: np.searchsorted(self.ref_sorted[m], v, side="right") / len(self.ref_sorted[m])
                for m, v in raw.items()}

    def zscores(self, raw: dict) -> dict:
        return {m: (v - self.ref_mu[m]) / self.ref_sd[m] for m, v in raw.items()}


def ecdf_ranks(ref_raw: dict, raw: dict) -> dict:
    """Rank normalisation against an explicit reference: the empirical CDF of
    the reference scores evaluated at each wallet's score. With the scores of
    wallets that were not used for fitting as reference (a calibration
    reference), the ranks of unseen ordinary wallets are uniform by
    construction, which the in-sample reference of Suite.ranks does not
    guarantee for a detector that fits its training data closely."""
    out = {}
    for m, v in raw.items():
        rs = np.sort(np.asarray(ref_raw[m], dtype=float))
        out[m] = np.searchsorted(rs, v, side="right") / len(rs)
    return out


def ref_zscores(ref_raw: dict, raw: dict) -> dict:
    return {m: (v - float(np.mean(ref_raw[m]))) / (float(np.std(ref_raw[m])) + EPS)
            for m, v in raw.items()}


def volume_scores(feats: pd.DataFrame) -> dict:
    """Label-free baselines that use no model: a wallet is ranked by its
    received value or by its transaction count alone."""
    return {"VolumeReceived": feats["log_total_received"].values.astype(float),
            "VolumeCount": feats["log_n_tx"].values.astype(float)}


# ----------------------------------------------------------------------
# 5. Score aggregation rules
# ----------------------------------------------------------------------
def gated(r_ae, r_if, tau):
    """Sigmoid-gated rule of the original submission (its Eqs. 6 and 7)."""
    a = expit((r_ae - r_if) / tau)
    return a * r_ae + (1.0 - a) * r_if


def fusion_scores(r: dict, z: dict) -> dict:
    """Every aggregation rule compared in the paper. r holds rank-normalised
    scores and z holds standardised raw scores."""
    a, f = r["DeepAutoencoder"], r["IsolationForest"]
    out = {
        "DRIFT": 0.5 * (a + f),                       # mean of ranks
        "MaxRank": np.maximum(a, f),
        "MinRank": np.minimum(a, f),
        "GeoMeanRank": np.sqrt(a * f),
        "WAvg_AE0.75": 0.75 * a + 0.25 * f,
        "WAvg_AE0.25": 0.25 * a + 0.75 * f,
        "ZMean": 0.5 * (z["DeepAutoencoder"] + z["IsolationForest"]),
        "ZMax": np.maximum(z["DeepAutoencoder"], z["IsolationForest"]),
        "All5MeanRank": np.mean([r[m] for m in ["OneClassSVM", "LOF", "PCA",
                                                "IsolationForest", "DeepAutoencoder"]], axis=0),
    }
    for tau in (0.5, 0.35, 0.2, 0.1):
        out[f"Gated_tau{tau}"] = gated(a, f, tau)
    return out


# ----------------------------------------------------------------------
# 6. Metrics
# ----------------------------------------------------------------------
def evaluate(s, y, typ=None, typ_names=(), budgets=BUDGETS, rng=None) -> dict:
    """Threshold-free metrics, the oracle F1, and fixed-budget metrics.
    A budget b means that the analyst reviews the top b fraction of the
    scored set. Ties are broken at random."""
    rng = rng or np.random.RandomState(0)
    s = np.asarray(s, dtype=float)
    y = np.asarray(y).astype(int)
    out = {"roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s))}
    prec, rec, _ = precision_recall_curve(y, s)
    out["oracle_f1"] = float(np.max(2 * prec * rec / (prec + rec + EPS)))
    order = np.lexsort((rng.random_sample(len(s)), -s))
    n_pos = int(y.sum())
    out["r_precision"] = float(y[order[:n_pos]].sum() / max(n_pos, 1))
    for b in budgets:
        k = max(int(round(b * len(s))), 1)
        top = np.zeros(len(s), dtype=bool)
        top[order[:k]] = True
        hits = int(y[top].sum())
        out[f"P@{b}"] = hits / k
        out[f"R@{b}"] = hits / max(n_pos, 1)
        if typ is not None:
            for t in typ_names:
                m = typ == t
                out[f"R@{b}|{t}"] = float(top[m].sum() / max(int(m.sum()), 1))
    return out
