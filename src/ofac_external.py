"""
ofac_external.py  |  Positive-only external check on OFAC-listed addresses
===========================================================================
Every detector is fitted on the unlabeled corpus only. Each sanctioned
address is then scored as an unseen wallet and placed at its percentile in
the corpus score distribution (reference-ECDF rank normalisation).

Reported for every detector, with bootstrap confidence intervals over
addresses and averaging over five model seeds:
  * median and mean percentile, share in the corpus top 1%, 5% and 10%;
  * the same shares for matched control wallets drawn from the corpus
    (same script type; nearest neighbours in transaction count, received
    value and active span), and the resulting matched enrichment;
  * the probability that a sanctioned address outranks its matched control.

Run:  python revise1/src/ofac_external.py <features.csv> <tag>
"""
import json
import sys

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.neighbors import NearestNeighbors

import drift_core as dc

DETS = ["Heuristic", "OneClassSVM", "LOF", "PCA", "IsolationForest", "DeepAutoencoder",
        "Cascade", "DRIFT", "MaxRank", "MinRank"]
SEEDS = dc.SEEDS[:5]
M_CTRL = 5
N_BOOT = 2000
COVS = ["log_n_tx", "log_total_received", "log_active_days"]


def fit_and_score(seed, corpus, sanc):
    torch.set_num_threads(2)
    feats = dc.engineer_features(corpus)
    F = feats.values.astype(float)
    pre = dc.Preprocessor().fit(F)
    suite = dc.Suite(seed).fit(pre.transform(F), feats)
    ref_raw = suite.ref_scores
    ref = dict(ref_raw)
    ref.update(dc.fusion_scores(suite.ranks(ref_raw), suite.zscores(ref_raw)))
    fs = dc.engineer_features(sanc)
    raw = suite.score(pre.transform(fs.values.astype(float)), fs)
    sc = dict(raw)
    sc.update(dc.fusion_scores(suite.ranks(raw), suite.zscores(raw)))
    out = {}
    for m in DETS:
        rs = np.sort(ref[m])
        out[m] = (np.searchsorted(rs, sc[m], side="right") / len(rs) * 100.0,
                  np.searchsorted(rs, ref[m], side="right") / len(rs) * 100.0)
    print(f"[seed {seed}] fitted", flush=True)
    return out


def covariates(df):
    return np.c_[np.log1p(np.maximum(df["n_tx"].values.astype(float), 0)),
                 np.log1p(np.maximum(df["total_received"].values.astype(float), 0) * 1e8),
                 np.log1p(np.maximum(df["active_days"].values.astype(float), 0))]


def match_controls(corpus, sanc, m=M_CTRL, zero_balance=False):
    """Nearest-neighbour matching with replacement inside each script type. With
    zero_balance=True the strata are script type crossed with an empty balance, a
    sensitivity check for the different observation times: the corpus wallets were
    sampled while active, the sanctioned addresses were observed years after their
    activity and are more often empty."""
    ct = corpus["address_type"].map(dc.canonical_type).values
    st = sanc["address_type"].map(dc.canonical_type).values
    if zero_balance:
        empty = lambda df: np.where(df["final_balance"].values < 1e-8, "|empty", "|funded")
        ct, st = np.char.add(ct.astype(str), empty(corpus)), np.char.add(st.astype(str), empty(sanc))
    C, S = covariates(corpus), covariates(sanc)
    mu, sd = C.mean(0), C.std(0)
    Cz, Sz = (C - mu) / sd, (S - mu) / sd
    ctrl = np.zeros((len(sanc), m), dtype=int)
    dist = np.zeros((len(sanc), m))
    for t in np.unique(st):
        ci, si = np.where(ct == t)[0], np.where(st == t)[0]
        k = min(m, len(ci))
        d, ind = NearestNeighbors(n_neighbors=k).fit(Cz[ci]).kneighbors(Sz[si])
        ind = np.resize(ind, (len(si), m)) if k == m else np.tile(ind, (1, m))[:, :m]
        d = d if k == m else np.tile(d, (1, m))[:, :m]
        ctrl[si], dist[si] = ci[ind], d
    smd = {}
    for j, name in enumerate(COVS):
        pooled = np.sqrt(0.5 * (S[:, j].var() + C[:, j].var()))
        smd[name] = {"before": float((S[:, j].mean() - C[:, j].mean()) / pooled),
                     "after": float((S[:, j].mean() - C[ctrl][:, :, j].mean()) / pooled)}
    return ctrl, dist, smd, st, ct


def stats_block(Ps, Pc, ct, st, idx):
    """Ps: [seeds, n_s] sanctioned percentiles; Pc: [seeds, n_s, m] matched
    control percentiles; idx: resampled address indices."""
    ps, pc = Ps[:, idx], Pc[:, idx, :]
    out = {"median_pct": float(np.median(ps, axis=1).mean()),
           "mean_pct": float(ps.mean())}
    for k in (1, 5, 10):
        s_share = float((ps >= 100 - k).mean())
        c_share = float((pc >= 100 - k).mean())
        out[f"top{k}"] = s_share
        out[f"top{k}_ctrl"] = c_share
        out[f"enrich{k}_corpus"] = s_share / (k / 100.0)
        out[f"enrich{k}_matched"] = s_share / c_share if c_share > 0 else float("nan")
    gt = (ps[:, :, None] > pc).mean() + 0.5 * (ps[:, :, None] == pc).mean()
    out["p_outrank_matched"] = float(gt)
    out["median_pct_ctrl"] = float(np.median(pc.reshape(pc.shape[0], -1), axis=1).mean())
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    feat_path, tag = sys.argv[1], sys.argv[2]
    list_path = sys.argv[3] if len(sys.argv) > 3 else None
    corpus = dc.load_data()
    sanc = pd.read_csv(feat_path)
    for c in dc.NUMERIC_COLS:
        sanc[c] = pd.to_numeric(sanc[c], errors="coerce")
    list_info = {}
    if list_path:
        # restrict to the addresses of the given list (the SDN-derived list)
        listed = [a.strip() for a in open(list_path).read().splitlines() if a.strip()]
        valid = [a for a in listed if a.startswith(("1", "3")) or a.lower().startswith("bc1")]
        have = set(sanc["address"].astype(str))
        list_info = {"n_list_entries": len(listed), "n_bitcoin_addresses": len(valid),
                     "non_bitcoin_entries": sorted(set(listed) - set(valid)),
                     "n_with_features": len(set(valid) & have),
                     "without_features": sorted(set(valid) - have),
                     "features_not_on_list": sorted(have - set(valid))}
        sanc = sanc[sanc["address"].astype(str).isin(set(valid))].reset_index(drop=True)
        print(f"list: {len(listed)} entries, {len(valid)} Bitcoin addresses, "
              f"{list_info['n_with_features']} with features, "
              f"{len(list_info['without_features'])} without", flush=True)
    n_listed = len(sanc)
    sanc = sanc.drop_duplicates("address")
    n_dup = n_listed - len(sanc)
    overlap = set(sanc["address"].astype(str)) & set(corpus["address"].astype(str))
    sanc = sanc[~sanc["address"].astype(str).isin(overlap)]
    n_inactive = int((sanc["n_tx"] <= 0).sum())
    sanc = sanc[sanc["n_tx"] > 0].reset_index(drop=True)
    print(f"sanctioned scored: {len(sanc)} (duplicates {n_dup}, overlap with corpus "
          f"{len(overlap)}, no on-chain activity {n_inactive})", flush=True)

    ctrl, dist, smd, st, ct = match_controls(corpus, sanc)
    runs = Parallel(n_jobs=min(5, len(SEEDS)))(delayed(fit_and_score)(s, corpus, sanc) for s in SEEDS)

    rng = np.random.RandomState(0)
    boot_idx = [rng.randint(0, len(sanc), len(sanc)) for _ in range(N_BOOT)]
    full_idx = np.arange(len(sanc))
    res = {"tag": tag, "n_scored": int(len(sanc)), "n_duplicates_removed": int(n_dup),
           "n_overlap_with_corpus": int(len(overlap)), "n_without_activity": n_inactive,
           "corpus_size": int(len(corpus)), "seeds": SEEDS, "controls_per_address": M_CTRL,
           "smd": smd, "match_distance_median": float(np.median(dist)),
           "script_types_scored": pd.Series(st).value_counts().to_dict(),
           "script_types_corpus": pd.Series(ct).value_counts().to_dict(),
           "list": list_info,
           "retrieved_utc": ([str(sanc["retrieved_utc"].min()), str(sanc["retrieved_utc"].max())]
                             if "retrieved_utc" in sanc.columns else None),
           "detectors": {}}
    desc = lambda df: {"median_n_tx": float(df["n_tx"].median()),
                       "median_received_btc": float(df["total_received"].median()),
                       "median_active_days": float(df["active_days"].median()),
                       "zero_balance_share": float((df["final_balance"] < 1e-8).mean()),
                       "share_ntx_le_2": float((df["n_tx"] <= 2).mean())}
    res["describe"] = {"sanctioned": desc(sanc), "corpus": desc(corpus),
                       "matched_controls": desc(corpus.iloc[ctrl.ravel()])}

    per_address = {"address": sanc["address"].tolist(), "script_type": list(st),
                   "n_tx": sanc["n_tx"].astype(int).tolist()}
    for m in DETS:
        Ps = np.stack([r[m][0] for r in runs])                 # [seeds, n_s]
        Pc = np.stack([r[m][1][ctrl] for r in runs])           # [seeds, n_s, m]
        point = stats_block(Ps, Pc, ct, st, full_idx)
        boots = [stats_block(Ps, Pc, ct, st, bi) for bi in boot_idx]
        ci = {k: [float(np.nanpercentile([b[k] for b in boots], 2.5)),
                  float(np.nanpercentile([b[k] for b in boots], 97.5))] for k in point}
        # same-script-type reference: corpus wallets weighted by the sanctioned type mix
        type_share = {}
        for k in (1, 5, 10):
            acc = 0.0
            for t, w in pd.Series(st).value_counts(normalize=True).items():
                acc += w * np.mean([(r[m][1][ct == t] >= 100 - k).mean() for r in runs])
            type_share[f"top{k}_same_type"] = float(acc)
            type_share[f"enrich{k}_same_type"] = point[f"top{k}"] / acc if acc > 0 else float("nan")
        d = Ps.mean(0) - Pc.mean(axis=(0, 2))
        try:
            w_p = float(wilcoxon(d).pvalue)
        except ValueError:
            w_p = float("nan")
        by_type = {t: {"n": int((st == t).sum()),
                       "median_pct": float(np.median(Ps[:, st == t], axis=1).mean()),
                       "top10": float((Ps[:, st == t] >= 90).mean()),
                       "top10_ctrl": float((Pc[:, st == t, :] >= 90).mean())}
                   for t in np.unique(st)}
        res["detectors"][m] = {"point": point, "ci95": ci, **type_share,
                               "paired_median_diff_pct": float(np.median(d)),
                               "wilcoxon_p": w_p, "by_type": by_type}
        per_address[m] = Ps.mean(0).round(3).tolist()
        per_address[m + "_ctrl"] = Pc.mean(axis=(0, 2)).round(3).tolist()
        per_address[m + "_ctrl_all"] = Pc.mean(axis=0).round(3).ravel().tolist()
        print(f"{m:16s} median {point['median_pct']:.1f}  top10 {point['top10']:.3f} "
              f"[{ci['top10'][0]:.3f},{ci['top10'][1]:.3f}]  ctrl-top10 {point['top10_ctrl']:.3f}  "
              f"matched-enrich {point['enrich10_matched']:.2f} "
              f"[{ci['enrich10_matched'][0]:.2f},{ci['enrich10_matched'][1]:.2f}]  "
              f"P(outrank) {point['p_outrank_matched']:.3f}", flush=True)

    # how many sanctioned addresses rank above the median corpus wallet (DRIFT)
    res["drift_above_corpus_median"] = int((np.array(per_address["DRIFT"]) > 50).sum())
    low = np.argsort(per_address["DRIFT"])[:max(len(sanc) // 10, 1)]
    res["drift_lowest_decile"] = {"median_n_tx": float(sanc["n_tx"].values[low].mean()),
                                  "share_ntx_le_2": float((sanc["n_tx"].values[low] <= 2).mean())}
    (dc.OUT_DIR / f"ofac_external_{tag}.json").write_text(json.dumps(res, indent=1))
    (dc.OUT_DIR / f"ofac_per_address_{tag}.json").write_text(json.dumps(per_address))
    print(f"[save] {dc.OUT_DIR / f'ofac_external_{tag}.json'}", flush=True)


if __name__ == "__main__":
    main()
