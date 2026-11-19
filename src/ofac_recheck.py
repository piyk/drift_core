"""ofac_recheck.py  |  Why the sanctioned addresses rank high: an independent recheck.

The submitted version reported that DRIFT places about half of the sanctioned
addresses in the most anomalous 10% of the corpus. This script rechecks that
number and tests, with methods that do not rely on nearest-neighbour matching,
whether it reflects illicit behaviour or the size and format of the wallets.

  1. Reproduction on the 356 addresses of the submitted version and on the full
     sample of 527 addresses.
  2. Where the sanctioned addresses sit in the corpus distribution of volume.
  3. Stratification: share of ORDINARY corpus wallets in the DRIFT top 10%, by
     received value, next to the sanctioned addresses of the same stratum.
  4. Direct standardisation: the top-10% share expected for the sanctioned
     sample if each address behaved like a corpus wallet of its own stratum.
  5. A trivial baseline: ranking wallets by received value alone.
  6. Robustness of the matched-control result to the matching design.

Run:  python revise1/src/ofac_recheck.py
"""
import json

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors

import drift_core as dc

SEEDS = dc.SEEDS[:5]
OUT = dc.OUT_DIR
LIST = dc.ROOT / "revise1" / "ofac" / "lists" / "sanctioned_addresses_XBT.txt"


def load_sanctioned(path, restrict=None):
    s = pd.read_csv(path)
    for c in dc.NUMERIC_COLS:
        s[c] = pd.to_numeric(s[c], errors="coerce")
    if restrict is not None:
        s = s[s["address"].isin(restrict)]
    return s[s["n_tx"] > 0].drop_duplicates("address").reset_index(drop=True)


def drift_percentiles(seed, corpus, samples):
    """DRIFT fitted on the corpus only; percentile of every corpus wallet and of
    every address of each sample within the corpus DRIFT distribution."""
    torch.set_num_threads(2)
    feats = dc.engineer_features(corpus)
    pre = dc.Preprocessor().fit(feats.values.astype(float))
    X = pre.transform(feats.values.astype(float))
    forest = IsolationForest(n_estimators=300, random_state=seed, n_jobs=2).fit(X)
    ae, _ = dc.train_autoencoder(X, seed)
    ref_if = np.sort(-forest.score_samples(X))
    ref_ae = np.sort(dc.ae_scores(ae, X)[0])

    def drift(df):
        Xs = pre.transform(dc.engineer_features(df).values.astype(float))
        r_if = np.searchsorted(ref_if, -forest.score_samples(Xs), side="right") / len(ref_if)
        r_ae = np.searchsorted(ref_ae, dc.ae_scores(ae, Xs)[0], side="right") / len(ref_ae)
        return 0.5 * (r_if + r_ae)

    d_corpus = drift(corpus)
    ref = np.sort(d_corpus)
    pct = lambda d: np.searchsorted(ref, d, side="right") / len(ref) * 100.0
    return pct(d_corpus), {k: pct(drift(v)) for k, v in samples.items()}


def covs(df, cols):
    m = {"ntx": np.log1p(np.maximum(df["n_tx"].values.astype(float), 0)),
         "rec": np.log1p(np.maximum(df["total_received"].values.astype(float), 0) * 1e8),
         "span": np.log1p(np.maximum(df["active_days"].values.astype(float), 0))}
    return np.c_[tuple(m[c] for c in cols)]


def matched(corpus, sanc, cols, m=5, replace=True, caliper=None, zero_balance=False):
    """Indices of matched controls, [n_s, m]; -1 where no control qualifies. With
    zero_balance=True the exact-match strata are script type crossed with an empty balance."""
    ct = corpus["address_type"].map(dc.canonical_type).values
    st = sanc["address_type"].map(dc.canonical_type).values
    if zero_balance:
        empty = lambda df: np.where(df["final_balance"].values < 1e-8, "|empty", "|funded")
        ct, st = np.char.add(ct.astype(str), empty(corpus)), np.char.add(st.astype(str), empty(sanc))
    C, S = covs(corpus, cols), covs(sanc, cols)
    mu, sd = C.mean(0), C.std(0)
    Cz, Sz = (C - mu) / sd, (S - mu) / sd
    ctrl = -np.ones((len(sanc), m), dtype=int)
    for t in np.unique(st):
        ci, si = np.where(ct == t)[0], np.where(st == t)[0]
        if replace:
            d, ind = NearestNeighbors(n_neighbors=min(m, len(ci))).fit(Cz[ci]).kneighbors(Sz[si])
            for r, (dd, ii) in enumerate(zip(d, ind)):
                keep = ii if caliper is None else ii[dd <= caliper]
                ctrl[si[r], :len(keep)] = ci[keep]
        else:                                   # greedy 1:1 matching without replacement
            d, ind = NearestNeighbors(n_neighbors=min(50, len(ci))).fit(Cz[ci]).kneighbors(Sz[si])
            used = set()
            for r in np.argsort(d[:, 0]):
                for j in ind[r]:
                    if ci[j] not in used:
                        used.add(ci[j])
                        ctrl[si[r], 0] = ci[j]
                        break
    return ctrl


def main():
    corpus = dc.load_data()
    listed = {a.strip() for a in LIST.read_text().splitlines() if a.strip()}
    sdn = load_sanctioned(OUT / "ofac_features_sdn_2026-11-12.csv", listed)
    sub = load_sanctioned(OUT / "ofac_features_v7_356.csv")
    runs = Parallel(n_jobs=5)(delayed(drift_percentiles)(s, corpus, {"sdn": sdn, "sub": sub}) for s in SEEDS)
    Pc = np.stack([r[0] for r in runs])            # [seeds, N]
    Ps = {k: np.stack([r[1][k] for r in runs]) for k in ("sdn", "sub")}
    top = lambda P: float((P >= 90).mean())
    res = {}

    print("1. REPRODUCTION (DRIFT fitted on the corpus only, five seeds)")
    for k, name in (("sub", "356 addresses of the submitted version"), ("sdn", "full SDN sample")):
        res[f"repro_{k}"] = {"n": int(Ps[k].shape[1]),
                             "median_percentile": float(np.median(Ps[k], axis=1).mean()),
                             "top10": top(Ps[k])}
        print(f"   {name:42s} n={Ps[k].shape[1]:3d}  median percentile "
              f"{res[f'repro_{k}']['median_percentile']:.1f}  in top 10%: {100 * top(Ps[k]):.1f}%")
    print("   submitted version reported: median percentile 90.0, in top 10%: 50.0%")

    print("\n2. WHERE THE SANCTIONED ADDRESSES SIT IN THE CORPUS (received value, transaction count)")
    rec_c, rec_s = corpus["total_received"].values, sdn["total_received"].values
    q90, q50 = np.quantile(rec_c, 0.9), np.quantile(rec_c, 0.5)
    res["volume"] = {"corpus_median_btc": float(q50), "corpus_q90_btc": float(q90),
                     "sanctioned_median_btc": float(np.median(rec_s)),
                     "sanctioned_median_as_corpus_percentile": float((rec_c <= np.median(rec_s)).mean() * 100),
                     "sanctioned_above_corpus_q90": float((rec_s > q90).mean()),
                     "sanctioned_above_corpus_median": float((rec_s > q50).mean())}
    v = res["volume"]
    print(f"   corpus: median received {v['corpus_median_btc']:.4f} BTC, 90th percentile {v['corpus_q90_btc']:.2f} BTC")
    print(f"   sanctioned: median received {v['sanctioned_median_btc']:.1f} BTC = corpus percentile "
          f"{v['sanctioned_median_as_corpus_percentile']:.1f}")
    print(f"   sanctioned addresses above the corpus 90th percentile of received value: "
          f"{100 * v['sanctioned_above_corpus_q90']:.1f}% (10% expected for a random corpus wallet)")

    print("\n3. STRATIFICATION BY RECEIVED VALUE: share in the DRIFT top 10%")
    edges = [0, 0.01, 0.1, 1, 10, 100, np.inf]
    labels = ["< 0.01", "0.01-0.1", "0.1-1", "1-10", "10-100", ">= 100"]
    bc = np.digitize(rec_c, edges[1:-1])
    bs = np.digitize(rec_s, edges[1:-1])
    rows = []
    print(f"   {'received (BTC)':14s} {'corpus n':>9s} {'corpus top10':>13s} {'sanct. n':>9s} {'sanct. top10':>13s}")
    for b, lab in enumerate(labels):
        mc, ms = bc == b, bs == b
        row = {"bin": lab, "corpus_n": int(mc.sum()), "corpus_top10": float((Pc[:, mc] >= 90).mean()) if mc.any() else None,
               "sanctioned_n": int(ms.sum()), "sanctioned_top10": float((Ps["sdn"][:, ms] >= 90).mean()) if ms.any() else None}
        rows.append(row)
        f = lambda x: "     -" if x is None else f"{100 * x:6.1f}%"
        print(f"   {lab:14s} {row['corpus_n']:9d} {f(row['corpus_top10']):>13s} {row['sanctioned_n']:9d} {f(row['sanctioned_top10']):>13s}")
    res["strata_received"] = rows

    print("\n4. DIRECT STANDARDISATION: top-10% share expected if every sanctioned address")
    print("   behaved like an ordinary corpus wallet of its own stratum")
    ct = corpus["address_type"].map(dc.canonical_type).values
    st = sdn["address_type"].map(dc.canonical_type).values
    ntx_edges = [3, 10, 100, 1000]
    nc, ns = np.digitize(corpus["n_tx"].values, ntx_edges), np.digitize(sdn["n_tx"].values, ntx_edges)
    in_top_c = (Pc >= 90).mean(axis=0)              # per corpus wallet, averaged over seeds
    designs = {"received value": (bc, bs),
               "script type": (ct, st),
               "script type x received value": (np.char.add(ct.astype(str), bc.astype(str)), np.char.add(st.astype(str), bs.astype(str))),
               "script type x received value x transaction count":
                   (np.char.add(np.char.add(ct.astype(str), bc.astype(str)), nc.astype(str)),
                    np.char.add(np.char.add(st.astype(str), bs.astype(str)), ns.astype(str)))}
    res["standardised"] = {}
    for name, (kc, ks) in designs.items():
        kc, ks = np.asarray(kc).astype(str), np.asarray(ks).astype(str)
        table = pd.Series(in_top_c).groupby(kc).mean()
        exp = pd.Series(ks).map(table)
        cover = float(exp.notna().mean())
        res["standardised"][name] = {"expected_top10": float(exp.mean()), "covered": cover}
        print(f"   strata = {name:52s} expected {100 * exp.mean():5.1f}%   (strata found for {100 * cover:.0f}% of addresses)")
    print(f"   observed for the sanctioned addresses: {100 * top(Ps['sdn']):.1f}%   whole corpus: 10.0%")

    print("\n5. TRIVIAL BASELINE: rank wallets by received value alone (no model)")
    pct_rec = np.searchsorted(np.sort(rec_c), rec_s, side="right") / len(rec_c) * 100
    res["volume_only"] = {"median_percentile": float(np.median(pct_rec)), "top10": float((pct_rec >= 90).mean())}
    print(f"   sanctioned addresses: median percentile {np.median(pct_rec):.1f}, in the top 10% by received value: "
          f"{100 * (pct_rec >= 90).mean():.1f}%")

    print("\n6. ROBUSTNESS OF THE MATCHED CONTROLS (DRIFT, full SDN sample)")
    obs = top(Ps["sdn"])
    res["matching"] = {}
    variants = {
        "5 nearest, type + count + value + span (paper)": dict(cols=("ntx", "rec", "span"), m=5),
        "1 nearest, same covariates": dict(cols=("ntx", "rec", "span"), m=1),
        "10 nearest, same covariates": dict(cols=("ntx", "rec", "span"), m=10),
        "1:1 without replacement": dict(cols=("ntx", "rec", "span"), m=1, replace=False),
        "5 nearest within caliper 0.25 SD": dict(cols=("ntx", "rec", "span"), m=5, caliper=0.25),
        "5 nearest, type + value only": dict(cols=("rec",), m=5),
        "5 nearest, type + value + count": dict(cols=("ntx", "rec"), m=5),
        "5 nearest, type x empty + count + value + span": dict(cols=("ntx", "rec", "span"), m=5,
                                                               zero_balance=True),
    }
    for name, kw in variants.items():
        ctrl = matched(corpus, sdn, **kw)
        ok = ctrl >= 0
        has = ok.any(axis=1)
        share_c = float(np.mean([(Pc[s][ctrl[ok]] >= 90).mean() for s in range(len(SEEDS))]))
        share_s = float((Ps["sdn"][:, has] >= 90).mean())
        distinct = len(np.unique(ctrl[ok]))
        res["matching"][name] = {"sanctioned_top10": share_s, "controls_top10": share_c,
                                 "matched_enrichment": share_s / share_c, "addresses_matched": int(has.sum()),
                                 "distinct_controls": distinct}
        print(f"   {name:48s} sanctioned {100 * share_s:5.1f}%  controls {100 * share_c:5.1f}%  "
              f"ratio {share_s / share_c:.2f}  (matched {has.sum()}, distinct controls {distinct})")
    print(f"   observed top-10% share of all sanctioned addresses: {100 * obs:.1f}%")

    (OUT / "ofac_recheck.json").write_text(json.dumps(res, indent=1))
    print(f"\n[save] {OUT / 'ofac_recheck.json'}")


if __name__ == "__main__":
    main()
