"""babd_external.py  |  Labeled external checks on samples of BABD-13.

Two samples of strongly labeled addresses are used (babd_scrape.py).

  1. A stratified random sample of service or user classes (centralized
     exchange, mining pool, gambling, individual wallet) and of classes
     associated with illicit activity (darknet market, tumbler, blackmail,
     Ponzi scheme). Question: are the illicit-associated addresses ranked
     above the others?
  2. A random sample of exchange and gambling addresses with a total degree of
     at least 100 in the BABD-13 window: verified high-activity services.
     Question: do such services outrank the injected wallets of the benchmark
     and the sanctioned addresses? This replaces the presumed hard negatives of
     the injected benchmark by verified ones.

Detectors are fitted on the corpus only, as in the check on sanctioned
addresses, and every address is scored as an unseen wallet and placed at its
percentile in the corpus score distribution. Two label-free volume rankings
are reported next to the detectors.

Run:  python revise1/src/babd_external.py
"""
import json
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

import drift_core as dc
import ofac_external as oe

DETS = ["Heuristic", "OneClassSVM", "LOF", "PCA", "IsolationForest", "DeepAutoencoder",
        "DRIFT", "MinRank"]
VOL = ["VolumeReceived", "VolumeCount"]
N_BOOT = 2000
N_INJECTED = 1250
OFAC_FEATURES = dc.OUT_DIR / "ofac_features_sdn_2026-11-12.csv"
OFAC_LIST = dc.ROOT / "revise1" / "ofac" / "lists" / "sanctioned_addresses_XBT.txt"


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))


def boot_auc(pos, neg, rng, n=N_BOOT):
    bs = [auc(pos[rng.randint(0, len(pos), len(pos))], neg[rng.randint(0, len(neg), len(neg))]) for _ in range(n)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def load_features(names):
    parts = [pd.read_csv(dc.OUT_DIR / n) for n in names if (dc.OUT_DIR / n).exists()]
    if not parts:
        return None
    f = pd.concat(parts, ignore_index=True).drop_duplicates("address")
    for c in dc.NUMERIC_COLS:
        f[c] = pd.to_numeric(f[c], errors="coerce")
    return f


def load_sanctioned(corpus):
    s = pd.read_csv(OFAC_FEATURES)
    for c in dc.NUMERIC_COLS:
        s[c] = pd.to_numeric(s[c], errors="coerce")
    listed = [a.strip() for a in open(OFAC_LIST).read().splitlines() if a.strip()]
    valid = {a for a in listed if a.startswith(("1", "3")) or a.lower().startswith("bc1")}
    s = s[s["address"].astype(str).isin(valid)].drop_duplicates("address")
    s = s[~s["address"].astype(str).isin(set(corpus["address"].astype(str)))]
    return s[s["n_tx"] > 0].reset_index(drop=True)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    corpus = dc.load_data()
    in_corpus = set(corpus["address"].astype(str))

    sample = pd.read_csv(dc.OUT_DIR / "babd_sample.csv")
    feats = load_features(["babd_features.csv", "babd_features_bci.csv"])
    lab = sample.merge(feats, on="address", how="inner").drop_duplicates("address")
    n_retrieved = len(lab)
    lab = lab[~lab["address"].astype(str).isin(in_corpus)]
    lab = lab[lab["n_tx"] > 0].reset_index(drop=True)

    high = None
    feats_h = load_features(["babd_features_high.csv", "babd_features_high_bci.csv"])
    if feats_h is not None and (dc.OUT_DIR / "babd_sample_high.csv").exists():
        sh = pd.read_csv(dc.OUT_DIR / "babd_sample_high.csv")
        high = sh.merge(feats_h, on="address", how="inner").drop_duplicates("address")
        n_high_retrieved = len(high)
        high = high[~high["address"].astype(str).isin(in_corpus)]
        high = high[high["n_tx"] > 0].reset_index(drop=True)

    injected = dc.assemble(dc.prims_g2(N_INJECTED, np.random.RandomState(20261114), corpus))
    sanc = load_sanctioned(corpus)

    # one scoring pass per seed over all four groups
    groups = [("lab", lab), ("injected", injected), ("sanctioned", sanc)]
    if high is not None:
        groups.append(("high", high))
    cols = [c for c in corpus.columns if all(c in g.columns for _, g in groups)]
    allx = pd.concat([g[cols] for _, g in groups], ignore_index=True)
    bounds = np.cumsum([0] + [len(g) for _, g in groups])
    sl = {name: slice(bounds[i], bounds[i + 1]) for i, (name, _) in enumerate(groups)}
    runs = Parallel(n_jobs=min(5, len(oe.SEEDS)))(delayed(oe.fit_and_score)(s, corpus, allx) for s in oe.SEEDS)
    P = {m: np.stack([r[m][0] for r in runs]).mean(0) for m in DETS}          # seed-mean percentile
    fc, fa = dc.engineer_features(corpus), dc.engineer_features(allx)
    for v, col in (("VolumeReceived", "log_total_received"), ("VolumeCount", "log_n_tx")):
        ref = np.sort(fc[col].values)
        P[v] = np.searchsorted(ref, fa[col].values, side="right") / len(ref) * 100.0

    ill = (lab["group"] == "illicit").values
    srv = ~ill
    big = lab["n_tx"].values >= 100
    print(f"sample 1: sampled {len(sample)}, retrieved {n_retrieved}, scored {len(lab)}: "
          f"{int(ill.sum())} illicit-associated, {int(srv.sum())} service or user", flush=True)

    # matched service controls for every illicit-associated address
    cov = oe.covariates(lab)
    z = (cov - cov.mean(0)) / (cov.std(0) + dc.EPS)
    _, ind = NearestNeighbors(n_neighbors=5).fit(z[srv]).kneighbors(z[ill])
    srv_idx = np.where(srv)[0][ind]

    res = {"n_sampled": int(len(sample)), "n_retrieved": int(n_retrieved), "n_scored": int(len(lab)),
           "n_illicit": int(ill.sum()), "n_service": int(srv.sum()),
           "n_illicit_ge100": int((ill & big).sum()), "n_service_ge100": int((srv & big).sum()),
           "seeds": list(oe.SEEDS),
           "retrieved_utc": [str(lab["retrieved_utc"].min()), str(lab["retrieved_utc"].max())],
           "sources": lab["source"].value_counts().to_dict(),
           "class_counts": lab.groupby(["group", "class"]).size().reset_index(name="n").to_dict("records"),
           "describe": {g: {"median_n_tx": float(lab.loc[m, "n_tx"].median()),
                            "median_received_btc": float(lab.loc[m, "total_received"].median()),
                            "zero_balance_share": float((lab.loc[m, "final_balance"] < 1e-8).mean()),
                            "share_ntx_ge_100": float((lab.loc[m, "n_tx"] >= 100).mean())}
                        for g, m in (("illicit", ill), ("service", srv))},
           "detectors": {}, "by_class": {}}
    rng = np.random.RandomState(0)
    for m in DETS + VOL:
        p = P[m][sl["lab"]]
        pi, ps = p[ill], p[srv]
        res["detectors"][m] = {
            "auc_illicit_vs_service": auc(pi, ps), "auc_ci95": boot_auc(pi, ps, rng),
            "auc_matched_service": float((pi[:, None] > p[srv_idx]).mean() + 0.5 * (pi[:, None] == p[srv_idx]).mean()),
            "top10_illicit": float((pi >= 90).mean()), "top10_service": float((ps >= 90).mean()),
            "median_pct_illicit": float(np.median(pi)), "median_pct_service": float(np.median(ps))}
        d = res["detectors"][m]
        print(f"{m:16s} AUC illicit vs service {d['auc_illicit_vs_service']:.3f} "
              f"[{d['auc_ci95'][0]:.3f},{d['auc_ci95'][1]:.3f}]  matched {d['auc_matched_service']:.3f}  "
              f"top10 illicit {d['top10_illicit']:.3f} service {d['top10_service']:.3f}", flush=True)
    for cls, g in lab.groupby("class"):
        i = g.index.values
        res["by_class"][cls] = {"group": g["group"].iloc[0], "n": int(len(i)),
                                "median_n_tx": float(g["n_tx"].median()),
                                "median_received_btc": float(g["total_received"].median()),
                                **{m: {"median_pct": float(np.median(P[m][sl["lab"]][i])),
                                       "top10": float((P[m][sl["lab"]][i] >= 90).mean())} for m in DETS + VOL}}
    # Script type is nearly determined by the class in this sample, so the comparison
    # is repeated within each script type that holds at least 20 addresses of both groups.
    stype = lab["address_type"].map(dc.canonical_type).values
    res["script_type_by_class"] = {cls: g["address_type"].map(dc.canonical_type).value_counts(normalize=True).round(3).to_dict()
                                   for cls, g in lab.groupby("class")}
    res["script_type_by_group"] = {name: pd.Series(stype[m]).value_counts(normalize=True).round(3).to_dict()
                                   for name, m in (("illicit", ill), ("service", srv))}
    by_type = {}
    for t in np.unique(stype):
        mi, ms = ill & (stype == t), srv & (stype == t)
        if mi.sum() >= 20 and ms.sum() >= 20:
            by_type[t] = {"n_illicit": int(mi.sum()), "n_service": int(ms.sum()),
                          "classes_illicit": lab.loc[mi, "class"].value_counts().to_dict(),
                          "classes_service": lab.loc[ms, "class"].value_counts().to_dict(),
                          **{m: {"auc": auc(P[m][sl["lab"]][mi], P[m][sl["lab"]][ms]),
                                 "ci95": boot_auc(P[m][sl["lab"]][mi], P[m][sl["lab"]][ms], rng, 1000)}
                             for m in DETS + VOL}}
    res["auc_within_script_type"] = by_type
    # how the DRIFT percentile relates to simple properties of an address (Spearman correlation)
    pl = pd.Series(P["DRIFT"][sl["lab"]])
    res["drift_spearman"] = {
        "log_n_tx": float(pl.corr(pd.Series(np.log1p(lab["n_tx"].values)), method="spearman")),
        "log_received": float(pl.corr(pd.Series(np.log1p(lab["total_received"].values)), method="spearman")),
        "log_active_days": float(pl.corr(pd.Series(np.log1p(lab["active_days"].values)), method="spearman"))}
    res["drift_median_pct_by_script_type"] = {t: float(np.median(P["DRIFT"][sl["lab"]][stype == t]))
                                              for t in np.unique(stype)}
    (dc.OUT_DIR / "babd_per_address.json").write_text(json.dumps(
        {"address": lab["address"].tolist(), "class": lab["class"].tolist(), "group": lab["group"].tolist(),
         "script_type": list(stype), "n_tx": lab["n_tx"].astype(int).tolist(),
         **{m: np.round(P[m][sl["lab"]], 3).tolist() for m in DETS + VOL}}))
    small = [c for c, v in res["by_class"].items() if v["group"] == "illicit" and v["median_n_tx"] <= 2]
    res["illicit_small_classes"] = {"classes": small,
                                    "share_of_illicit": float(sum(res["by_class"][c]["n"] for c in small) / ill.sum())}

    # ---------------- sample 2: verified high-activity services ----------------
    if high is not None:
        hi = {"n_sampled": int(len(sh)), "n_retrieved": int(n_high_retrieved), "n_scored": int(len(high)),
              "class_counts": high.groupby("class").size().to_dict(),
              "median_n_tx": float(high["n_tx"].median()),
              "median_received_btc": float(high["total_received"].median()),
              "share_ntx_ge_100": float((high["n_tx"] >= 100).mean()),
              "share_ntx_ge_1000": float((high["n_tx"] >= 1000).mean()),
              "sanctioned_ge100": int((sanc["n_tx"].values >= 100).sum()),
              "n_injected": N_INJECTED, "detectors": {}}
        s_big = sanc["n_tx"].values >= 100
        for m in DETS + VOL:
            ph, pj, pz = P[m][sl["high"]], P[m][sl["injected"]], P[m][sl["sanctioned"]]
            hi["detectors"][m] = {
                "median_pct": float(np.median(ph)), "top10": float((ph >= 90).mean()),
                "top5": float((ph >= 95).mean()),
                "auc_injected_vs_service": auc(pj, ph), "auc_injected_ci95": boot_auc(pj, ph, rng),
                "auc_sanctioned_vs_service": auc(pz, ph), "auc_sanctioned_ci95": boot_auc(pz, ph, rng),
                "auc_sanctioned_ge100_vs_service": auc(pz[s_big], ph),
                "top10_injected": float((pj >= 90).mean())}
            d = hi["detectors"][m]
            print(f"[high] {m:16s} services: median pct {d['median_pct']:.1f} top10 {d['top10']:.3f} | "
                  f"AUC injected vs services {d['auc_injected_vs_service']:.3f} "
                  f"[{d['auc_injected_ci95'][0]:.3f},{d['auc_injected_ci95'][1]:.3f}] | "
                  f"AUC sanctioned vs services {d['auc_sanctioned_vs_service']:.3f} "
                  f"(sanctioned with >=100 tx: {d['auc_sanctioned_ge100_vs_service']:.3f})", flush=True)
        res["high_activity_services"] = hi
    (dc.OUT_DIR / "babd_external.json").write_text(json.dumps(res, indent=1))
    print(f"[save] {dc.OUT_DIR / 'babd_external.json'}", flush=True)


if __name__ == "__main__":
    main()
