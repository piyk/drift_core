"""ofac_figure_data.py  |  Data of the external-check figure.

For the figure that shows (a) the probability density of the DRIFT score for the
corpus wallets, the sanctioned addresses and their matched corpus controls, and
(b) the percentile ranks of the sanctioned addresses within the corpus.

DRIFT is fitted on the corpus only, with the five seeds and the settings of
ofac_external.py, so the top-decile share of panel (b) equals the top-10% share
reported in the table of the external check. Histograms are averaged over seeds.

Run:  python revise1/src/ofac_figure_data.py [tag]
"""
import json
import sys

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.ensemble import IsolationForest

import drift_core as dc
from ofac_external import match_controls
from ofac_recheck import LIST, load_sanctioned

SEEDS = dc.SEEDS[:5]
BINS = np.linspace(0.0, 1.0, 41)          # 40 bins of the DRIFT score
DECILES = np.arange(0, 101, 10)


def scores(seed, corpus, sanc):
    """DRIFT score of every corpus wallet and of every sanctioned address (as an
    unseen wallet), and their percentiles within the corpus."""
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

    d_c, d_s = drift(corpus), drift(sanc)
    ref = np.sort(d_c)
    pct = lambda d: np.searchsorted(ref, d, side="right") / len(ref) * 100.0
    return d_c, d_s, pct(d_c), pct(d_s)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "sdn"
    corpus = dc.load_data()
    if tag == "sdn":
        listed = {a.strip() for a in LIST.read_text().splitlines() if a.strip()}
        sanc = load_sanctioned(dc.OUT_DIR / "ofac_features_sdn_2026-11-12.csv", listed)
    else:
        sanc = load_sanctioned(dc.OUT_DIR / "ofac_features_v7_356.csv")
    ctrl = match_controls(corpus, sanc)[0]                       # [n_s, 5] corpus indices
    runs = Parallel(n_jobs=5)(delayed(scores)(s, corpus, sanc) for s in SEEDS)

    dens = lambda x: np.histogram(x, bins=BINS, density=True)[0]
    share = lambda p: np.histogram(np.clip(p, 0, 99.999), bins=DECILES)[0] / p.size * 100.0
    out = {
        "tag": tag, "n_corpus": int(len(corpus)), "n_sanctioned": int(len(sanc)),
        "n_controls": int(ctrl.size), "seeds": SEEDS,
        "score_bins": BINS.tolist(), "deciles": DECILES.tolist(),
        "density_corpus": np.mean([dens(r[0]) for r in runs], axis=0).tolist(),
        "density_sanctioned": np.mean([dens(r[1]) for r in runs], axis=0).tolist(),
        "density_controls": np.mean([dens(r[0][ctrl].ravel()) for r in runs], axis=0).tolist(),
        "decile_share_sanctioned": np.mean([share(r[3]) for r in runs], axis=0).tolist(),
        "decile_share_controls": np.mean([share(r[2][ctrl].ravel()) for r in runs], axis=0).tolist(),
        "corpus_p90_score": float(np.mean([np.quantile(r[0], 0.90) for r in runs])),
        "top10_sanctioned": float(np.mean([(r[3] >= 90).mean() for r in runs])),
        "top10_controls": float(np.mean([(r[2][ctrl] >= 90).mean() for r in runs])),
        "median_score_sanctioned": float(np.mean([np.median(r[1]) for r in runs])),
        "median_score_corpus": float(np.mean([np.median(r[0]) for r in runs])),
    }
    (dc.OUT_DIR / f"ofac_figure_{tag}.json").write_text(json.dumps(out, indent=1))
    print(f"n corpus {out['n_corpus']}, sanctioned {out['n_sanctioned']}, controls {out['n_controls']}")
    print(f"corpus 90th-percentile DRIFT score {out['corpus_p90_score']:.3f}")
    print(f"top-10% share: sanctioned {100 * out['top10_sanctioned']:.1f}%, controls {100 * out['top10_controls']:.1f}%")
    print("decile shares, sanctioned:", [round(v, 1) for v in out["decile_share_sanctioned"]])
    print("decile shares, controls:  ", [round(v, 1) for v in out["decile_share_controls"]])
    print(f"[save] {dc.OUT_DIR / f'ofac_figure_{tag}.json'}")


if __name__ == "__main__":
    main()
