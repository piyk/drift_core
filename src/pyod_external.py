"""pyod_external.py  |  External checks for the four PyOD detectors.

ECOD, COPOD, Deep Isolation Forest and Deep SVDD (PyOD defaults, as in
experiment E13) are fitted on the whole corpus and applied to the sanctioned
addresses, to the two BABD-13 samples and to injected G2 wallets, with the
matched controls, the statistics and the seeds of ofac_external.py and
babd_external.py. The reference distribution of a detector is its score on
the corpus wallets (the fit data), as for the other detectors.

ECOD and COPOD use run_extensions.HeldOutTail: tail probabilities from the corpus
only, so that a candidate address is scored independently of the others; their
reference distribution is the in-sample score of the corpus wallets. Deep Isolation
Forest and Deep SVDD score each wallet independently in PyOD itself.

Run:  python revise1/src/pyod_external.py
"""
import contextlib
import io
import json
import sys

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

import babd_external as be
import drift_core as dc
import ofac_external as oe
from run_extensions import pyod_models

NAMES = ["ECOD", "COPOD", "DIF", "DeepSVDD"]
N_BOOT = 2000


def fit_and_score(seed, corpus, cand):
    torch.set_num_threads(2)
    fc = dc.engineer_features(corpus)
    pre = dc.Preprocessor().fit(fc.values.astype(float))
    Xc = pre.transform(fc.values.astype(float))
    Xa = pre.transform(dc.engineer_features(cand).values.astype(float))
    out = {}
    for name, make in pyod_models(seed).items():
        dc.set_seed(seed)
        with contextlib.redirect_stdout(io.StringIO()):
            clf = make().fit(Xc)
        # reference: scores of the corpus wallets (in sample, as for the other detectors);
        # candidates: each scored on its own
        ref = (np.asarray(clf.reference_scores(), dtype=float) if hasattr(clf, "reference_scores")
               else np.asarray(clf.decision_function(Xc), dtype=float))
        s = np.asarray(clf.decision_function(Xa), dtype=float)
        rs = np.sort(ref)
        out[name] = (np.searchsorted(rs, s, side="right") / len(rs) * 100.0,
                     np.searchsorted(rs, ref, side="right") / len(rs) * 100.0)
        print(f"[seed {seed}] {name} done", flush=True)
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    corpus = dc.load_data()
    in_corpus = set(corpus["address"].astype(str))
    sanc = be.load_sanctioned(corpus)

    sample = pd.read_csv(dc.OUT_DIR / "babd_sample.csv")
    feats = be.load_features(["babd_features.csv", "babd_features_bci.csv"])
    lab = sample.merge(feats, on="address", how="inner").drop_duplicates("address")
    lab = lab[~lab["address"].astype(str).isin(in_corpus)]
    lab = lab[lab["n_tx"] > 0].reset_index(drop=True)
    sh = pd.read_csv(dc.OUT_DIR / "babd_sample_high.csv")
    feats_h = be.load_features(["babd_features_high.csv", "babd_features_high_bci.csv"])
    high = sh.merge(feats_h, on="address", how="inner").drop_duplicates("address")
    high = high[~high["address"].astype(str).isin(in_corpus)]
    high = high[high["n_tx"] > 0].reset_index(drop=True)
    injected = dc.assemble(dc.prims_g2(be.N_INJECTED, np.random.RandomState(20261114), corpus))

    groups = [("sanctioned", sanc), ("lab", lab), ("high", high), ("injected", injected)]
    cols = [c for c in corpus.columns if all(c in g.columns for _, g in groups)]
    allx = pd.concat([g[cols] for _, g in groups], ignore_index=True)
    bounds = np.cumsum([0] + [len(g) for _, g in groups])
    sl = {name: slice(bounds[i], bounds[i + 1]) for i, (name, _) in enumerate(groups)}
    print({k: v.stop - v.start for k, v in sl.items()}, flush=True)

    ctrl, dist, smd, st, ct = oe.match_controls(corpus, sanc)
    runs = Parallel(n_jobs=min(5, len(oe.SEEDS)))(delayed(fit_and_score)(s, corpus, allx) for s in oe.SEEDS)

    rng = np.random.RandomState(0)
    boot_idx = [rng.randint(0, len(sanc), len(sanc)) for _ in range(N_BOOT)]
    full_idx = np.arange(len(sanc))
    ill = (lab["group"] == "illicit").values
    res = {"seeds": list(oe.SEEDS), "n_sanctioned": int(len(sanc)), "n_lab": int(len(lab)),
           "n_high": int(len(high)), "n_injected": int(len(injected)), "detectors": {}}
    for m in NAMES:
        Ps = np.stack([r[m][0][sl["sanctioned"]] for r in runs])
        Pc = np.stack([r[m][1][ctrl] for r in runs])
        point = oe.stats_block(Ps, Pc, ct, st, full_idx)
        boots = [oe.stats_block(Ps, Pc, ct, st, bi) for bi in boot_idx]
        ci = {k: [float(np.nanpercentile([b[k] for b in boots], 2.5)),
                  float(np.nanpercentile([b[k] for b in boots], 97.5))]
              for k in ("top10", "enrich10_matched", "p_outrank_matched")}
        acc = 0.0
        for t, w in pd.Series(st).value_counts(normalize=True).items():
            acc += w * np.mean([(r[m][1][ct == t] >= 90).mean() for r in runs])
        P = np.stack([r[m][0] for r in runs]).mean(0)
        pl, ph, pi, psn = P[sl["lab"]], P[sl["high"]], P[sl["injected"]], P[sl["sanctioned"]]
        d = {"ofac": {"point": point, "ci95": ci, "top10_same_type": float(acc)},
             "babd": {"auc_illicit_vs_service": be.auc(pl[ill], pl[~ill]),
                      "auc_ci95": be.boot_auc(pl[ill], pl[~ill], rng)},
             "high": {"median_pct": float(np.median(ph)), "top10": float((ph >= 90).mean()),
                      "auc_injected_vs_high": be.auc(pi, ph), "auc_injected_ci95": be.boot_auc(pi, ph, rng),
                      "auc_sanctioned_vs_high": be.auc(psn, ph)}}
        res["detectors"][m] = d
        print(f"{m:9s} OFAC top10 {point['top10']:.3f} ctrl {point['top10_ctrl']:.3f} "
              f"enrich {point['enrich10_matched']:.2f} [{ci['enrich10_matched'][0]:.2f},{ci['enrich10_matched'][1]:.2f}] "
              f"P(out) {point['p_outrank_matched']:.3f} | BABD AUC {d['babd']['auc_illicit_vs_service']:.3f} "
              f"| injected>services {d['high']['auc_injected_vs_high']:.3f} services top10 {d['high']['top10']:.3f}",
              flush=True)
    (dc.OUT_DIR / "pyod_external.json").write_text(json.dumps(res, indent=1))
    print("[save] pyod_external.json", flush=True)


if __name__ == "__main__":
    main()
