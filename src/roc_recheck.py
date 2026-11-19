"""roc_recheck.py  |  Why the ROC-AUC of DRIFT changed from the submitted version.

The submitted version reported a ROC-AUC of 0.952 for DRIFT; the revised paper
reports 0.927. Three things differ between the two numbers: the software
environment, the anomaly generator (G1 with three artefacts, G2 without), and the
evaluation protocol (transductive, with the injected wallets in the fitting sample,
against held-out, with a fit split that contains no injected wallet).

This script separates the three. It evaluates every combination of the three
generator artefacts under both protocols, so the change can be attributed:

  type   script types that are rare in the corpus (a third of the mixer wallets
         receive a type held by three corpus wallets)
  unc    a nonzero unconfirmed count in 25% of synthetic wallets (1.0% in the corpus)
  noise  independent noise on derived fields, which breaks accounting identities

All three artefacts on reproduces G1; all three off reproduces G2. Only the two
DRIFT components are fitted (Isolation Forest and autoencoder), with the
hyperparameters of the paper.

Run:  python revise1/src/roc_recheck.py
"""
import itertools
import json

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

import drift_core as dc

HELD_SEEDS = dc.SEEDS            # ten seeds, as in the paper
TRANS_SEEDS = dc.SEEDS[:5]       # five seeds, as in the submitted version
G1_TYPES = {"peeling_chain": ["P2WPKH", "Legacy", "P2SH"],
            "mixer_passthrough": ["P2WPKH", "P2SH", "Bech32_other"],
            "rapid_cashout": ["P2WPKH", "Taproot", "P2SH"],
            "dormant_burst": ["Legacy", "P2SH", "P2WPKH"]}
VARIANTS = list(itertools.product([True, False], repeat=3))      # (type, unc, noise)


def name(v):
    on = [n for n, f in zip(("type", "unc", "noise"), v) if f]
    return "+".join(on) if on else "none"


def gen_variant(n, rng, pool, type_art, unc_art, noise_art, noise=(0.8, 1.25)):
    """The four typologies of the paper with a chosen subset of the G1 artefacts."""
    c = dc._split_counts(n, 4)
    ntx = np.concatenate([rng.randint(120, 1600, c[0]), rng.randint(400, 5000, c[1]),
                          rng.randint(8, 90, c[2]), rng.randint(3, 22, c[3])]).astype(float)
    days = np.concatenate([rng.randint(2, 45, c[0]), rng.randint(1, 8, c[1]),
                           rng.randint(1, 3, c[2]), rng.randint(280, 1800, c[3])]).astype(float)
    rec = np.concatenate([10 ** rng.uniform(-0.6, 1.4, c[0]), 10 ** rng.uniform(0.3, 2.6, c[1]),
                          10 ** rng.uniform(1.0, 3.0, c[2]), 10 ** rng.uniform(-0.4, 1.8, c[3])])
    sf = np.concatenate([rng.uniform(0.965, 0.999, c[0]), rng.uniform(0.985, 1.0, c[1]),
                         rng.uniform(0.97, 1.0, c[2]), rng.uniform(0.90, 1.0, c[3])])
    typ = np.repeat(dc.TYPOLOGIES, c)
    f = lambda: rng.uniform(noise[0], noise[1], n)
    if type_art:
        types = np.array([rng.choice(G1_TYPES[t]) for t in typ], dtype=object)
    else:
        types = pool["address_type"].values[rng.randint(0, len(pool), n)]
    if unc_art:
        unc = rng.choice([0, 0, 0, 1], size=n)
    else:
        unc = (rng.uniform(size=n) < float((pool["unconfirmed_n_tx"].values > 0).mean())).astype(int)
    if not noise_art:                                  # identity-preserving noise (G2)
        retained = np.clip((1.0 - sf) * f(), 0.0, 1.0)
        return dc.assemble(dict(n_tx=ntx * f(), days=days, rec=rec * f(), sf=1.0 - retained,
                                type=types, unc=unc, typ=typ))
    # noise path of G1: derive the fields, then perturb each one independently
    rec = np.maximum(rec, 1e-6)
    sent = rec * sf
    bal = np.maximum(rec - sent, 0.0)
    days_i = np.maximum(days.astype(int), 1)
    tpd = ntx / days_i
    ntx_n = np.clip(np.round(ntx * f()), 1, None).astype(int)
    df = pd.DataFrame(dict(
        address=[f"SYN_{t}_{i}" for i, t in enumerate(typ)], address_type=types, n_tx=ntx_n,
        unconfirmed_n_tx=unc, final_n_tx=ntx_n, total_received=rec * f(), total_sent=sent * f(),
        balance=bal, unconfirmed_balance=np.zeros(n), final_balance=bal * f(), active_days=days_i,
        tx_per_day=tpd * f()))
    for col in dc._ZERO_COLS:
        df[col] = 0.0
    df["__label_typology"] = typ
    return df


class Light:
    """Preprocessing, Isolation Forest and autoencoder fitted on one reference set."""

    def __init__(self, ref_df, seed):
        torch.set_num_threads(2)
        F = dc.engineer_features(ref_df).values.astype(float)
        self.pre = dc.Preprocessor().fit(F)
        X = self.pre.transform(F)
        self.forest = IsolationForest(n_estimators=300, random_state=seed, n_jobs=2).fit(X)
        self.ae, _ = dc.train_autoencoder(X, seed)
        self.ref = {"IF": np.sort(-self.forest.score_samples(X)), "AE": np.sort(dc.ae_scores(self.ae, X)[0])}

    def scores(self, df):
        X = self.pre.transform(dc.engineer_features(df).values.astype(float))
        raw = {"IF": -self.forest.score_samples(X), "AE": dc.ae_scores(self.ae, X)[0]}
        r = {k: np.searchsorted(self.ref[k], v, side="right") / len(self.ref[k]) for k, v in raw.items()}
        return {"IF": raw["IF"], "AE": raw["AE"], "DRIFT": 0.5 * (r["IF"] + r["AE"])}


def metrics(s_neg, s_pos):
    y = np.r_[np.zeros(len(s_neg)), np.ones(len(s_pos))]
    s = np.r_[s_neg, s_pos]
    return {"roc": float(roc_auc_score(y, s)), "pr": float(average_precision_score(y, s))}


def held_out(seed, raw):
    rng = np.random.RandomState(seed)
    n = len(raw)
    idx = rng.permutation(n)
    n_fit, n_val = int(0.50 * n), int(0.15 * n)
    fit_df = raw.iloc[idx[:n_fit]].reset_index(drop=True)
    test_df = raw.iloc[idx[n_fit + n_val:]].reset_index(drop=True)
    model = Light(fit_df, seed)
    s_real = model.scores(test_df)
    n_an = int(round(len(test_df) * 0.05 / 0.95))
    out = {}
    for j, v in enumerate(VARIANTS):
        an = gen_variant(n_an, np.random.RandomState(seed * 1000 + 500 + j), fit_df, *v)
        s_an = model.scores(an)
        out[name(v)] = {m: metrics(s_real[m], s_an[m]) for m in ("IF", "AE", "DRIFT")}
    print(f"[held-out seed {seed}] " + "  ".join(f"{k}:{d['DRIFT']['roc']:.3f}" for k, d in out.items()), flush=True)
    return out


def transductive(seed, v, raw):
    rng = np.random.RandomState(seed)
    n_an = int(len(raw) * 0.05 / 0.95)
    syn = gen_variant(n_an, rng, raw, *v)
    real = raw.copy()
    real["__label_typology"] = "real"
    comb = pd.concat([real, syn], ignore_index=True, sort=False)
    comb = comb.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    pos = (comb["__label_typology"].values != "real")
    model = Light(comb.drop(columns="__label_typology"), seed)     # injected wallets are in the fitting sample
    s = model.scores(comb)
    out = {m: metrics(s[m][~pos], s[m][pos]) for m in ("IF", "AE", "DRIFT")}
    print(f"[transductive seed {seed} {name(v):15s}] DRIFT roc={out['DRIFT']['roc']:.3f}", flush=True)
    return name(v), out


def agg(list_of_dicts):
    return {m: {k: {"mean": float(np.mean([d[m][k] for d in list_of_dicts])),
                    "std": float(np.std([d[m][k] for d in list_of_dicts]))} for k in ("roc", "pr")}
            for m in ("IF", "AE", "DRIFT")}


def main():
    raw = dc.load_data()
    held = Parallel(n_jobs=5)(delayed(held_out)(s, raw) for s in HELD_SEEDS)
    res = {"held_out": {name(v): agg([h[name(v)] for h in held]) for v in VARIANTS}}
    jobs = [(s, v) for v in VARIANTS for s in TRANS_SEEDS]
    trans = Parallel(n_jobs=5)(delayed(transductive)(s, v, raw) for s, v in jobs)
    res["transductive"] = {name(v): agg([o for k, o in trans if k == name(v)]) for v in VARIANTS}
    (dc.OUT_DIR / "roc_recheck.json").write_text(json.dumps(res, indent=1))

    for proto, label in (("transductive", "SUBMITTED PROTOCOL (injected wallets in the fitting sample, 5 seeds)"),
                         ("held_out", "HELD-OUT PROTOCOL (fit split without injected wallets, 10 seeds)")):
        print(f"\n{label}")
        print(f"  {'artefacts present':18s} {'DRIFT ROC':>10s} {'DRIFT PR':>9s} {'IF ROC':>8s} {'AE ROC':>8s}")
        for v in VARIANTS:
            r = res[proto][name(v)]
            tag = "  (= G1)" if all(v) else "  (= G2)" if not any(v) else ""
            print(f"  {name(v):18s} {r['DRIFT']['roc']['mean']:10.3f} {r['DRIFT']['pr']['mean']:9.3f} "
                  f"{r['IF']['roc']['mean']:8.3f} {r['AE']['roc']['mean']:8.3f}{tag}")
    print(f"\n[save] {dc.OUT_DIR / 'roc_recheck.json'}")


if __name__ == "__main__":
    main()
