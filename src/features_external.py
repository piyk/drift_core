"""
features_external.py  |  External checks of the pre-specified feature revision
===============================================================================
Specification: revise1/features_prespecification.md (criterion S2 and the external
measurements). Five seeds, detectors fitted on the whole corpus as in the manuscript
(in-sample design), and five-fold cross-fitting for the sanctioned addresses, for every
feature set (F21, W21, B, BR, BR-u); the four PyOD detectors in the in-sample design for
F21 and BR.

  sanctioned  527 addresses against the matched controls of the manuscript and against
              controls matched also on an empty balance; same-type share; P2WSH shares
  BABD-13     sample 1: illicit-associated against service addresses, overall and within
              P2PKH and P2SH
  services    sample 2: injected G2 wallets and sanctioned addresses against the verified
              high-activity services (criterion S2)

Run:  python revise1/src/features_external.py --check
      python revise1/src/features_external.py [--jobs N]
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.model_selection import KFold

import babd_external as be
import drift_core as dc
import features_v2 as fv
import ofac_external as oe

OUT = dc.OUT_DIR / "features"
PARTS = OUT / "external_parts"
SEEDS = oe.SEEDS
K_FOLDS = 5
BASE7 = dc.BASE + ["Cascade"]
NAMES = ["Heuristic", "OneClassSVM", "LOF", "PCA", "IsolationForest", "DeepAutoencoder",
         "DRIFT", "MinRank", "All5MeanRank"]
PYOD = ["ECOD", "COPOD", "DIF", "DeepSVDD"]
KEYS = ("median_pct", "top10", "top10_ctrl", "enrich10_matched", "p_outrank_matched")


def pct(ref, v):
    rs = np.sort(np.asarray(ref, dtype=float))
    return np.searchsorted(rs, v, side="right") / len(rs) * 100.0


def names_for(name, kind):
    out = list(NAMES) + (["DRIFT-S"] if name == "BR" else [])
    if kind == "insample" and name in ("F21", "BR"):
        out += PYOD
    return out


def stratified(ref_raw, ref_types, raw, types):
    """DRIFT-S against an explicit reference (cross-fitting): each component ranked among
    the reference wallets of the same script type, or the whole reference if the type
    has fewer than fv.MIN_TYPE of them."""
    ranks = []
    for m in ("DeepAutoencoder", "IsolationForest"):
        out = np.empty(len(types))
        whole = np.sort(ref_raw[m])
        for t in np.unique(types):
            rows = types == t
            rt = ref_raw[m][ref_types == t]
            rs = np.sort(rt) if len(rt) >= fv.MIN_TYPE else whole
            out[rows] = np.searchsorted(rs, raw[m][rows], side="right") / len(rs)
        ranks.append(out)
    return 0.5 * (ranks[0] + ranks[1])


# ----------------------------------------------------------------------
# jobs
# ----------------------------------------------------------------------
def insample_job(seed, name, corpus, allx, with_pyod=True):
    """Detectors fitted on the whole corpus; percentiles of the scored addresses (x) and
    of the corpus wallets themselves (c) in the corpus distribution."""
    torch.set_num_threads(2)
    sc = fv.fit_scorer(corpus, seed, name)
    ref = sc.reference()
    S, _, X = sc(allx)
    out = {}
    for m in NAMES:
        out[f"{m}|x"], out[f"{m}|c"] = pct(ref[m], S[m]), pct(ref[m], ref[m])
    if name == "BR":
        dref = sc.drift_s(sc.suite.ref_scores, fv.canonical_types(corpus))
        dx = sc.drift_s(S, fv.canonical_types(allx))
        out["DRIFT-S|x"], out["DRIFT-S|c"] = pct(dref, dx), pct(dref, dref)
    if name in ("F21", "BR") and with_pyod:
        Xc = sc.matrix(corpus)
        for p, clf in fv.fit_pyod(Xc, seed).items():
            r = fv.pyod_reference(clf, Xc)
            out[f"{p}|x"] = pct(r, np.asarray(clf.decision_function(X), dtype=float))
            out[f"{p}|c"] = pct(r, r)
    return out


def crossfit_job(seed, name, corpus, sanc):
    """Five-fold cross-fitting as in ofac_crossfit.py: each fold is scored by detectors
    fitted on the other four, and its out-of-sample scores are the reference."""
    torch.set_num_threads(2)
    names = names_for(name, "crossfit")
    Pc = {m: np.zeros(len(corpus)) for m in names}
    Ps = {m: np.zeros(len(sanc)) for m in names}
    ctypes, stypes = fv.canonical_types(corpus), fv.canonical_types(sanc)
    f21_c, f21_s = dc.engineer_features(corpus), dc.engineer_features(sanc)
    for tr, ho in KFold(n_splits=K_FOLDS, shuffle=True, random_state=seed).split(np.arange(len(corpus))):
        # As in ofac_crossfit.py, the feature matrix of the whole corpus is built once and its
        # rows are sliced. The slices are C-ordered, whereas a matrix built from the fold's own
        # frame is Fortran-ordered; the two orders sum the standardization statistics in a
        # different order (differences of about 1e-12), which the autoencoder amplifies.
        fmap = fv.FeatureMap(name).fit(corpus.iloc[tr].reset_index(drop=True))
        F_all, F_s = fmap.transform(corpus), fmap.transform(sanc)
        pre = fv.Prep(winsorize=(name != "BR-u")).fit(F_all[tr])
        suite = fv.SuiteV(seed, F_all.shape[1]).fit(pre.transform(F_all[tr]), f21_c.iloc[tr])
        Sh = suite.score(pre.transform(F_all[ho]), f21_c.iloc[ho])
        Ss = suite.score(pre.transform(F_s), f21_s)
        ref = {m: Sh[m] for m in BASE7}
        sh, ss = dict(Sh), dict(Ss)
        sh.update(dc.fusion_scores(dc.ecdf_ranks(ref, ref), dc.ref_zscores(ref, ref)))
        rs = {m: Ss[m] for m in BASE7}
        ss.update(dc.fusion_scores(dc.ecdf_ranks(ref, rs), dc.ref_zscores(ref, rs)))
        if name == "BR":
            sh["DRIFT-S"] = stratified(ref, ctypes[ho], ref, ctypes[ho])
            ss["DRIFT-S"] = stratified(ref, ctypes[ho], rs, stypes)
        for m in names:
            Pc[m][ho] = pct(sh[m], sh[m])
            Ps[m] += pct(sh[m], ss[m]) / K_FOLDS
    out = {}
    for m in names:
        out[f"{m}|x"], out[f"{m}|c"] = Ps[m], Pc[m]
    return out


def part_file(kind, name, seed):
    return PARTS / f"{kind}_{name}_{seed}.npz"


def run_and_save(kind, name, seed, corpus, allx, sanc):
    f = part_file(kind, name, seed)
    if f.exists():
        return
    t = time.time()
    out = insample_job(seed, name, corpus, allx) if kind == "insample" else crossfit_job(seed, name, corpus, sanc)
    tmp = f.with_name(f.stem + f".tmp{os.getpid()}.npz")
    np.savez_compressed(tmp, **out)
    os.replace(tmp, f)          # atomic: a second process working through the list never sees a partial file
    print(f"[external {kind} {name} seed {seed}] {time.time() - t:.0f}s", flush=True)


def load_parts(kind, name):
    runs = []
    for s in SEEDS:
        z = np.load(part_file(kind, name, s))
        runs.append({k: z[k] for k in z.files})
    return runs


# ----------------------------------------------------------------------
# summaries
# ----------------------------------------------------------------------
def ofac_block(Ps, Pc, idx):
    b = oe.stats_block(Ps, Pc, None, None, idx)
    return {k: b[k] for k in KEYS}


def ofac_summary(runs, names, ctrl, st, ct, profiles, rng, n_boot, sl=slice(None)):
    """runs hold percentiles of the scored addresses (x, restricted by sl to the sanctioned
    ones) and of the corpus (c)."""
    runs = [{k: (v[sl] if k.endswith("|x") else v) for k, v in r.items()} for r in runs]
    n = len(st)
    idx_all = np.arange(n)
    prof = np.asarray(profiles)
    uniq = np.unique(prof)
    members = {p: idx_all[prof == p] for p in uniq}
    boot_a = [rng.randint(0, n, n) for _ in range(n_boot)]
    boot_p = [np.concatenate([members[p] for p in rng.choice(uniq, size=len(uniq))]) for _ in range(n_boot)]
    out = {}
    for m in names:
        Ps = np.stack([r[f"{m}|x"] for r in runs])
        Pc = np.stack([r[f"{m}|c"][ctrl] for r in runs])
        point = ofac_block(Ps, Pc, idx_all)
        ci = {}
        for cname, boots in (("ci95_address", boot_a), ("ci95_profile", boot_p)):
            bs = [ofac_block(Ps, Pc, bi)["enrich10_matched"] for bi in boots]
            ci[cname] = [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]
        same = 0.0
        for t, w in pd.Series(st).value_counts(normalize=True).items():
            same += w * np.mean([(r[f"{m}|c"][ct == t] >= 90).mean() for r in runs])
        wsh = st == "P2WSH"
        out[m] = {"point": point, **ci, "top10_same_type": float(same),
                  "p2wsh_top10": float((Ps[:, wsh] >= 90).mean()),
                  "p2wsh_top10_ctrl": float((Pc[:, wsh, :] >= 90).mean()),
                  "by_type": {t: {"n": int((st == t).sum()), "top10": float((Ps[:, st == t] >= 90).mean()),
                                  "top10_ctrl": float((Pc[:, st == t, :] >= 90).mean())} for t in np.unique(st)}}
    return out


def summarize(corpus, sanc, lab, high, injected, sl, n_boot):
    ctrl, _, _, st, ct = oe.match_controls(corpus, sanc)
    ctrl_zb, _, _, _, _ = oe.match_controls(corpus, sanc, zero_balance=True)
    prog = json.load(open(dc.OUT_DIR / "ofac_programs.json"))["addresses"]
    profiles = [prog.get(a, {}).get("profile_id", a) for a in sanc["address"].astype(str)]
    fc = dc.engineer_features(corpus)
    vol_c = fc["log_total_received"].values
    res = {"spec_sha256": fv.spec_digest(), "seeds": list(SEEDS), "folds": K_FOLDS, "n_boot": n_boot,
           "n": {"sanctioned": int(len(sanc)), "lab": int(len(lab)), "high": int(len(high)),
                 "injected": int(len(injected))}, "sets": {}}
    ill = (lab["group"] == "illicit").values
    srv = ~ill
    ltype = lab["address_type"].map(dc.canonical_type).values
    for name in fv.SETS:
        ins = load_parts("insample", name)
        cvs = load_parts("crossfit", name)
        n_in, n_cv = names_for(name, "insample"), names_for(name, "crossfit")
        d = {"ofac_insample": ofac_summary(ins, n_in, ctrl, st, ct, profiles, np.random.RandomState(0), n_boot,
                                           sl["sanctioned"]),
             "ofac_crossfit": ofac_summary(cvs, n_cv, ctrl, st, ct, profiles, np.random.RandomState(0), n_boot),
             "ofac_crossfit_zero_balance": ofac_summary(cvs, n_cv, ctrl_zb, st, ct, profiles,
                                                        np.random.RandomState(0), n_boot)}
        # seed-mean percentiles for the labeled checks (in-sample design, as in the manuscript)
        P = {m: np.stack([r[f"{m}|x"] for r in ins]).mean(0) for m in n_in}
        allx_vol = np.r_[dc.engineer_features(sanc)["log_total_received"].values,
                         dc.engineer_features(lab)["log_total_received"].values,
                         dc.engineer_features(high)["log_total_received"].values,
                         dc.engineer_features(injected)["log_total_received"].values]
        P["VolumeReceived"] = pct(vol_c, allx_vol)
        rng = np.random.RandomState(0)
        babd = {"overall": {}, "within": {}}
        for m in P:
            p = P[m][sl["lab"]]
            babd["overall"][m] = {"auc": be.auc(p[ill], p[srv]), "ci95": be.boot_auc(p[ill], p[srv], rng, 1000)}
        for tt in ("Legacy", "P2SH"):
            mi, ms = ill & (ltype == tt), srv & (ltype == tt)
            babd["within"][tt] = {"n_illicit": int(mi.sum()), "n_service": int(ms.sum())}
            for m in P:
                p = P[m][sl["lab"]]
                babd["within"][tt][m] = {"auc": be.auc(p[mi], p[ms]), "ci95": be.boot_auc(p[mi], p[ms], rng, 1000)}
        d["babd_sample1"] = babd
        hs = {}
        for m in P:
            ph, pj, pz = P[m][sl["high"]], P[m][sl["injected"]], P[m][sl["sanctioned"]]
            hs[m] = {"top10_services": float((ph >= 90).mean()),
                     "auc_injected_vs_services": be.auc(pj, ph), "ci95_injected": be.boot_auc(pj, ph, rng, 1000),
                     "auc_sanctioned_vs_services": be.auc(pz, ph), "ci95_sanctioned": be.boot_auc(pz, ph, rng, 1000)}
        d["services"] = hs
        res["sets"][name] = d
    a_br = res["sets"]["BR"]["services"]["DRIFT"]["auc_injected_vs_services"]
    a_21 = res["sets"]["F21"]["services"]["DRIFT"]["auc_injected_vs_services"]
    res["criteria"] = {"S2": {"auc_BR": a_br, "auc_F21": a_21, "diff": a_br - a_21,
                              "holds": bool(a_br >= a_21 - 0.03)}}
    return res


# ----------------------------------------------------------------------
def load_groups():
    corpus = dc.load_data()
    in_corpus = set(corpus["address"].astype(str))
    sample = pd.read_csv(dc.OUT_DIR / "babd_sample.csv")
    lab = sample.merge(be.load_features(["babd_features.csv", "babd_features_bci.csv"]),
                       on="address", how="inner").drop_duplicates("address")
    lab = lab[~lab["address"].astype(str).isin(in_corpus)]
    lab = lab[lab["n_tx"] > 0].reset_index(drop=True)
    sh = pd.read_csv(dc.OUT_DIR / "babd_sample_high.csv")
    high = sh.merge(be.load_features(["babd_features_high.csv", "babd_features_high_bci.csv"]),
                    on="address", how="inner").drop_duplicates("address")
    high = high[~high["address"].astype(str).isin(in_corpus)]
    high = high[high["n_tx"] > 0].reset_index(drop=True)
    injected = dc.assemble(dc.prims_g2(be.N_INJECTED, np.random.RandomState(20261114), corpus))
    sanc = be.load_sanctioned(corpus)
    groups = [("sanctioned", sanc), ("lab", lab), ("high", high), ("injected", injected)]
    cols = [c for c in corpus.columns if all(c in g.columns for _, g in groups)]
    allx = pd.concat([g[cols] for _, g in groups], ignore_index=True)
    bounds = np.cumsum([0] + [len(g) for _, g in groups])
    sl = {nm: slice(bounds[i], bounds[i + 1]) for i, (nm, _) in enumerate(groups)}
    return corpus, sanc, lab, high, injected, allx, sl


def check(corpus, sanc, allx, sl):
    """(1) The F21 in-sample path equals ofac_external.fit_and_score for seed 11;
    (2) every set runs through both designs on a 6,000-wallet sub-corpus (finiteness only)."""
    seed = SEEDS[0]
    t = time.time()
    mine = insample_job(seed, "F21", corpus, sanc, with_pyod=False)
    ref = oe.fit_and_score(seed, corpus, sanc)
    diff = max(float(np.max(np.abs(mine[f"{m}|x"] - ref[m][0]))) for m in ("IsolationForest", "DeepAutoencoder",
                                                                         "DRIFT", "MinRank", "LOF"))
    diff = max(diff, max(float(np.max(np.abs(mine[f"{m}|c"] - ref[m][1]))) for m in ("DRIFT", "MinRank")))
    print(f"[check] F21 in-sample seed {seed}: max abs percentile difference to ofac_external {diff:.3g} "
          f"({time.time() - t:.0f}s)", flush=True)
    ok = diff == 0.0
    sub = corpus.sample(n=6000, random_state=0).reset_index(drop=True)
    small = allx.iloc[np.r_[0:30, sl["high"].start:sl["high"].start + 30]].reset_index(drop=True)
    for name in fv.SETS[1:]:
        a = insample_job(seed, name, sub, small)
        b = crossfit_job(seed, name, sub, small)
        fin = all(np.isfinite(v).all() for v in a.values()) and all(np.isfinite(v).all() for v in b.values())
        print(f"[check] {name}: in-sample keys {len(a)}, cross-fit keys {len(b)}, all finite: {fin}", flush=True)
        ok &= fin
    print(f"[check] passed: {ok}", flush=True)
    return ok


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    corpus, sanc, lab, high, injected, allx, sl = load_groups()
    print(f"corpus {len(corpus):,}; scored: sanctioned {len(sanc)}, lab {len(lab)}, high {len(high)}, "
          f"injected {len(injected)}; spec sha256 {fv.spec_digest()}", flush=True)
    if "--check" in sys.argv:
        sys.exit(0 if check(corpus, sanc, allx, sl) else 1)
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 4
    PARTS.mkdir(parents=True, exist_ok=True)
    todo = ([("insample", n, s) for n in ("BR", "F21") for s in SEEDS]
            + [("crossfit", n, s) for n in fv.SETS for s in SEEDS]
            + [("insample", n, s) for n in ("B", "W21", "BR-u") for s in SEEDS])
    if "--reverse" in sys.argv:         # a second process may work through the list from the end
        todo = todo[::-1]
    t = time.time()
    Parallel(n_jobs=jobs)(delayed(run_and_save)(k, n, s, corpus, allx, sanc) for k, n, s in todo)
    res = summarize(corpus, sanc, lab, high, injected, sl, oe.N_BOOT)
    res["seconds"] = time.time() - t
    (OUT / "features_external.json").write_text(json.dumps(res, indent=1))
    print(f"[save] {OUT / 'features_external.json'} ({time.time() - t:.0f}s)", flush=True)
    print("criteria:", res["criteria"], flush=True)


if __name__ == "__main__":
    main()
