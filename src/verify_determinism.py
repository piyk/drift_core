"""
verify_determinism.py  |  Checks that the main run is reproduced exactly
=======================================================================
Compares every numeric value of run_experiments.py in outputs/ with

  outputs/rerun_repeat/            the verification run of run_all.py (same machine,
                                   pinned settings, separate processes);
  outputs/crosscheck_i7-6700K/     the earlier run of the same code on an Intel Core
                                   i7-6700K, a processor without AVX-512, whose
                                   default kernels are the pinned AVX2 kernels.

Writes outputs/determinism_check.json.
Run:  python revise1/src/verify_determinism.py
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "outputs"
FILES = ["experiments_per_seed.json", "transductive.json"]


def leaves(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f"{path}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{path}[{i}]")
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield path, float(x)


def compare(a_path, b_path, only=None):
    """Compare every numeric value of a with b; with only, the values whose path
    contains one of the given parts (experiments that both runs define identically)."""
    a = dict(leaves(json.load(open(a_path))))
    b = dict(leaves(json.load(open(b_path))))
    timing = lambda p: "seconds" in p or "fit_seconds" in p or "score_seconds" in p
    keys = [k for k in a if not timing(k) and (only is None or any(o in k for o in only))]
    missing = [k for k in keys if k not in b]
    diffs = [abs(a[k] - b[k]) for k in keys if k in b and a[k] != b[k]]
    return {"values_compared": len(keys) - len(missing), "missing_in_other": len(missing),
            "values_different": len(diffs), "max_abs_difference": max(diffs) if diffs else 0.0}


# The earlier i7-6700K run used G2 with the span over the whole activity, so it is compared
# only on the experiments that both runs define identically: G1, G3 and H, and the
# transductive protocol with G1.
SHARED = {"experiments_per_seed.json": ("/E2_G1/", "/E2_G3/", "/E2_H/"),
          "transductive.json": ("/G1/", "/G1_per_seed")}


def main():
    res = {}
    if (OUT / "rerun_repeat").exists():
        res["rerun_repeat"] = {f: compare(OUT / f, OUT / "rerun_repeat" / f) for f in FILES
                               if (OUT / "rerun_repeat" / f).exists()}
    if (OUT / "crosscheck_i7-6700K").exists():
        res["crosscheck_i7-6700K"] = {f: compare(OUT / f, OUT / "crosscheck_i7-6700K" / f, SHARED[f]) for f in FILES
                                      if (OUT / "crosscheck_i7-6700K" / f).exists()}
    (OUT / "determinism_check.json").write_text(json.dumps(res, indent=1))
    for other, d in res.items():
        for f, c in d.items():
            print(f"{other}/{f}: {c['values_compared']} values, {c['values_different']} different, "
                  f"max |diff| {c['max_abs_difference']:.3g}, missing {c['missing_in_other']}")


if __name__ == "__main__":
    main()
