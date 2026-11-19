"""
run_all.py  |  Every result of the revision from one deterministic run
=====================================================================
Sets the pinned floating-point environment of drift_core.PINNED_ENV for every
script, runs all experiments in two independent lanes, regenerates every table and
figure, and writes outputs/run_manifest.json (software, processor, settings, and
the duration and exit code of every step). Each step logs to outputs/logs/.

A verification run then repeats run_experiments.py into outputs/rerun_repeat/, and
verify_determinism.py compares it, value by value, with the main run.

Run:  python revise1/src/run_all.py
"""
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

SRC = Path(__file__).resolve().parent
REV = SRC.parent
OUT = REV / "outputs"
LOGS = OUT / "logs"
PINNED_ENV = {"MKL_CBWR": "AVX2", "ATEN_CPU_CAPABILITY": "avx2", "OPENBLAS_CORETYPE": "Haswell"}
SDN = ["../outputs/ofac_features_sdn_2026-11-12.csv", "sdn", "../ofac/lists/sanctioned_addresses_XBT.txt"]

LANE_INJECTED = [["run_experiments.py"], ["stats_tests.py"], ["separability.py"], ["roc_recheck.py"],
                 ["roc_decomposition.py"], ["run_extensions.py"], ["run_g2w.py"], ["run_variants.py"]]
LANE_EXTERNAL = [["ofac_compare_lists.py"], ["ofac_programs.py", "../ofac/SDN_ADVANCED.XML"],
                 ["ofac_external.py"] + SDN, ["ofac_external.py", "../outputs/ofac_features_v7_356.csv", "v7_356"],
                 ["ofac_recheck.py"], ["ofac_crossfit.py"] + SDN, ["babd_attach_labels.py"], ["babd_external.py"], ["pyod_external.py"],
                 ["adbench_repair.py"], ["variants_external.py"], ["ofac_figure_data.py", "sdn"]]
# The pre-specified feature revision (features_prespecification.md) runs after both lanes.
# Its scripts pin the BLAS and OpenMP pools to 3 threads, the count that joblib gave the
# workers of the 5-process pools of every other step on the 16-thread processor.
FEATURES = [["run_features.py"], ["features_external.py"]]
THREAD_ENV = {"OMP_NUM_THREADS": "3", "MKL_NUM_THREADS": "3", "OPENBLAS_NUM_THREADS": "3"}
FINAL = [["make_tables.py", "sdn"], ["make_tables_ext.py", "sdn"], ["make_tables_variants.py"],
         ["make_tables_text.py"], ["make_tables_features.py"], ["make_figures.py", "sdn"]]
VERIFY = [["run_experiments.py"]]

steps = []
lock = threading.Lock()


def run(cmd, extra_env=None, tag=""):
    env = dict(os.environ, **PINNED_ENV, PYTHONIOENCODING="utf-8", **(extra_env or {}))
    if cmd[0] in {c[0] for c in FEATURES} or cmd[0] == "make_tables_features.py":
        env.update(THREAD_ENV)
    name = "_".join(Path(c).stem for c in cmd[:1] + [c for c in cmd[1:] if not c.startswith("..")])
    name = re.sub(r"[^\w.-]+", "_", name + ("_verification" if extra_env else ""))
    t = time.time()
    with open(LOGS / f"{name}.log", "w", encoding="utf-8") as log:
        code = subprocess.call([sys.executable] + cmd, cwd=SRC, env=env, stdout=log, stderr=subprocess.STDOUT)
    rec = {"step": " ".join(cmd) + tag, "exit_code": code, "seconds": round(time.time() - t, 1),
           "finished": datetime.now().isoformat(timespec="seconds")}
    with lock:
        steps.append(rec)
        print(f"[{rec['finished']}] {rec['step']}: exit {code}, {rec['seconds']:.0f}s", flush=True)
    if code != 0:
        raise SystemExit(f"step failed: {rec['step']} (see outputs/logs/{name}.log)")


def lane(cmds, errors):
    try:
        for c in cmds:
            run(c)
    except SystemExit as e:
        errors.append(str(e))


def write_manifest(t0):
    versions = subprocess.run(
        [sys.executable, "-c", "import numpy,pandas,scipy,sklearn,torch,matplotlib,pyod,joblib,json;"
         "print(json.dumps({m.__name__: m.__version__ for m in (numpy,pandas,scipy,sklearn,torch,matplotlib,pyod,joblib)}))"],
        capture_output=True, text=True, env=dict(os.environ, **PINNED_ENV)).stdout
    manifest = {"started": datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
                "finished": datetime.now().isoformat(timespec="seconds"),
                "python": sys.version.split()[0], "packages": json.loads(versions),
                "processor": platform.processor(), "platform": platform.platform(),
                "pinned_environment": PINNED_ENV, "torch_threads_per_process": 2,
                "steps": sorted(steps, key=lambda s: s["finished"])}
    path = OUT / "run_manifest.json"
    if path.exists() and steps and all(s["step"].split()[0] in {c[0] for c in FINAL} for s in steps):
        old = json.loads(path.read_text())                   # --tables-only: keep the experiment steps
        old["steps"] += manifest["steps"]
        old["tables_regenerated"] = manifest["finished"]
        manifest = old
    path.write_text(json.dumps(manifest, indent=1))
    print(f"manifest written ({(time.time() - t0) / 60:.0f} min)", flush=True)


def resume_steps():
    """--resume: take the finished experiment steps from the log of the interrupted run
    (outputs/logs/run_all.log) and continue with the verification run."""
    pat = re.compile(r"^\[(?P<t>[\d\-T:]+)\] (?P<s>.+): exit (?P<c>-?\d+), (?P<sec>[\d.]+)s$")
    for line in (LOGS / "run_all.log").read_text(encoding="utf-8").splitlines():
        m = pat.match(line.strip())
        if m:
            steps.append({"step": m["s"], "exit_code": int(m["c"]), "seconds": float(m["sec"]),
                          "finished": m["t"], "from_log": "run_all.log"})


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if "--resume" in sys.argv:
        resume_steps()
        for c in VERIFY:
            run(c, {"DRIFT_OUT_DIR": str(OUT / "rerun_repeat")}, tag=" (verification run into outputs/rerun_repeat)")
        run(["verify_determinism.py"])
        write_manifest(t0)
    elif "--tables-only" not in sys.argv:
        errors = []
        threads = [threading.Thread(target=lane, args=(l, errors)) for l in (LANE_INJECTED, LANE_EXTERNAL)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        if errors:
            raise SystemExit("; ".join(errors))
        for c in FEATURES:
            run(c)
        for c in VERIFY:
            run(c, {"DRIFT_OUT_DIR": str(OUT / "rerun_repeat")}, tag=" (verification run into outputs/rerun_repeat)")
        run(["verify_determinism.py"])
        write_manifest(t0)
    for c in FINAL:
        run(c)
    write_manifest(t0)


if __name__ == "__main__":
    main()
