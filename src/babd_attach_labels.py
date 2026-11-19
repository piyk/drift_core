"""babd_attach_labels.py  |  Restore the BABD-13 labels of the two labeled samples.

The supplementary package lists the addresses of the two BABD-13 samples
(outputs/babd_sample.csv and outputs/babd_sample_high.csv) without their labels,
because the licence of BABD-13 (CC BY-NC-ND 4.0) does not permit redistributing a
modified copy of its content. This script looks the addresses up in the BABD-13 file
and writes the label columns back, with the rules by which babd_scrape.py drew the
samples: the first strongly labeled row (SW = SA) of each address and, for the
high-activity sample, the first such row with a total degree (PDIa1-3) of at least 100.
The restored files are identical to the ones the reported results were computed from.

Place BABD-13.csv (https://www.kaggle.com/datasets/lemonx/babd13) in
revise1/external/babd13/ before running. Files that already hold their labels are
left unchanged. run_all.py runs this step before the labeled external checks.

Run:  python revise1/src/babd_attach_labels.py
"""
import sys
from pathlib import Path

import pandas as pd

import babd_scrape as bs

COLUMNS = {False: ["address", "SW", "label", "class", "group"],
           True: ["address", "SW", "babd_degree", "label", "class", "group"]}


def restore(path: Path, high: bool, out: Path = None) -> str:
    """Write the label columns of one sample back (to `out`, by default in place)."""
    out = out or path
    s = pd.read_csv(path, dtype={"address": str})
    if set(COLUMNS[high]) <= set(s.columns):
        return f"{path.name}: labels present, unchanged"
    if not bs.BABD.exists():
        raise SystemExit(f"{path.name} holds addresses only. Place BABD-13.csv in {bs.BABD.parent} "
                         "(https://www.kaggle.com/datasets/lemonx/babd13) and run this script again.")
    cols = ["account", "SW", "label"] + (["PDIa1-3"] if high else [])
    b = pd.read_csv(bs.BABD, usecols=cols)
    b["account"] = b["account"].astype(str)
    b = b[b["SW"] == "SA"]
    if high:
        b = b[b["PDIa1-3"] >= bs.HIGH_DEGREE]
    b = b[b["account"].isin(set(s["address"]))].drop_duplicates("account")
    b = b.rename(columns={"account": "address", "PDIa1-3": "babd_degree"})
    m = s[["address"]].merge(b, on="address", how="left", validate="one_to_one")
    missing = int(m["label"].isna().sum())
    if missing:
        raise SystemExit(f"{path.name}: {missing} addresses have no matching row in {bs.BABD.name}; "
                         "check that the file is the published BABD-13 release.")
    m["label"] = m["label"].astype(int)
    m["class"] = m["label"].map(bs.CLASSES)
    m["group"] = "service_high" if high else m["label"].map(bs.GROUP)
    m[COLUMNS[high]].to_csv(out, index=False)
    return f"{path.name}: labels of {len(m)} addresses restored"


def main():
    for path, high in ((bs.SAMPLE, False), (bs.SAMPLE_HIGH, True)):
        print(restore(path, high), flush=True)


if __name__ == "__main__":
    sys.exit(main())
