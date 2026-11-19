"""babd_source_check.py  |  Do the two explorer services return the same wallet fields?

In the labeled sample the illicit-associated addresses were retrieved from
blockchain.info and most service addresses from the Esplora hosts. This script
retrieves a random subset of the blockchain.info records again from an Esplora
host and compares the fields that enter the 21 features, so that the
comparison between the two groups cannot be an effect of the data source.

Run:  python revise1/src/babd_source_check.py [n]
"""
import json
import sys
import time

import numpy as np
import pandas as pd
import requests

import drift_core as dc
import ofac_scrape as osc

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
FIELDS = ["n_tx", "total_received", "total_sent", "final_balance", "active_days"]


def main():
    bci = pd.read_csv(dc.OUT_DIR / "babd_features_bci.csv")
    rng = np.random.RandomState(7)
    pick = bci.iloc[rng.choice(len(bci), size=min(N, len(bci)), replace=False)]
    sess = requests.Session()
    sess.headers.update({"User-Agent": "btc-research-client/1.0"})
    rows = []
    for _, r in pick.iterrows():
        rec = osc.fetch(sess, r["address"], osc.HOSTS[0]) or osc.fetch(sess, r["address"], osc.HOSTS[1])
        time.sleep(osc.SLEEP)
        if rec is None:
            continue
        rows.append({"address": r["address"],
                     **{f + "_bci": float(r[f]) for f in FIELDS},
                     **{f + "_esplora": float(rec[f]) for f in FIELDS}})
    d = pd.DataFrame(rows)
    out = {"n_requested": int(len(pick)), "n_compared": int(len(d)), "fields": {}}
    for f in FIELDS:
        a, b = d[f + "_bci"].values, d[f + "_esplora"].values
        same = np.isclose(a, b, rtol=0, atol=1e-8)
        out["fields"][f] = {"identical": int(same.sum()), "max_abs_diff": float(np.abs(a - b).max())}
    Fa = dc.engineer_features(d.rename(columns={f + "_bci": f for f in FIELDS}).assign(
        unconfirmed_n_tx=0, tx_per_day=lambda x: x["n_tx"] / x["active_days"].clip(lower=1), address_type="P2PKH"))
    Fb = dc.engineer_features(d.rename(columns={f + "_esplora": f for f in FIELDS}).assign(
        unconfirmed_n_tx=0, tx_per_day=lambda x: x["n_tx"] / x["active_days"].clip(lower=1), address_type="P2PKH"))
    out["max_abs_feature_diff"] = float(np.abs(Fa.values - Fb.values).max())
    out["all_identical"] = bool(all(v["identical"] == len(d) for v in out["fields"].values()))
    (dc.OUT_DIR / "babd_source_check.json").write_text(json.dumps(out, indent=1))
    d.to_csv(dc.OUT_DIR / "babd_source_check.csv", index=False)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
