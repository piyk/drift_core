"""babd_scrape.py  |  A labeled address sample from BABD-13, measured like the corpus.

BABD-13 (Xiang et al., IEEE Trans. Inf. Forensics Security, 2024) labels
544,462 Bitcoin addresses with thirteen behaviour types. Only its address and
label columns are used here. A stratified random sample of strongly labeled
addresses is drawn, and the wallet fields of every sampled address are
retrieved with the procedure of ofac_scrape.py (Esplora chain statistics and
the active span over at most 50 recent transactions), so that the labeled
addresses are measured exactly like the sanctioned addresses.

  * service classes (verified negatives): centralized exchange, mining pool,
    gambling, individual wallet;
  * classes associated with illicit activity: darknet market, tumbler,
    blackmail, money laundering, Ponzi scheme, government criminal blacklist.

Addresses that carry more than one label, that occur in the corpus, or that
are on the OFAC list are excluded. The BABD-13 file is not redistributed
(licence CC BY-NC-ND 4.0); place BABD-13.csv in revise1/external/babd13/.

Run:  python revise1/src/babd_scrape.py [--mempool-first]          (resumable)
"""
import sys
import time

import numpy as np
import pandas as pd
import requests

import drift_core as dc
import ofac_scrape as osc

BABD = dc.ROOT / "revise1" / "external" / "babd13" / "BABD-13.csv"
SAMPLE = dc.OUT_DIR / "babd_sample.csv"
CACHE = dc.OUT_DIR / "babd_features.csv"
CACHE_BCI = dc.OUT_DIR / "babd_features_bci.csv"
SAMPLE_HIGH = dc.OUT_DIR / "babd_sample_high.csv"
CACHE_HIGH = dc.OUT_DIR / "babd_features_high.csv"
CACHE_HIGH_BCI = dc.OUT_DIR / "babd_features_high_bci.csv"
HIGH_DEGREE = 100
QUOTA_HIGH = {3: 200, 6: 100}                    # centralized exchange, gambling
SEED = 2026
CLASSES = {0: "Blackmail", 1: "Cyber-security service", 2: "Darknet market",
           3: "Centralized exchange", 4: "P2P financial infrastructure service",
           5: "P2P financial service", 6: "Gambling", 7: "Government criminal blacklist",
           8: "Money laundering", 9: "Ponzi scheme", 10: "Mining pool", 11: "Tumbler",
           12: "Individual wallet"}
# number of strongly labeled addresses drawn per class (None: all of them)
QUOTA = {3: 250, 10: 250, 6: 150, 12: None, 2: 200, 11: 150, 0: None, 7: None, 8: None, 9: None}
GROUP = {3: "service", 10: "service", 6: "service", 12: "service",
         2: "illicit", 11: "illicit", 0: "illicit", 7: "illicit", 8: "illicit", 9: "illicit"}


def draw_sample() -> pd.DataFrame:
    df = pd.read_csv(BABD, usecols=["account", "SW", "label"])
    df["account"] = df["account"].astype(str)
    n_labels = df.groupby("account")["label"].nunique()
    multi = set(n_labels[n_labels > 1].index)
    corpus = set(dc.load_data()["address"].astype(str))
    ofac = {a.strip() for a in osc.LIST_FILE.read_text().splitlines() if a.strip()}
    df = df[(df["SW"] == "SA") & ~df["account"].isin(multi | corpus | ofac)]
    df = df[df["account"].map(osc.is_bitcoin_address)].drop_duplicates("account")
    rng = np.random.RandomState(SEED)
    parts = []
    for lab, q in QUOTA.items():
        pool = df[df["label"] == lab].sort_values("account")
        take = pool if q is None or len(pool) <= q else pool.iloc[rng.choice(len(pool), size=q, replace=False)]
        parts.append(take)
    s = pd.concat(parts, ignore_index=True).rename(columns={"account": "address"})
    s["class"] = s["label"].map(CLASSES)
    s["group"] = s["label"].map(GROUP)
    s.to_csv(SAMPLE, index=False)
    print(f"sample: {len(s)} addresses\n{s.groupby(['group', 'class']).size()}", flush=True)
    print(f"excluded: {len(multi)} with several labels; corpus and OFAC-listed addresses", flush=True)
    return s


def draw_high_activity_sample() -> pd.DataFrame:
    """Second sample: strongly labeled service addresses with a total degree of
    at least 100 in the BABD-13 observation window, as verified high-activity
    negatives. The illicit-associated classes of BABD-13 contain almost no such
    address (nine in total), so this sample holds services only."""
    df = pd.read_csv(BABD, usecols=["account", "SW", "label", "PDIa1-3"])
    df["account"] = df["account"].astype(str)
    n_labels = df.groupby("account")["label"].nunique()
    multi = set(n_labels[n_labels > 1].index)
    corpus = set(dc.load_data()["address"].astype(str))
    ofac = {a.strip() for a in osc.LIST_FILE.read_text().splitlines() if a.strip()}
    first = set(pd.read_csv(SAMPLE)["address"].astype(str)) if SAMPLE.exists() else set()
    df = df[(df["SW"] == "SA") & (df["PDIa1-3"] >= HIGH_DEGREE) & ~df["account"].isin(multi | corpus | ofac | first)]
    df = df[df["account"].map(osc.is_bitcoin_address)].drop_duplicates("account")
    rng = np.random.RandomState(SEED + 1)
    parts = []
    for lab, q in QUOTA_HIGH.items():
        pool = df[df["label"] == lab].sort_values("account")
        parts.append(pool if len(pool) <= q else pool.iloc[rng.choice(len(pool), size=q, replace=False)])
    s = pd.concat(parts, ignore_index=True).rename(columns={"account": "address", "PDIa1-3": "babd_degree"})
    s["class"] = s["label"].map(CLASSES)
    s["group"] = "service_high"
    s.to_csv(SAMPLE_HIGH, index=False)
    print(f"high-activity sample: {len(s)} addresses\n{s.groupby('class').size()}", flush=True)
    return s


def cached_addresses(path):
    try:
        return set(pd.read_csv(path, usecols=["address"])["address"].astype(str))
    except Exception:
        return set()


def main():
    """Two workers can run side by side, each at the request rate that its
    service permits: the default worker uses the Esplora hosts and walks the
    sample from the front; the --bci worker uses blockchain.info, the second
    fallback of the corpus scraper, and walks it from the back. Each worker
    writes its own cache and skips what the other has fetched."""
    use_bci = "--bci" in sys.argv
    if "--high" in sys.argv:                      # the high-activity service sample
        mine, other = (CACHE_HIGH_BCI, CACHE_HIGH) if use_bci else (CACHE_HIGH, CACHE_HIGH_BCI)
        s = pd.read_csv(SAMPLE_HIGH) if SAMPLE_HIGH.exists() else draw_high_activity_sample()
    else:
        mine, other = (CACHE_BCI, CACHE) if use_bci else (CACHE, CACHE_BCI)
        s = pd.read_csv(SAMPLE) if SAMPLE.exists() else draw_sample()
    rows = pd.read_csv(mine).to_dict("records") if mine.exists() else []
    done = {str(r["address"]) for r in rows} | cached_addresses(other)
    todo = [a for a in s["address"].astype(str) if a not in done]
    if use_bci:
        todo = todo[::-1]
    print(f"{len(s)} sampled, {len(done)} cached, {len(todo)} to fetch "
          f"({'blockchain.info' if use_bci else 'Esplora'} worker)", flush=True)
    # --mempool-first: use when blockstream.info answers HTTP 429 (both hosts run Esplora)
    hosts = osc.HOSTS[::-1] if "--mempool-first" in sys.argv else osc.HOSTS
    sess = requests.Session()
    sess.headers.update({"User-Agent": "btc-research-client/1.0"})
    failed, streak, fetched = [], 0, 0
    for i, addr in enumerate(todo, 1):
        if i % 10 == 1:
            done |= cached_addresses(other)
        if addr in done:
            continue
        rec = osc.fetch_bci(sess, addr) if use_bci else (
            osc.fetch(sess, addr, hosts[0]) or osc.fetch(sess, addr, hosts[1]))
        if rec is None:
            failed.append(addr)
            streak += 1
        else:
            streak = 0
            fetched += 1
            rows.append(rec)
            if len(rows) % 10 == 0:
                pd.DataFrame(rows).to_csv(mine, index=False)
        if i % 25 == 0:
            print(f"  position {i}/{len(todo)}, {fetched} fetched, {len(failed)} failed", flush=True)
        if streak >= 5:
            print("five consecutive failures: the service is rate-limiting, stopping; "
                  "rerun later to resume from the cache", flush=True)
            break
        time.sleep(11.0 if use_bci else osc.SLEEP)
    if rows:
        pd.DataFrame(rows).to_csv(mine, index=False)
    print(f"done: {len(rows)} records in {mine.name}, {len(failed)} failed", flush=True)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main()
