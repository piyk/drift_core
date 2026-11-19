"""
ofac_scrape.py  |  Features of every OFAC-listed Bitcoin address
=================================================================
The original submission scored only the first 360 addresses of a mirrored
list in lexicographic order, which excluded every bech32 address. This
script removes that cap. The address list is the one extracted from the
OFAC SDN Advanced XML (revise1/ofac/SDN_ADVANCED.XML, issued 9 Nov 2026)
with the generate-address-list.py tool of the 0xB10C repository:

    python -I revise1/ofac/repo/ofac-sanctioned-digital-currency-addresses-main/generate-address-list.py \
        XBT -f TXT JSON -path revise1/ofac/lists -sdn revise1/ofac/SDN_ADVANCED.XML

Wallet features are retrieved with the same Esplora procedure as the corpus
scraper (blockchain-dataset/Wallet3.py): chain statistics from
/address/<addr>, and the active span from the two most recent pages of
confirmed transactions (at most 50 transactions). Entries of the list that
are not Bitcoin addresses are skipped and reported.

The result is cached incrementally, so an interrupted run resumes. The
request rate is kept low because the public explorers rate-limit.

Run:  python revise1/src/ofac_scrape.py
"""
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

import drift_core as dc

LIST_FILE = dc.ROOT / "revise1" / "ofac" / "lists" / "sanctioned_addresses_XBT.txt"
CACHE = dc.OUT_DIR / "ofac_features_full.csv"
HOSTS = ["https://blockstream.info/api", "https://mempool.space/api"]
MAX_TX_PAGES = 2
SLEEP = 1.1          # seconds between requests; faster rates were answered with HTTP 429


def is_bitcoin_address(a: str) -> bool:
    """Mainnet formats only: Base58 P2PKH (1...), P2SH (3...), and bech32 (bc1...)."""
    return a.startswith("1") or a.startswith("3") or a.lower().startswith("bc1")


def classify_type(a: str) -> str:
    low = a.lower()
    if a.startswith("1"):
        return "Legacy (P2PKH, Base58)"
    if a.startswith("3"):
        return "P2SH (Base58)"
    if low.startswith("bc1p"):
        return "Taproot (P2TR, bech32m)"
    if low.startswith("bc1q"):
        if len(a) == 42:
            return "Native SegWit (P2WPKH, bech32)"
        if len(a) in (62, 64):
            return "Native SegWit (P2WSH, bech32)"
        return "Native SegWit (bech32)"
    if low.startswith("bc1"):
        return "Bech32 (SegWit family)"
    return "Unknown"


def get_json(session, url, tries=4):
    for t in range(tries):
        try:
            r = session.get(url, timeout=15)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2.0 * (t + 1))
                continue
            return None
        except requests.RequestException:
            time.sleep(1.5 * (t + 1))
    return None


def fetch(session, addr, base):
    d = get_json(session, f"{base}/address/{addr}")
    if not d:
        return None
    chain = d.get("chain_stats", {}) or {}
    mem = d.get("mempool_stats", {}) or {}
    funded, spent = chain.get("funded_txo_sum", 0), chain.get("spent_txo_sum", 0)
    balance = (funded - spent) / 1e8
    unconf = (mem.get("funded_txo_sum", 0) - mem.get("spent_txo_sum", 0)) / 1e8
    times, pages = [], 0
    time.sleep(SLEEP)
    page = get_json(session, f"{base}/address/{addr}/txs")
    if page is None and chain.get("tx_count", 0) > 0:
        return None          # transaction page unavailable: do not store a record without a span
    while page and pages < MAX_TX_PAGES:
        for tx in page:
            st = tx.get("status") or {}
            if st.get("confirmed") and "block_time" in st:
                times.append(st["block_time"])
        pages += 1
        if pages >= MAX_TX_PAGES or len(page) < 25:
            break
        time.sleep(SLEEP)
        page = get_json(session, f"{base}/address/{addr}/txs/chain/{page[-1]['txid']}")
    n_tx = chain.get("tx_count", 0)
    if times:
        fs = datetime.fromtimestamp(min(times), tz=timezone.utc)
        ls = datetime.fromtimestamp(max(times), tz=timezone.utc)
        active_days = max((ls - fs).days, 1)
        tx_per_day = n_tx / active_days
    else:
        fs = ls = None
        active_days, tx_per_day = 0, 0.0
    return dict(address=addr, address_type=classify_type(addr), n_tx=n_tx,
                unconfirmed_n_tx=mem.get("tx_count", 0),
                final_n_tx=n_tx + mem.get("tx_count", 0),
                total_received=funded / 1e8, total_sent=spent / 1e8, balance=balance,
                unconfirmed_balance=unconf, final_balance=balance + unconf,
                first_seen=fs, last_seen=ls, active_days=active_days, tx_per_day=tx_per_day,
                avg_tx_value=0.0, max_tx_value=0.0, min_tx_value=0.0, total_tx_value=0.0,
                retrieved_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                source=base)


def fetch_bci(session, addr):
    """blockchain.info, the second fallback of the corpus scraper: totals
    from /rawaddr and the active span from its 50 most recent transactions."""
    r = None
    for attempt in range(3):
        try:
            r = session.get(f"https://blockchain.info/rawaddr/{addr}?limit=50", timeout=(6, 20))
        except requests.RequestException:
            r = None
        if r is not None and r.status_code == 200:
            break
        time.sleep(30.0 * (attempt + 1))     # respect the rate limit before retrying
    if r is None or r.status_code != 200:
        return None
    d = r.json()
    n_tx = d.get("n_tx", 0)
    times = [tx.get("time") for tx in (d.get("txs") or []) if isinstance(tx.get("time"), int)]
    if times:
        fs = datetime.fromtimestamp(min(times), tz=timezone.utc)
        ls = datetime.fromtimestamp(max(times), tz=timezone.utc)
        active_days = max((ls - fs).days, 1)
        tx_per_day = n_tx / active_days
    else:
        fs = ls = None
        active_days, tx_per_day = 0, 0.0
    bal = d.get("final_balance", 0) / 1e8
    return dict(address=addr, address_type=classify_type(addr), n_tx=n_tx,
                unconfirmed_n_tx=0, final_n_tx=n_tx,
                total_received=d.get("total_received", 0) / 1e8,
                total_sent=d.get("total_sent", 0) / 1e8, balance=bal,
                unconfirmed_balance=0.0, final_balance=bal,
                first_seen=fs, last_seen=ls, active_days=active_days, tx_per_day=tx_per_day,
                avg_tx_value=0.0, max_tx_value=0.0, min_tx_value=0.0, total_tx_value=0.0,
                retrieved_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                source="https://blockchain.info")


def main():
    use_bci = "--bci" in sys.argv        # use when the Esplora hosts answer HTTP 429
    listed = [a.strip() for a in LIST_FILE.read_text().splitlines() if a.strip()]
    skipped = [a for a in listed if not is_bitcoin_address(a)]
    addrs = [a for a in listed if is_bitcoin_address(a)]
    if skipped:
        print(f"skipped {len(skipped)} list entries that are not Bitcoin addresses: {skipped}", flush=True)
    rows, done = [], set()
    if CACHE.exists():
        prev = pd.read_csv(CACHE)
        rows = prev.to_dict("records")
        done = set(prev["address"].astype(str))
    todo = [a for a in addrs if a not in done]
    print(f"{len(listed)} listed, {len(addrs)} Bitcoin addresses, {len(done & set(addrs))} cached, "
          f"{len(todo)} to fetch", flush=True)
    s = requests.Session()
    s.headers.update({"User-Agent": "btc-research-client/1.0"})
    failed, streak = [], 0
    for i, addr in enumerate(todo, 1):
        if use_bci:
            rec = fetch_bci(s, addr)
        else:
            rec = fetch(s, addr, HOSTS[0]) or fetch(s, addr, HOSTS[1])   # second host is a fallback only
        if rec is None:
            failed.append(addr)
            streak += 1
        else:
            streak = 0
            rows.append(rec)
            if len(rows) % 10 == 0:
                pd.DataFrame(rows).to_csv(CACHE, index=False)
        if i % 25 == 0:
            print(f"  {i}/{len(todo)} fetched, {len(failed)} failed", flush=True)
        if streak >= 5:
            print("five consecutive failures: the explorers are rate-limiting, stopping; "
                  "rerun later to resume from the cache", flush=True)
            failed.extend(todo[i:])
            break
        time.sleep(11.0 if use_bci else SLEEP)   # blockchain.info allows about one request per 10 s
    if rows:
        pd.DataFrame(rows).to_csv(CACHE, index=False)
    print(f"done: {len(rows)} records, {len(failed)} failed", flush=True)
    if failed:
        print("failed:", failed, flush=True)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main()
