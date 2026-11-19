import pandas as pd
import requests, time
from datetime import datetime, timezone
import gc

# ==================== TUNABLES ====================
ADDR_LIMIT   = 100000         # ต้องการดึง address กี่อัน
MAX_PAGES    = 3000         # ไล่บล็อกย้อนหลังได้สูงสุดกี่ "เพจบล็อก" (1 เพจ ~ 10 บล็อก)
SLEEP_SEC    = 0.25         # หน่วงเวลาใช้ตอนไล่บล็อก/หน้า tx (เก็บ address)
SLEEP_BETWEEN_WALLETS = 1.0 # หน่วงระหว่างดึงฟีเจอร์แต่ละ address
TIMEOUT_SEC  = 15
RETRY_TIMES  = 3

# แหล่ง Esplora หลาย host ไว้หมุนใช้
# ESPLORA_SOURCES = ["https://blockstream.info/api", "https://mempool.space/api"]
ESPLORA_SOURCES = [ "https://mempool.space/api"]

# ถ้ามี BlockCypher token ใส่เพื่อลด 429
BLOCKCYPHER_TOKEN = ""
# ==================================================

def satoshi_to_btc(satoshi: int) -> float:
    return satoshi / 1e8

def classify_address_type(addr: str) -> str:
    a = (addr or "").strip()
    low = a.lower()
    # Legacy (Base58)
    if a.startswith("1"):
        return "Legacy (P2PKH, Base58)"
    if a.startswith("3"):
        return "P2SH (Base58)"

    # Bech32 / Bech32m family
    if low.startswith("bc1p"):
        # Taproot = bech32m
        return "Taproot (P2TR, bech32m)"
    if low.startswith("bc1q"):
        # Native SegWit = bech32
        # 42 ตัวอักษรโดยมากคือ P2WPKH, ~62 คือ P2WSH
        if len(a) == 42:
            return "Native SegWit (P2WPKH, bech32)"
        if len(a) in (62, 64):
            return "Native SegWit (P2WSH, bech32)"
        return "Native SegWit (bech32)"
    if low.startswith("bc1"):
        return "Bech32 (SegWit family)"

    return "Unknown"


def _get_json_with_retry(session: requests.Session, url: str, timeout=TIMEOUT_SEC, retries=RETRY_TIMES):
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(1.0 * attempt, 5.0))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == retries:
                return None
            time.sleep(min(1.0 * attempt, 5.0))
    return None

# ========== STEP 1: เก็บ Address (แบ่งหน้า + เก็บทั้งส่ง/รับ) ==========
def get_recent_wallet_addresses(limit=ADDR_LIMIT, max_pages=MAX_PAGES, sleep_sec=SLEEP_SEC):
    base = ESPLORA_SOURCES[0]  # ใช้ Blockstream เป็นหลักตอนเก็บ address
    s = requests.Session()
    s.headers.update({"User-Agent": "btc-addr-scraper/merged/1.0"})

    tip = s.get(f"{base}/blocks/tip/height", timeout=TIMEOUT_SEC)
    tip.raise_for_status()
    start_height = int(tip.text)

    addrs = set()
    pages = 0

    while len(addrs) < limit and pages < max_pages and start_height >= 0:
        blocks = _get_json_with_retry(s, f"{base}/blocks/{start_height}")
        if not blocks:
            break

        for blk in blocks:
            if len(addrs) >= limit: break
            bhash = blk["id"]

            # ไล่หน้า txs ในบล็อก
            idx = 0
            while len(addrs) < limit:
                txs = _get_json_with_retry(s, f"{base}/block/{bhash}/txs/{idx}")
                if not txs:  # หมดหน้า/ดึงไม่สำเร็จ
                    break

                for tx in txs:
                    # ผู้รับ (vout)
                    for vout in tx.get("vout", []):
                        a = vout.get("scriptpubkey_address")
                        if a:
                            addrs.add(a)
                            if len(addrs) >= limit: break
                    if len(addrs) >= limit: break

                    # ผู้ส่ง (vin.prevout)
                    for vin in tx.get("vin", []):
                        prev = vin.get("prevout") or {}
                        a = prev.get("scriptpubkey_address")
                        if a:
                            addrs.add(a)
                            if len(addrs) >= limit: break
                    if len(addrs) >= limit: break

                if len(txs) < 25:   # Esplora คืนทีละ ~25; ถ้าน้อยกว่าแปลว่าหน้าสุดท้าย
                    break
                idx += len(txs)
                time.sleep(sleep_sec)

            time.sleep(sleep_sec)

        pages += 1
        start_height = blocks[-1]["height"] - 1
        time.sleep(sleep_sec)

    return list(addrs)

# ========== STEP 2A: ฟีเจอร์จาก Esplora (Blockstream/mempool) ==========
def fetch_wallet_features_from_esplora(address: str, max_tx_pages=2, sleep_sec=0.3):
    s = requests.Session()
    s.headers.update({"User-Agent": "btc-addr-scraper/merged/1.0"})

    def get_json(base, path):
        return _get_json_with_retry(s, f"{base}{path}")

    for base in ESPLORA_SOURCES:
        d = get_json(base, f"/address/{address}")
        if not d:
            continue

        chain = d.get("chain_stats", {}) or {}
        mem   = d.get("mempool_stats", {}) or {}

        funded_chain = chain.get("funded_txo_sum", 0)
        spent_chain  = chain.get("spent_txo_sum", 0)
        funded_mem   = mem.get("funded_txo_sum", 0)
        spent_mem    = mem.get("spent_txo_sum", 0)

        balance = (funded_chain - spent_chain) / 1e8
        unconf  = (funded_mem - spent_mem) / 1e8
        final_balance = balance + unconf

        # เวลาที่ใช้งาน (จำกัดหน้าเพื่อลดโหลด/เลี่ยง 429)
        times = []
        page = get_json(base, f"/address/{address}/txs")
        pages = 0
        last_txid = None
        while page and pages < max_tx_pages:
            for tx in page:
                st = tx.get("status") or {}
                if st.get("confirmed") and "block_time" in st:
                    times.append(st["block_time"])
            last_txid = page[-1]["txid"]
            page = get_json(base, f"/address/{address}/txs/chain/{last_txid}")
            pages += 1
            time.sleep(sleep_sec)

        if times:
            first_seen = datetime.fromtimestamp(min(times), tz=timezone.utc)
            last_seen  = datetime.fromtimestamp(max(times), tz=timezone.utc)
            active_days = max((last_seen - first_seen).days, 1)
            tx_per_day  = (chain.get("tx_count", 0)) / active_days
        else:
            first_seen = last_seen = None
            active_days = 0
            tx_per_day = 0.0

        return {
            "address": address,
            "address_type": classify_address_type(address),
            "n_tx": chain.get("tx_count", 0),
            "unconfirmed_n_tx": mem.get("tx_count", 0),
            "final_n_tx": chain.get("tx_count", 0) + mem.get("tx_count", 0),
            "total_received": funded_chain / 1e8,
            "total_sent": spent_chain / 1e8,
            "balance": balance,
            "unconfirmed_balance": unconf,
            "final_balance": final_balance,
            "first_seen": first_seen,
            "last_seen": last_seen,
            "active_days": active_days,
            "tx_per_day": tx_per_day,
            "avg_tx_value": 0.0, "max_tx_value": 0.0,
            "min_tx_value": 0.0, "total_tx_value": 0.0
        }
    return None

# ========== STEP 2B: ฟีเจอร์จาก Blockchain.info ==========
def fetch_wallet_features_from_blockchain_info(address: str):
    url = f"https://blockchain.info/rawaddr/{address}?limit=50"
    try:
        r = requests.get(url, timeout=TIMEOUT_SEC)
        if not r.ok:
            return None
        d = r.json()
    except requests.RequestException:
        return None

    n_tx = d.get("n_tx", 0)
    total_received = satoshi_to_btc(d.get("total_received", 0))
    total_sent     = satoshi_to_btc(d.get("total_sent", 0))
    final_balance  = satoshi_to_btc(d.get("final_balance", 0))

    txs = d.get("txs", []) or []
    times = [tx.get("time") for tx in txs if isinstance(tx.get("time"), int)]
    if times:
        first_seen = datetime.fromtimestamp(min(times), tz=timezone.utc)
        last_seen  = datetime.fromtimestamp(max(times), tz=timezone.utc)
        active_days = max((last_seen - first_seen).days, 1)
        tx_per_day  = n_tx / active_days if active_days else 0.0
    else:
        first_seen = last_seen = None
        active_days = 0
        tx_per_day = 0.0

    return {
        "address": address,
        "address_type": classify_address_type(address),
        "n_tx": n_tx,
        "unconfirmed_n_tx": 0,
        "final_n_tx": n_tx,
        "total_received": total_received,
        "total_sent": total_sent,
        "balance": final_balance,
        "unconfirmed_balance": 0.0,
        "final_balance": final_balance,
        "first_seen": first_seen,
        "last_seen": last_seen,
        "active_days": active_days,
        "tx_per_day": tx_per_day,
        "avg_tx_value": 0.0, "max_tx_value": 0.0,
        "min_tx_value": 0.0, "total_tx_value": 0.0
    }

# ========== STEP 2C: BlockCypher + fallback ==========
def fetch_wallet_features_full(address: str):
    url = f"https://api.blockcypher.com/v1/btc/main/addrs/{address}?limit=50"
    if BLOCKCYPHER_TOKEN:
        url += f"&token={BLOCKCYPHER_TOKEN}"

    try:
        resp = requests.get(url, timeout=TIMEOUT_SEC)
    except requests.RequestException:
        # ถ้าเรียกไม่ได้ ให้ fallback
        return (fetch_wallet_features_from_esplora(address)
                or fetch_wallet_features_from_blockchain_info(address))

    if resp.status_code == 200:
        data = resp.json()
        txrefs = data.get("txrefs", []) or []
        values = [tx.get("value", 0) for tx in txrefs]
        timestamps = [tx.get("confirmed") for tx in txrefs if tx.get("confirmed")]

        if timestamps:
            try:
                dates = [datetime.fromisoformat(t.replace("Z", "+00:00")) for t in timestamps]
            except ValueError:
                dates = [datetime.fromisoformat(t[:-1]) for t in timestamps if t.endswith("Z")]
            first_seen = min(dates); last_seen = max(dates)
            active_days = max((last_seen - first_seen).days, 1)
            tx_per_day  = len(dates) / active_days
        else:
            first_seen = last_seen = None
            active_days = 0
            tx_per_day  = 0.0

        total_tx_value = sum(values) / 1e8 if values else 0.0
        avg_tx_value   = (sum(values) / len(values)) / 1e8 if values else 0.0
        max_tx_value   = max(values) / 1e8 if values else 0.0
        min_tx_value   = min(values) / 1e8 if values else 0.0

        return {
            "address": address,
            "address_type": classify_address_type(address),
            "n_tx": data.get("n_tx", 0),
            "unconfirmed_n_tx": data.get("unconfirmed_n_tx", 0),
            "final_n_tx": data.get("final_n_tx", 0),
            "total_received": satoshi_to_btc(data.get("total_received", 0)),
            "total_sent": satoshi_to_btc(data.get("total_sent", 0)),
            "balance": satoshi_to_btc(data.get("balance", 0)),
            "unconfirmed_balance": satoshi_to_btc(data.get("unconfirmed_balance", 0)),
            "final_balance": satoshi_to_btc(data.get("final_balance", 0)),
            "first_seen": first_seen,
            "last_seen": last_seen,
            "active_days": active_days,
            "tx_per_day": tx_per_day,
            "avg_tx_value": avg_tx_value,
            "max_tx_value": max_tx_value,
            "min_tx_value": min_tx_value,
            "total_tx_value": total_tx_value
        }
    else:
        # HTTP 429/403/5xx → fallback
        return (fetch_wallet_features_from_esplora(address)
                or fetch_wallet_features_from_blockchain_info(address))
    



# ========== STEP 3: รวมทั้งหมด ==========
def main():
    print("📦 Loading recent wallet addresses (with pagination, in/out capture)...")
    wallets = get_recent_wallet_addresses(limit=ADDR_LIMIT, max_pages=MAX_PAGES, sleep_sec=SLEEP_SEC)
    print(f"✅ Found {len(wallets)} addresses")
    if not wallets:
        print("⚠️ No addresses collected. Try increasing MAX_PAGES or adjusting SLEEP_SEC."); return
    #print("🔎 Sample:", wallets[:10], "\n")

    features, failed = [], []
    for i, addr in enumerate(wallets, start=1):
        print(f"({i}/{len(wallets)}) Fetching data of {addr} ...")
        info = fetch_wallet_features_full(addr)
        if info: 
            features.append(info)
        else:
            failed.append(addr)
        time.sleep(SLEEP_BETWEEN_WALLETS)

    df = pd.DataFrame(features)
    print(f"\n✅ Retrieved data: {len(df)} rows (from {len(wallets)} addresses)")
    if failed:
        print("⚠️ Failed addresses:", failed)
    ts = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    out = f"wallet_dataset_detailed_{ts}.csv"
    df.to_csv(out, index=False)
    print(f"\n💾 Saved file as '{out}'")
    gc.collect()

if __name__ == "__main__":
    main()
