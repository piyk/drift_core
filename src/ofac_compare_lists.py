"""ofac_compare_lists.py  |  Compare the address list extracted from the SDN
Advanced XML with the list used so far, with the corpus, and with the cached
feature records.

Run:  python revise1/src/ofac_compare_lists.py
"""
import collections
import json

import pandas as pd

import drift_core as dc

REV = dc.ROOT / "revise1"
NEW = REV / "ofac" / "lists" / "sanctioned_addresses_XBT.txt"
OLD = REV / "outputs" / "ofac_xbt_list_2026-05-17.txt"
CACHE = REV / "outputs" / "ofac_features_full.csv"


def kind(a):
    low = a.lower()
    if a.startswith("1"):
        return "P2PKH (1...)"
    if a.startswith("3"):
        return "P2SH (3...)"
    if low.startswith("bc1p"):
        return "P2TR (bc1p...)"
    if low.startswith("bc1q"):
        return "P2WPKH (bc1q, 42)" if len(a) == 42 else "P2WSH (bc1q, 62)" if len(a) == 62 else "bech32 other"
    return "other"


new = [a.strip() for a in NEW.read_text().splitlines() if a.strip()]
old = [a.strip() for a in OLD.read_text().splitlines() if a.strip()]
print("new list:", len(new), "distinct:", len(set(new)), "sorted:", new == sorted(new))
print("old list:", len(old), "distinct:", len(set(old)))
print("in both:", len(set(new) & set(old)), "| only new:", len(set(new) - set(old)),
      "| only old:", len(set(old) - set(new)))
print("types new:", dict(collections.Counter(map(kind, new))))
print("types only-new:", dict(collections.Counter(map(kind, set(new) - set(old)))))
print("only old:", sorted(set(old) - set(new)))
odd = [a for a in new if kind(a) in ("other", "bech32 other") or " " in a or len(a) < 26]
print("non-standard entries:", odd)
# case-insensitive duplicates among bech32 addresses
low = collections.Counter(a.lower() if a.lower().startswith("bc1") else a for a in new)
print("case-insensitive duplicates:", [a for a, c in low.items() if c > 1])

corpus = set(dc.load_data()["address"].astype(str))
print("overlap with corpus:", len(set(new) & corpus))
cache = pd.read_csv(CACHE)
have = set(cache["address"].astype(str))
print("cached feature records:", len(have), "| of new list cached:", len(set(new) & have),
      "| still to retrieve:", len(set(new) - have))
print("types still to retrieve:", dict(collections.Counter(map(kind, set(new) - have))))
(REV / "outputs" / "ofac_list_comparison.json").write_text(json.dumps({
    "sdn_issue_date": "2026-11-09", "n_new": len(new), "n_old": len(old),
    "in_both": len(set(new) & set(old)), "only_new": sorted(set(new) - set(old)),
    "only_old": sorted(set(old) - set(new)),
    "types_new": dict(collections.Counter(map(kind, new))),
    "overlap_with_corpus": len(set(new) & corpus)}, indent=1))
