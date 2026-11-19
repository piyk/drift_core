"""ofac_programs.py  |  Sanctions program and party type of every listed address.

Reads the OFAC SDN Advanced XML file and writes, for every Bitcoin address of
the list, the profile that holds it, whether that profile is an individual or
an entity, and the sanctions programs under which the profile is designated.
The external check uses this file to stratify the sanctioned addresses
(ofac_crossfit.py). Nothing is fitted here.

Run (from the revise1 folder):
    python src/ofac_programs.py ofac/SDN_ADVANCED.XML
"""
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "ofac_programs.json"
ASSET = "Digital Currency Address - XBT"

# Programs grouped by the kind of activity that the designation addresses.
# A profile that carries programs of several groups is assigned to the first
# group of this list that matches, so that every address has one stratum.
GROUPS = [
    ("Cyber-related", ("CYBER",)),
    ("North Korea", ("DPRK",)),
    ("Narcotics", ("SDNTK", "SDNT", "ILLICIT-DRUGS", "NARCO")),
    ("Russia-related", ("RUSSIA", "UKRAINE", "ELECTION", "MAGNIT")),
    ("Counter-terrorism and Iran", ("SDGT", "IRAN", "IRGC", "IFSR", "NPWMD", "HOSTAGES", "HRIT")),
    ("Transnational crime", ("TCO",)),
]


def group_of(programs):
    for name, keys in GROUPS:
        if any(p.upper().startswith(k) or k in p.upper() for p in programs for k in keys):
            return name
    return "Other"


def main():
    xml_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "ofac" / "SDN_ADVANCED.XML"
    root = ET.parse(xml_path).getroot()
    ns = root.tag.split("}")[0] + "}"
    rvs = root.find(ns + "ReferenceValueSets")
    fid = [e.attrib["ID"] for e in rvs.find(ns + "FeatureTypeValues") if (e.text or "").strip() == ASSET][0]
    party_type = {e.attrib["ID"]: (e.text or "").strip() for e in rvs.find(ns + "PartyTypeValues")}
    sub_type = {e.attrib["ID"]: party_type.get(e.attrib.get("PartyTypeID"), "Unknown")
                for e in rvs.find(ns + "PartySubTypeValues")}

    programs = defaultdict(set)
    for entry in root.find(ns + "SanctionsEntries"):
        pid = entry.attrib.get("ProfileID")
        for meas in entry.findall(ns + "SanctionsMeasure"):
            c = meas.find(ns + "Comment")
            if c is not None and (c.text or "").strip():
                programs[pid].add(c.text.strip())

    rows = {}
    for party in root.find(ns + "DistinctParties"):
        for prof in party.findall(ns + "Profile"):
            pid = prof.attrib["ID"]
            for feat in prof.findall(ns + "Feature"):
                if feat.attrib.get("FeatureTypeID") != fid:
                    continue
                for det in feat.iter(ns + "VersionDetail"):
                    addr = (det.text or "").strip()
                    if not addr:
                        continue
                    progs = sorted(programs.get(pid, []))
                    rows[addr] = {"profile_id": pid,
                                  "party_type": sub_type.get(prof.attrib.get("PartySubTypeID"), "Unknown"),
                                  "programs": progs, "group": group_of(progs)}

    summary = {"n_addresses": len(rows),
               "n_profiles": len({r["profile_id"] for r in rows.values()}),
               "by_group": dict(Counter(r["group"] for r in rows.values())),
               "by_party_type": dict(Counter(r["party_type"] for r in rows.values())),
               "by_program": dict(Counter(p for r in rows.values() for p in r["programs"]))}
    OUT.write_text(json.dumps({"source": xml_path.name, "summary": summary, "addresses": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"[save] {OUT}")


if __name__ == "__main__":
    main()
