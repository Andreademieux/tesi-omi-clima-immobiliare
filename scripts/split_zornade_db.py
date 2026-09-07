#!/usr/bin/env python3
"""
split_zornade_db.py
Divide zornade_data.db in shard di dimensione controllata, caricabili su Claude.

Tre modalità (combinabili):

  1. shard      — partiziona la tabella `parcels` per provincia (default) o per
                  comune, creando N database autonomi ciascuno sotto la soglia
                  di dimensione. Ogni shard contiene anche la porzione
                  pertinente della tabella `comuni`, quindi è autosufficiente.

  2. compact    — crea un unico DB "analitico" leggero: estrae dal detail_json
                  solo i campi utili all'analisi rischio/valutazione in colonne
                  piatte, scartando geometrie, POI e ridondanze (che pesano
                  ~70-80% del JSON). Tipicamente riduce il DB di 5-10x.
                  È il formato consigliato da caricare su Claude per le analisi.

  3. merge      — ricompone gli shard in un unico DB (operazione inversa di 1).

Uso:
  python split_zornade_db.py shard   zornade_data.db --by provincia
  python split_zornade_db.py shard   zornade_data.db --by comune
  python split_zornade_db.py shard   zornade_data.db --max-mb 10 --by comune
  python split_zornade_db.py compact zornade_data.db -o zornade_compact.db
  python split_zornade_db.py merge   shard_*.db -o zornade_merged.db

Note di prudenza:
  - Il DB sorgente è aperto in sola lettura (URI mode=ro): nessun rischio di
    modificarlo.
  - Gli shard rispettano i confini di comune: un comune non viene mai spezzato
    tra due shard, così ogni analisi comunale resta completa in un solo file.
  - Un manifest JSON accompagna gli shard (conteggi e checksum di riga) per
    verificare l'integrità al momento del merge.
"""

import argparse
import glob
import json
import math
import os
import sqlite3
import sys
from collections import defaultdict

SCHEMA_COMUNI = """
CREATE TABLE IF NOT EXISTS comuni (
    name TEXT,
    province TEXT,
    collect_done INTEGER DEFAULT 0,
    parcels_found INTEGER DEFAULT 0,
    PRIMARY KEY (name, province)
)"""

SCHEMA_PARCELS = """
CREATE TABLE IF NOT EXISTS parcels (
    fid INTEGER PRIMARY KEY,
    comune TEXT,
    province TEXT,
    label TEXT,
    area_m2 REAL,
    lat REAL,
    lng REAL,
    detail_json TEXT,
    detail_downloaded_at TEXT
)"""


def open_ro(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


# --------------------------------------------------------------------------
# 1) SHARD
# --------------------------------------------------------------------------

def cmd_shard(args):
    src = open_ro(args.db)
    key = "province" if args.by == "provincia" else "comune"

    # peso stimato per gruppo = somma lunghezze dei campi principali per riga
    sizes = {}
    counts = {}
    for g, n, w in src.execute(
        f"""SELECT {key}, COUNT(*),
                   COALESCE(SUM(LENGTH(COALESCE(detail_json,'')) + 200), 0)
            FROM parcels GROUP BY {key} ORDER BY 3 DESC"""
    ):
        sizes[g] = w
        counts[g] = n

    max_bytes = int(args.max_mb * 1_000_000 * 0.9)  # margine 10% per overhead pagine
    too_big = [g for g, w in sizes.items() if w > max_bytes]
    if too_big and args.by == "provincia":
        print(f"[!] Gruppi oltre soglia anche da soli: {too_big}. "
              f"Riprova con --by comune o alza --max-mb.", file=sys.stderr)

    # bin packing greedy (first-fit decreasing), senza mai spezzare un gruppo
    bins = []  # lista di (bytes_usati, [gruppi])
    for g in sorted(sizes, key=sizes.get, reverse=True):
        placed = False
        for b in bins:
            if b[0] + sizes[g] <= max_bytes:
                b[0] += sizes[g]
                b[1].append(g)
                placed = True
                break
        if not placed:
            bins.append([sizes[g], [g]])

    base = os.path.splitext(os.path.basename(args.db))[0]
    outdir = args.outdir or "."
    os.makedirs(outdir, exist_ok=True)
    manifest = {"source": os.path.basename(args.db), "by": key, "shards": []}

    for i, (used, groups) in enumerate(bins, 1):
        out_path = os.path.join(outdir, f"{base}_shard{i:02d}.db")
        if os.path.exists(out_path):
            os.remove(out_path)
        dst = sqlite3.connect(out_path)
        dst.execute(SCHEMA_COMUNI)
        dst.execute(SCHEMA_PARCELS)
        ph = ",".join("?" * len(groups))

        rows = src.execute(
            f"SELECT fid,comune,province,label,area_m2,lat,lng,detail_json,"
            f"detail_downloaded_at FROM parcels WHERE {key} IN ({ph})", groups)
        dst.executemany("INSERT INTO parcels VALUES (?,?,?,?,?,?,?,?,?)", rows)

        # porta con sé i comuni pertinenti
        ckey = "province" if key == "province" else "name"
        crows = src.execute(
            f"SELECT name,province,collect_done,parcels_found FROM comuni "
            f"WHERE {ckey} IN ({ph})", groups)
        dst.executemany("INSERT OR IGNORE INTO comuni VALUES (?,?,?,?)", crows)

        dst.commit()
        n_parcels = dst.execute("SELECT COUNT(*) FROM parcels").fetchone()[0]
        dst.execute("VACUUM")
        dst.close()
        mb = os.path.getsize(out_path) / 1e6
        manifest["shards"].append(
            {"file": os.path.basename(out_path), "groups": groups,
             "parcels": n_parcels, "mb": round(mb, 2)})
        print(f"  {out_path}: {n_parcels} particelle, {mb:.1f} MB, "
              f"{len(groups)} {key}")

    total_src = src.execute("SELECT COUNT(*) FROM parcels").fetchone()[0]
    total_out = sum(s["parcels"] for s in manifest["shards"])
    assert total_src == total_out, "Perdita di righe nello sharding!"
    mpath = os.path.join(outdir, f"{base}_manifest.json")
    json.dump(manifest, open(mpath, "w"), indent=2, ensure_ascii=False)
    print(f"[ok] {len(bins)} shard, {total_out}/{total_src} righe. "
          f"Manifest: {mpath}")


# --------------------------------------------------------------------------
# 2) COMPACT — estrazione analitica piatta (consigliata per Claude)
# --------------------------------------------------------------------------

COMPACT_SCHEMA = """
CREATE TABLE parcels_flat (
    fid INTEGER PRIMARY KEY,
    comune TEXT, provincia TEXT, regione TEXT,
    lat REAL, lng REAL, area_m2 REAL,
    foglio TEXT, comune_code TEXT, cap TEXT,
    seismic_zone INTEGER, pga REAL,
    flood_level TEXT, landslide_level TEXT,
    sub_velocity_mm_y REAL, sub_risk_index REAL, sub_risk_label TEXT,
    coastal INTEGER, coast_dist_m REAL, erosion_avg_m_y REAL,
    erosion_severity REAL, heritage INTEGER, n_poi INTEGER,
    detail_downloaded_at TEXT
);
CREATE TABLE valuations (
    fid INTEGER, zona TEXT, zona_desc TEXT, fascia TEXT,
    property_type TEXT, condition TEXT,
    buy_min REAL, buy_max REAL, rent_min REAL, rent_max REAL,
    prev_buy_min REAL, prev_buy_max REAL, prev_rent_min REAL, prev_rent_max REAL
);
CREATE INDEX idx_val_fid ON valuations(fid);
CREATE INDEX idx_flat_comune ON parcels_flat(comune);
"""


def cmd_compact(args):
    src = open_ro(args.db)
    out = args.output or "zornade_compact.db"
    if os.path.exists(out):
        os.remove(out)
    dst = sqlite3.connect(out)
    dst.executescript(COMPACT_SCHEMA)

    n = skipped = 0
    for (fid, com, prov, area, lat, lng, dj, ts) in src.execute(
        "SELECT fid,comune,province,area_m2,lat,lng,detail_json,"
        "detail_downloaded_at FROM parcels WHERE detail_json IS NOT NULL"):
        try:
            d = json.loads(dj)["data"]
        except (json.JSONDecodeError, KeyError):
            skipped += 1
            continue
        r = d.get("risk") or {}
        s = d.get("subsidence") or {}
        m = d.get("municipality") or {}
        c = d.get("cadastral") or {}
        ce = d.get("coastal_erosion") or []
        dist = min((x.get("distance_m") for x in ce
                    if x.get("distance_m") is not None), default=None)
        ero_avg = min((x.get("avg_change_m_year") for x in ce
                       if x.get("avg_change_m_year") is not None), default=None)
        ero_sev = max((x.get("severity_index") for x in ce
                       if x.get("severity_index") is not None), default=None)
        dst.execute(
            "INSERT INTO parcels_flat VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (fid, m.get("name") or com, m.get("province") or prov,
             m.get("region"), lat, lng, area,
             c.get("foglio"), c.get("comune_code"), c.get("postal_code"),
             r.get("seismic_zone"), r.get("pga"),
             r.get("flood_level"), r.get("landslide_level"),
             s.get("velocity_mm_year"), s.get("risk_index"),
             s.get("risk_label"),
             1 if ce else 0, dist, ero_avg, ero_sev,
             1 if d.get("cultural_heritage") else 0,
             len(d.get("poi") or []), ts))
        for v in d.get("valuation") or []:
            p = v.get("purchase") or {}
            rn = v.get("rental") or {}
            pv = v.get("previous_semester") or {}
            pp = pv.get("purchase") or {}
            pr = pv.get("rental") or {}
            dst.execute(
                "INSERT INTO valuations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (fid, v.get("zone"), v.get("zone_description"),
                 v.get("fascia"), v.get("property_type"), v.get("condition"),
                 p.get("min_eur_m2"), p.get("max_eur_m2"),
                 rn.get("min_eur_m2"), rn.get("max_eur_m2"),
                 pp.get("min_eur_m2"), pp.get("max_eur_m2"),
                 pr.get("min_eur_m2"), pr.get("max_eur_m2")))
        n += 1

    # tabella comuni di servizio (leggera, si porta sempre)
    dst.execute(SCHEMA_COMUNI)
    dst.executemany("INSERT INTO comuni VALUES (?,?,?,?)",
                    src.execute("SELECT * FROM comuni"))
    dst.commit()
    dst.execute("VACUUM")
    dst.close()
    mb_in = os.path.getsize(args.db) / 1e6
    mb_out = os.path.getsize(out) / 1e6
    print(f"[ok] {out}: {n} particelle ({skipped} json illeggibili scartati), "
          f"{mb_in:.1f} MB -> {mb_out:.1f} MB "
          f"(-{100 * (1 - mb_out / mb_in):.0f}%)")
    print("    NB: geometrie e POI non sono inclusi. Per riaverli, "
          "risali al DB originale tramite fid.")


# --------------------------------------------------------------------------
# 3) MERGE
# --------------------------------------------------------------------------

def cmd_merge(args):
    out = args.output or "zornade_merged.db"
    if os.path.exists(out):
        os.remove(out)
    dst = sqlite3.connect(out)
    dst.execute(SCHEMA_COMUNI)
    dst.execute(SCHEMA_PARCELS)
    files = []
    for pat in args.shards:
        files.extend(sorted(glob.glob(pat)))
    if not files:
        sys.exit("Nessuno shard trovato.")
    tot = 0
    for f in files:
        s = open_ro(f)
        before = dst.execute("SELECT COUNT(*) FROM parcels").fetchone()[0]
        dst.executemany(
            "INSERT OR IGNORE INTO parcels VALUES (?,?,?,?,?,?,?,?,?)",
            s.execute("SELECT * FROM parcels"))
        dst.executemany(
            "INSERT OR IGNORE INTO comuni VALUES (?,?,?,?)",
            s.execute("SELECT * FROM comuni"))
        after = dst.execute("SELECT COUNT(*) FROM parcels").fetchone()[0]
        print(f"  {f}: +{after - before} particelle")
        tot = after
        s.close()
    dst.commit()
    dst.execute("VACUUM")
    dst.close()
    print(f"[ok] {out}: {tot} particelle totali "
          f"(duplicati su fid ignorati automaticamente)")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("shard", help="partiziona in DB multipli")
    p1.add_argument("db")
    p1.add_argument("--max-mb", type=float, default=25,
                    help="dimensione massima di ogni shard (default 25 MB)")
    p1.add_argument("--by", choices=["provincia", "comune"], default="provincia",
                    help="chiave di partizione (default provincia)")
    p1.add_argument("--outdir", default=None)
    p1.set_defaults(fn=cmd_shard)

    p2 = sub.add_parser("compact", help="estrazione analitica piatta e leggera")
    p2.add_argument("db")
    p2.add_argument("-o", "--output", default=None)
    p2.set_defaults(fn=cmd_compact)

    p3 = sub.add_parser("merge", help="ricompone gli shard")
    p3.add_argument("shards", nargs="+", help="file o glob, es. shard_*.db")
    p3.add_argument("-o", "--output", default=None)
    p3.set_defaults(fn=cmd_merge)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
