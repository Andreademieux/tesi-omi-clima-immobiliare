#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integra nel database Zornade il rischio incendio aggregato EFFIS/JRC
(Wildfire Risk Viewer, harmonised pan-European Wildfire Risk Assessment),
che copre TUTTE e quattro le province (Torino, Milano, Roma, Napoli).

Fonte e formato: vedi dati_incendio_effis/README.md. In sintesi:
  - griglia EURO-CORDEX EUR-11 (~12 km), celle = quadrilateri lon/lat;
  - tre CSV = prevalenze (0-1) delle classi di rischio basso / intermedio /
    alto nella cella (terzili del rank dell'indice aggregato JRC);
  - celle non bruciabili (urbano denso, acqua) = NaN in tutti e tre i file.

Cosa fa:
  1. DROP della vecchia tabella rischio_incendio_lombardia (zone AIB, solo
     Milano): sostituita integralmente da questa fonte per evitare
     sovrapposizioni tra due variabili incendio disomogenee.
  2. Carica le celle nel bounding box italiano nella tabella
     rischio_incendio_effis (fire_low/fire_int/fire_high NULL = non bruciabile).
  3. Assegna ogni particella alla cella che la contiene (test punto-in-
     quadrilatero) e, per le particelle in celle non bruciabili, cerca la
     cella bruciabile piu' vicina entro MAX_NEAR_KM: tabella
     parcels_rischio_incendio con valori diretti (fire_*) e di prossimita'
     (fire_*_near, fire_near_dist_km).

fire_score = 0.5*fire_int + 1.0*fire_high  (0-1, comodo come variabile unica).

Uso:
    python integra_rischio_incendio.py
"""
import csv
import math
import os
import sqlite3
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "zornade_data.db")
CSV_DIR = os.path.join(BASE, "dati_incendio_effis")
CSV_TPL = "var-risk-aggr-l3p122_unit-dimensionless_stat-rank_class3-{n}.csv"
# ordine delle classi nei tre file, verificato empiricamente (vedi README)
CLASS_ORDER = {1: "low", 2: "int", 3: "high"}

# bounding box Italia con margine (le 4 province sono ben all'interno)
LON_MIN, LON_MAX = 5.5, 19.5
LAT_MIN, LAT_MAX = 35.0, 48.0

BUCKET = 0.5        # lato dei bucket spaziali in gradi
MAX_NEAR_KM = 25.0  # raggio massimo di ricerca della cella bruciabile vicina


def parse_val(s):
    return None if s == "NaN" else float(s)


def load_cells():
    """Legge i 3 CSV e restituisce {id: cella} per il solo bbox italiano."""
    cells = {}
    for n, cls in CLASS_ORDER.items():
        path = os.path.join(CSV_DIR, CSV_TPL.format(n=n))
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # header: % id, lon-1..4, lat-1..4, x-pos, y-pos, value
            for row in reader:
                cid = int(row[0])
                lons = [float(x) for x in row[1:5]]
                lats = [float(x) for x in row[5:9]]
                clon = sum(lons) / 4.0
                clat = sum(lats) / 4.0
                if not (LON_MIN <= clon <= LON_MAX and LAT_MIN <= clat <= LAT_MAX):
                    continue
                c = cells.get(cid)
                if c is None:
                    c = cells[cid] = {
                        "id": cid, "x": int(row[9]), "y": int(row[10]),
                        "lons": lons, "lats": lats,
                        "clon": clon, "clat": clat,
                        "low": None, "int": None, "high": None,
                    }
                c[cls] = parse_val(row[11])
    return cells


def point_in_quad(lon, lat, lons, lats):
    """Il punto sta nel quadrilatero convesso (vertici in ordine di anello)?"""
    sign = 0
    for i in range(4):
        j = (i + 1) % 4
        cross = ((lons[j] - lons[i]) * (lat - lats[i])
                 - (lats[j] - lats[i]) * (lon - lons[i]))
        if cross == 0:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def bucket_key(lon, lat):
    return (int(math.floor(lon / BUCKET)), int(math.floor(lat / BUCKET)))


def fire_score(c):
    if c["high"] is None:
        return None
    return 0.5 * c["int"] + 1.0 * c["high"]


def main():
    print("Carico le celle EFFIS (bbox Italia)...")
    cells = load_cells()
    burnable = [c for c in cells.values() if c["high"] is not None]
    print(f"  celle nel bbox: {len(cells):,}; bruciabili: {len(burnable):,}")

    # indici spaziali: bucket -> celle il cui bbox tocca il bucket (containment)
    # e bucket -> celle bruciabili per centroide (nearest)
    contain_idx = defaultdict(list)
    for c in cells.values():
        bmin = bucket_key(min(c["lons"]), min(c["lats"]))
        bmax = bucket_key(max(c["lons"]), max(c["lats"]))
        for bx in range(bmin[0], bmax[0] + 1):
            for by in range(bmin[1], bmax[1] + 1):
                contain_idx[(bx, by)].append(c)
    near_idx = defaultdict(list)
    for c in burnable:
        near_idx[bucket_key(c["clon"], c["clat"])].append(c)

    def find_cell(lon, lat):
        for c in contain_idx.get(bucket_key(lon, lat), ()):
            if point_in_quad(lon, lat, c["lons"], c["lats"]):
                return c
        return None

    def find_nearest_burnable(lon, lat):
        bx, by = bucket_key(lon, lat)
        rings = int(math.ceil(MAX_NEAR_KM / (BUCKET * 111.0 * 0.7))) + 1
        best, best_d = None, None
        for ring in range(rings + 1):
            for dx in range(-ring, ring + 1):
                for dy in range(-ring, ring + 1):
                    if max(abs(dx), abs(dy)) != ring:
                        continue
                    for c in near_idx.get((bx + dx, by + dy), ()):
                        d = haversine_km(lat, lon, c["clat"], c["clon"])
                        if best_d is None or d < best_d:
                            best, best_d = c, d
            # se ho gia' un candidato piu' vicino del bordo interno del
            # prossimo anello, posso fermarmi
            if best_d is not None and best_d <= ring * BUCKET * 111.0 * 0.7:
                break
        if best is not None and best_d <= MAX_NEAR_KM:
            return best, best_d
        return None, None

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    print("Rimuovo la vecchia tabella rischio_incendio_lombardia (solo Milano)...")
    cur.executescript("""
        DROP TABLE IF EXISTS rischio_incendio_lombardia;
        DROP TABLE IF EXISTS rischio_incendio_effis;
        DROP TABLE IF EXISTS parcels_rischio_incendio;
        CREATE TABLE rischio_incendio_effis (
            cell_id INTEGER PRIMARY KEY,   -- id nella griglia EUR-11 completa
            x_pos INTEGER, y_pos INTEGER,
            lon_c REAL, lat_c REAL,        -- centroide
            lon_1 REAL, lon_2 REAL, lon_3 REAL, lon_4 REAL,
            lat_1 REAL, lat_2 REAL, lat_3 REAL, lat_4 REAL,
            fire_low REAL,                 -- prevalenza classe rischio basso
            fire_int REAL,                 -- prevalenza classe rischio intermedio
            fire_high REAL,                -- prevalenza classe rischio alto
            fire_score REAL                -- 0.5*int + 1.0*high
        );
        CREATE TABLE parcels_rischio_incendio (
            fid INTEGER PRIMARY KEY,
            cell_id INTEGER,               -- cella contenente (NULL se fuori griglia)
            fire_low REAL, fire_int REAL, fire_high REAL, fire_score REAL,
            near_cell_id INTEGER,          -- cella bruciabile piu' vicina (<=25 km)
            fire_low_near REAL, fire_int_near REAL, fire_high_near REAL,
            fire_score_near REAL,
            fire_near_dist_km REAL
        );
    """)

    cur.executemany(
        "INSERT INTO rischio_incendio_effis VALUES "
        "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [(c["id"], c["x"], c["y"], c["clon"], c["clat"],
          *c["lons"], *c["lats"],
          c["low"], c["int"], c["high"], fire_score(c))
         for c in cells.values()])
    print(f"  inserite {len(cells):,} celle in rischio_incendio_effis")

    print("Assegno le particelle alle celle...")
    parcels = cur.execute(
        "SELECT fid, lat, lng, province FROM parcels "
        "WHERE lat IS NOT NULL AND lng IS NOT NULL").fetchall()

    rows = []
    stats = defaultdict(lambda: [0, 0, 0, 0.0, 0])  # prov -> [tot, in_cella, dirette, somma_score, con_score]
    for fid, lat, lng, prov in parcels:
        c = find_cell(lng, lat)
        st = stats[prov]
        st[0] += 1
        direct = (None,) * 5
        if c is not None:
            st[1] += 1
            direct = (c["id"], c["low"], c["int"], c["high"], fire_score(c))
            if c["high"] is not None:
                st[2] += 1
        near = (None,) * 6
        if c is None or c["high"] is None:
            nc, nd = find_nearest_burnable(lng, lat)
            if nc is not None:
                near = (nc["id"], nc["low"], nc["int"], nc["high"],
                        fire_score(nc), round(nd, 2))
        else:
            # la cella contenente e' gia' bruciabile: nessuna ricerca necessaria
            near = (c["id"], c["low"], c["int"], c["high"], fire_score(c), 0.0)
        eff_score = direct[4] if direct[4] is not None else near[4]
        if eff_score is not None:
            st[3] += eff_score
            st[4] += 1
        rows.append((fid, *direct, *near))

    cur.executemany(
        "INSERT INTO parcels_rischio_incendio VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        rows)
    conn.commit()

    print(f"  inserite {len(rows):,} particelle in parcels_rischio_incendio\n")
    print(f"{'provincia':<10} {'particelle':>10} {'in cella':>9} {'bruciabile':>10} "
          f"{'con valore':>10} {'score medio':>11}")
    for prov in sorted(stats):
        tot, incell, direct, ssum, swith = stats[prov]
        print(f"{prov:<10} {tot:>10,} {incell:>9,} {direct:>10,} "
              f"{swith:>10,} {ssum / swith if swith else float('nan'):>11.3f}")

    conn.close()
    print("\n[OK] Integrazione completata.")


if __name__ == "__main__":
    main()
