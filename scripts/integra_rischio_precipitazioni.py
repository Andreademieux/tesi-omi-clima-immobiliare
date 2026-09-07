#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integra nel database Zornade la climatologia 1995-2019 dell'indice di
precipitazioni estreme rr20mm (Copernicus C3S / CMCC-KNMI, E-OBS), come
proxy di esposizione di lungo periodo al rischio pluviale, distinto dai
perimetri di pericolosita' PAI/PGRA statici gia' integrati (risk.flood_level).

Fonte e formato: vedi dati_precipitazioni_eobs/README.md. In sintesi:
  - griglia regolare E-OBS 0.1grad (~11 km), lat/lon;
  - rr20mm = numero di giorni/anno con precipitazione >=20mm, 25 annualita'
    (1995-2019); le celle marine/prive di stazioni sono NaN.

Cosa fa:
  1. Calcola la media climatologica 1995-2019 di rr20mm per ogni cella della
     griglia (skipna sulle annualita' mancanti in quella cella).
  2. Carica le celle nel bounding box italiano nella tabella
     precipitazioni_eobs_griglia.
  3. Attribuisce a ogni particella la cella di griglia piu' vicina
     (nearest-neighbor sulla griglia regolare) e salva la distanza
     particella-centro cella (km), utile per la discussione di attenuation
     bias da errore di misura (cfr. protocollo di analisi, S11): tabella
     parcels_rischio_precipitazioni.

Uso:
    python integra_rischio_precipitazioni.py
"""
import math
import os
import sqlite3

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "zornade_data.db")
NC_PATH = os.path.join(BASE, "dati_precipitazioni_eobs", "rr20mm_1995_2019.nc")

# bounding box Italia con margine (le 4 province sono ben all'interno)
LON_MIN, LON_MAX = 5.5, 19.5
LAT_MIN, LAT_MAX = 35.0, 48.0


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def main():
    print("Carico e aggrego la climatologia E-OBS rr20mm 1995-2019...")
    ds = xr.open_dataset(NC_PATH)
    n_years_total = ds.sizes["time"]
    mean_da = ds["rr20mm"].mean(dim="time", skipna=True)
    n_valid_da = ds["rr20mm"].notnull().sum(dim="time")

    lat_vals = ds["latitude"].values
    lon_vals = ds["longitude"].values
    lat_mask = (lat_vals >= LAT_MIN) & (lat_vals <= LAT_MAX)
    lon_mask = (lon_vals >= LON_MIN) & (lon_vals <= LON_MAX)
    lat_idx = np.nonzero(lat_mask)[0]
    lon_idx = np.nonzero(lon_mask)[0]

    sub_mean = mean_da.values[np.ix_(lat_idx, lon_idx)]
    sub_nvalid = n_valid_da.values[np.ix_(lat_idx, lon_idx)]
    sub_lat = lat_vals[lat_idx]
    sub_lon = lon_vals[lon_idx]

    lat_step = float(np.diff(lat_vals).mean())
    lon_step = float(np.diff(lon_vals).mean())
    print(f"  griglia: {len(sub_lat)} x {len(sub_lon)} celle nel bbox Italia "
          f"(passo lat={lat_step:.3f}, lon={lon_step:.3f})")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.executescript("""
        DROP TABLE IF EXISTS precipitazioni_eobs_griglia;
        DROP TABLE IF EXISTS parcels_rischio_precipitazioni;
        CREATE TABLE precipitazioni_eobs_griglia (
            cell_id INTEGER PRIMARY KEY,
            lat_c REAL,
            lon_c REAL,
            rr20mm_mean_1995_2019 REAL,   -- media climatologica, NULL se cella marina/senza dato
            n_anni_validi INTEGER          -- su 25 annualita' disponibili (1995-2019)
        );
        CREATE TABLE parcels_rischio_precipitazioni (
            fid INTEGER PRIMARY KEY,
            cell_id INTEGER,
            rr20mm_mean_1995_2019 REAL,
            dist_km REAL                   -- distanza particella-centro cella nearest
        );
    """)

    cells = []
    cell_id = 0
    cell_lookup = {}  # (i_lat, i_lon) -> cell_id, per ricostruire indice nearest veloce
    for i, la in enumerate(sub_lat):
        for j, lo in enumerate(sub_lon):
            val = sub_mean[i, j]
            nval = int(sub_nvalid[i, j])
            cells.append((cell_id, float(la), float(lo),
                          None if np.isnan(val) else float(val), nval))
            cell_lookup[(i, j)] = cell_id
            cell_id += 1
    cur.executemany(
        "INSERT INTO precipitazioni_eobs_griglia VALUES (?,?,?,?,?)", cells)
    print(f"  inserite {len(cells):,} celle in precipitazioni_eobs_griglia")

    n_with_data = sum(1 for c in cells if c[3] is not None)
    print(f"  celle con dato valido (non NaN): {n_with_data:,} / {len(cells):,}")

    print("Assegno le particelle alla cella di griglia valida piu' vicina (KD-tree)...")
    # Coordinate scalate (x = lon*cos(lat_media), y = lat) per un KD-tree in
    # gradi che approssima bene le distanze reali alle latitudini italiane;
    # la distanza riportata e' comunque ricalcolata con haversine esatta.
    mean_lat_rad = math.radians(float(np.mean(sub_lat)))
    cos_mean_lat = math.cos(mean_lat_rad)

    valid_cells = [c for c in cells if c[3] is not None]
    tree_points = np.array([[c[2] * cos_mean_lat, c[1]] for c in valid_cells])
    tree = cKDTree(tree_points)
    valid_ids = [c[0] for c in valid_cells]
    valid_latlon = {c[0]: (c[1], c[2]) for c in valid_cells}
    valid_val = {c[0]: c[3] for c in valid_cells}

    parcels = cur.execute(
        "SELECT fid, lat, lng, province FROM parcels "
        "WHERE lat IS NOT NULL AND lng IS NOT NULL").fetchall()
    query_points = np.array([[lng * cos_mean_lat, lat] for _, lat, lng, _ in parcels])
    _, nn_idx = tree.query(query_points, k=1)

    rows = []
    dist_list = []
    from collections import defaultdict
    stats = defaultdict(lambda: [0, 0, 0.0])  # prov -> [tot, con_valore, somma_dist]
    for (fid, lat, lng, prov), idx in zip(parcels, nn_idx):
        cid = valid_ids[idx]
        clat, clon = valid_latlon[cid]
        val = valid_val[cid]
        d = haversine_km(lat, lng, clat, clon)
        rows.append((fid, cid, val, round(d, 3)))
        st = stats[prov]
        st[0] += 1
        st[1] += 1
        st[2] += d
        dist_list.append(d)

    cur.executemany(
        "INSERT INTO parcels_rischio_precipitazioni VALUES (?,?,?,?)", rows)
    conn.commit()

    print(f"  inserite {len(rows):,} particelle in parcels_rischio_precipitazioni\n")
    print(f"{'provincia':<10} {'particelle':>10} {'con valore':>10} {'dist media km':>14}")
    for prov in sorted(stats):
        tot, withv, sd = stats[prov]
        print(f"{prov:<10} {tot:>10,} {withv:>10,} {sd/withv if withv else float('nan'):>14.3f}")

    dist_arr = np.array(dist_list)
    print(f"\nDistanza particella-nearest grid point (km): "
          f"media={dist_arr.mean():.3f}  mediana={np.median(dist_arr):.3f}  "
          f"p95={np.percentile(dist_arr,95):.3f}  max={dist_arr.max():.3f}")

    conn.close()
    print("\n[OK] Integrazione completata.")


if __name__ == "__main__":
    main()
