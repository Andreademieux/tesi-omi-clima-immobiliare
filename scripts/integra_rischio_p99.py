#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integra nel database Zornade la climatologia 1995-2019 dell'indicatore di
precipitazioni estreme r99pday (Copernicus C3S / CMCC-KNMI, E-OBS):
numero di giorni per anno con precipitazione giornaliera superiore al
99mo percentile LOCALE della distribuzione dei giorni piovosi della cella
(standard_name: precipitation_99-percentile_frequency, unita': giorni).

Questa e' la variabile HEADLINE per il rischio da precipitazioni estreme
(protocollo di studio, S4.2); rr20mm (soglia assoluta 20mm/giorno, gia'
integrata con integra_rischio_precipitazioni.py) resta come variante di
robustezza. Nota metodologica da riportare (S4.2/S11): r99pday e' un
indicatore RELATIVO — il percentile e' calcolato cella per cella, quindi
la media climatologica e' ~1 giorno/anno per costruzione e la variazione
cross-section (0.4-2.3 gg/anno in Italia, CV~0.27) riflette differenze
nella numerosita' dei giorni piovosi e negli scostamenti dal periodo di
riferimento, non l'intensita' assoluta dell'esposizione; rr20mm (CV~0.64)
cattura invece quest'ultima. Correlazione di griglia tra i due sull'Italia:
Pearson ~0.37.

Fonte e formato: identici a dati_precipitazioni_eobs/README.md (stesso
prodotto C3S_430 "Extreme precipitation indicators for Europe...", v1.0,
griglia regolare E-OBS 0.1 grad, ~11 km). File nella root del progetto:
r99pday_europe-extreme-precipitation-risk-indicators_yearly_e-obs_394353.nc
(30 annualita' 1990-2019; per coerenza con rr20mm si usa la finestra
1995-2019, 25 annualita').

Cosa fa (replica esatta della procedura rr20mm):
  1. media climatologica 1995-2019 di r99pday per cella (skipna);
  2. celle nel bbox Italia -> tabella precipitazioni_p99_griglia;
  3. nearest-neighbor valido per particella -> tabella parcels_rischio_p99,
     con distanza particella-centro cella (km) per la discussione di
     attenuation bias (S11).

Uso:
    py integra_rischio_p99.py
"""
import math
import os
import sqlite3
from collections import defaultdict

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "zornade_data.db")
NC_PATH = os.path.join(
    BASE, "r99pday_europe-extreme-precipitation-risk-indicators_yearly_e-obs_394353.nc")

LON_MIN, LON_MAX = 5.5, 19.5
LAT_MIN, LAT_MAX = 35.0, 48.0
T_MIN, T_MAX = "1995-01-01", "2019-12-31"


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def main():
    print("Carico e aggrego la climatologia E-OBS r99pday 1995-2019...")
    ds = xr.open_dataset(NC_PATH)
    da = ds["r99pday"].sel(time=slice(T_MIN, T_MAX))
    print(f"  annualita' usate: {da.sizes['time']} (finestra {T_MIN[:4]}-{T_MAX[:4]})")
    mean_da = da.mean(dim="time", skipna=True)
    n_valid_da = da.notnull().sum(dim="time")

    lat_vals = ds["latitude"].values
    lon_vals = ds["longitude"].values
    lat_idx = np.nonzero((lat_vals >= LAT_MIN) & (lat_vals <= LAT_MAX))[0]
    lon_idx = np.nonzero((lon_vals >= LON_MIN) & (lon_vals <= LON_MAX))[0]

    sub_mean = mean_da.values[np.ix_(lat_idx, lon_idx)]
    sub_nvalid = n_valid_da.values[np.ix_(lat_idx, lon_idx)]
    sub_lat = lat_vals[lat_idx]
    sub_lon = lon_vals[lon_idx]
    print(f"  griglia: {len(sub_lat)} x {len(sub_lon)} celle nel bbox Italia")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.executescript("""
        DROP TABLE IF EXISTS precipitazioni_p99_griglia;
        DROP TABLE IF EXISTS parcels_rischio_p99;
        CREATE TABLE precipitazioni_p99_griglia (
            cell_id INTEGER PRIMARY KEY,
            lat_c REAL,
            lon_c REAL,
            r99pday_mean_1995_2019 REAL,  -- media climatologica, NULL se cella marina/senza dato
            n_anni_validi INTEGER          -- su 25 annualita' (1995-2019)
        );
        CREATE TABLE parcels_rischio_p99 (
            fid INTEGER PRIMARY KEY,
            cell_id INTEGER,
            r99pday_mean_1995_2019 REAL,
            dist_km REAL                   -- distanza particella-centro cella nearest
        );
    """)

    cells = []
    cell_id = 0
    for i, la in enumerate(sub_lat):
        for j, lo in enumerate(sub_lon):
            val = sub_mean[i, j]
            cells.append((cell_id, float(la), float(lo),
                          None if np.isnan(val) else float(val),
                          int(sub_nvalid[i, j])))
            cell_id += 1
    cur.executemany("INSERT INTO precipitazioni_p99_griglia VALUES (?,?,?,?,?)", cells)
    n_with_data = sum(1 for c in cells if c[3] is not None)
    print(f"  inserite {len(cells):,} celle ({n_with_data:,} con dato valido)")

    print("Assegno le particelle alla cella valida piu' vicina (KD-tree)...")
    mean_lat_rad = math.radians(float(np.mean(sub_lat)))
    cos_mean_lat = math.cos(mean_lat_rad)
    valid_cells = [c for c in cells if c[3] is not None]
    tree = cKDTree(np.array([[c[2] * cos_mean_lat, c[1]] for c in valid_cells]))
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
    stats = defaultdict(lambda: [0, 0.0])
    for (fid, lat, lng, prov), idx in zip(parcels, nn_idx):
        cid = valid_ids[idx]
        clat, clon = valid_latlon[cid]
        d = haversine_km(lat, lng, clat, clon)
        rows.append((fid, cid, valid_val[cid], round(d, 3)))
        stats[prov][0] += 1
        stats[prov][1] += d
        dist_list.append(d)

    cur.executemany("INSERT INTO parcels_rischio_p99 VALUES (?,?,?,?)", rows)
    conn.commit()
    print(f"  inserite {len(rows):,} particelle in parcels_rischio_p99\n")
    print(f"{'provincia':<10} {'particelle':>10} {'dist media km':>14}")
    for prov in sorted(stats):
        tot, sd = stats[prov]
        print(f"{prov:<10} {tot:>10,} {sd/tot:>14.3f}")

    dist_arr = np.array(dist_list)
    print(f"\nDistanza particella-nearest grid point (km): "
          f"media={dist_arr.mean():.3f}  mediana={np.median(dist_arr):.3f}  "
          f"p95={np.percentile(dist_arr,95):.3f}  max={dist_arr.max():.3f}")

    conn.close()
    print("\n[OK] Integrazione r99pday completata.")


if __name__ == "__main__":
    main()
