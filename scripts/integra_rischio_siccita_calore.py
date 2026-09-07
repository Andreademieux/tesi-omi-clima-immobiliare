#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integra nel database Zornade gli indicatori di siccita' e ondate di calore
della Piattaforma Indicatori Climatici SNPA/ISPRA (dati prodotti da
Fondazione CMCC, REMHI Division; climatologia E-OBS 1981-2010, griglia
regolare 0.1 grad ~11 km, stessa famiglia di dato gia' in uso per
r99pday/rr20mm), come estensione del set di rischi meteo-climatici gia'
integrato con integra_rischio_precipitazioni.py / integra_rischio_p99.py.

Variabili integrate (§4.2 del protocollo, "eventi meteo estremi"):

  SICCITA'
    headline    cdd     giorni consecutivi secchi (prec. < 1mm/giorno) - RAW
    robustezza  pet_tw  evapotraspirazione potenziale annua (mm) - RAW

    Nota: SPI12 e' stato valutato ed escluso. Il pacchetto ISPRA/SNPA lo
    fornisce solo come 7 raster di classe (spi12_c1..c7, presumibilmente
    frequenza di anni in ciascuna classe di severita' 1981-2010), senza
    documentazione che ne chiarisca l'ordinamento (extreme_dry -> extreme_wet
    o viceversa): un errore di verso capovolgerebbe il segno del coefficiente.
    E' inoltre un indice gia' standardizzato (§4.2: "non indici sintetici
    costruiti a monte"). CDD e PET restano quindi le due variabili grezze
    coerenti col protocollo.

  CALORE (ondate di calore, stessa macro-categoria "eventi meteo estremi")
    headline    su95p   giorni estivi oltre soglia percentile 95 - RAW,
                        stessa logica metodologica dell'headline pr99prctile
                        (soglia percentile) gia' usata per le precipitazioni
    robustezza  tr      notti tropicali, soglia fissa (Tmin>=20 C) - RAW,
                        analogo a rr20mm (soglia fissa) come variante
                        dell'headline a soglia percentile
    robustezza  tg      temperatura media annua - RAW

Metodo: identico a integra_rischio_p99.py (nearest-neighbor su cella di
griglia valida via cKDTree, distanza haversine particella-centro cella
salvata per la discussione di attenuation bias, §11), adattato per leggere
GeoTIFF (rasterio) invece di NetCDF essendo questo il formato di
distribuzione ISPRA/SNPA. Griglia e bounding box sono gli stessi (0.1 grad,
Italia) del prodotto E-OBS gia' in uso: le celle sono quindi direttamente
confrontabili con quelle di precipitazioni_p99_griglia.

Uso:
    python integra_rischio_siccita_calore.py
"""
import math
import os
import sqlite3
from collections import defaultdict

import numpy as np
import rasterio
from scipy.spatial import cKDTree

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "zornade_data.db")
LAYERS_DIR = r"C:\tmp\clima\layers"

# (nome_variabile, file_tif, ruolo, tabella_griglia, tabella_parcels)
LAYERS = [
    ("cdd", os.path.join(LAYERS_DIR, "cdd_e-obs_year_italy_1981-2010_nan", "cdd.tif"),
     "headline_siccita", "siccita_cdd_griglia", "parcels_rischio_siccita_cdd"),
    ("pet_tw", os.path.join(LAYERS_DIR, "pet_tw_e-obs_year_italy_1981-2010_nan", "pet_tw.tif"),
     "robustezza_siccita", "siccita_pet_griglia", "parcels_rischio_siccita_pet"),
    ("su95p", os.path.join(LAYERS_DIR, "su95p_e-obs_year_italy_1981-2010_nan", "su95p.tif"),
     "headline_calore", "calore_su95p_griglia", "parcels_rischio_calore_su95p"),
    ("tr", os.path.join(LAYERS_DIR, "tr_e-obs_year_italy_1981-2010_nan", "tr.tif"),
     "robustezza_calore", "calore_tr_griglia", "parcels_rischio_calore_tr"),
    ("tg", os.path.join(LAYERS_DIR, "mean-temperature_e-obs_year_italy_1981-2010_nan", "tg.tif"),
     "robustezza_calore", "calore_tg_griglia", "parcels_rischio_calore_tg"),
]


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def integra_layer(conn, cur, var_name, tif_path, ruolo, tab_griglia, tab_parcels, parcels):
    print(f"\n=== {var_name} ({ruolo}) — {os.path.basename(tif_path)} ===")
    with rasterio.open(tif_path) as ds:
        arr = ds.read(1).astype(float)
        nodata = ds.nodata
        transform = ds.transform
        nrows, ncols = arr.shape
        # centro pixel (i,j): riga i dall'alto, colonna j da sinistra
        cols_idx = np.arange(ncols) + 0.5
        rows_idx = np.arange(nrows) + 0.5
        lon_centers, _ = rasterio.transform.xy(transform, [0] * ncols, cols_idx, offset="center")
        _, lat_centers = rasterio.transform.xy(transform, rows_idx, [0] * nrows, offset="center")
        lon_centers = np.array(lon_centers)
        lat_centers = np.array(lat_centers)
        print(f"  griglia: {nrows} x {ncols} celle, pixel={transform[0]:.3f} grad, "
              f"lat[{lat_centers.min():.2f},{lat_centers.max():.2f}] "
              f"lon[{lon_centers.min():.2f},{lon_centers.max():.2f}]")

    cur.executescript(f"""
        DROP TABLE IF EXISTS {tab_griglia};
        DROP TABLE IF EXISTS {tab_parcels};
        CREATE TABLE {tab_griglia} (
            cell_id INTEGER PRIMARY KEY,
            lat_c REAL,
            lon_c REAL,
            {var_name} REAL   -- NULL se cella marina/priva di dato (nodata)
        );
        CREATE TABLE {tab_parcels} (
            fid INTEGER PRIMARY KEY,
            cell_id INTEGER,
            {var_name} REAL,
            dist_km REAL      -- distanza particella-centro cella nearest (S11)
        );
    """)

    cells = []
    cell_id = 0
    for i in range(nrows):
        for j in range(ncols):
            val = arr[i, j]
            is_nodata = (nodata is not None and val == nodata) or np.isnan(val)
            cells.append((cell_id, float(lat_centers[i]), float(lon_centers[j]),
                          None if is_nodata else float(val)))
            cell_id += 1
    cur.executemany(f"INSERT INTO {tab_griglia} VALUES (?,?,?,?)", cells)
    n_with_data = sum(1 for c in cells if c[3] is not None)
    print(f"  inserite {len(cells):,} celle ({n_with_data:,} con dato valido)")

    valid_cells = [c for c in cells if c[3] is not None]
    mean_lat_rad = math.radians(float(np.mean([c[1] for c in valid_cells])))
    cos_mean_lat = math.cos(mean_lat_rad)
    tree = cKDTree(np.array([[c[2] * cos_mean_lat, c[1]] for c in valid_cells]))
    valid_ids = [c[0] for c in valid_cells]
    valid_latlon = {c[0]: (c[1], c[2]) for c in valid_cells}
    valid_val = {c[0]: c[3] for c in valid_cells}

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

    cur.executemany(f"INSERT INTO {tab_parcels} VALUES (?,?,?,?)", rows)
    conn.commit()
    print(f"  inserite {len(rows):,} particelle in {tab_parcels}")
    for prov in sorted(stats):
        tot, sd = stats[prov]
        print(f"    {prov:<10} {tot:>7,} particelle  dist media {sd/tot:.3f} km")
    dist_arr = np.array(dist_list)
    print(f"  distanza particella-cella (km): media={dist_arr.mean():.3f} "
          f"mediana={np.median(dist_arr):.3f} p95={np.percentile(dist_arr,95):.3f}")

    vals = np.array([v for v in valid_val.values()])
    print(f"  {var_name} sulla griglia Italia: min={vals.min():.2f} mean={vals.mean():.2f} "
          f"max={vals.max():.2f}")


def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    parcels = cur.execute(
        "SELECT fid, lat, lng, province FROM parcels "
        "WHERE lat IS NOT NULL AND lng IS NOT NULL").fetchall()
    print(f"Particelle con coordinate: {len(parcels):,}")

    for var_name, tif_path, ruolo, tab_griglia, tab_parcels in LAYERS:
        integra_layer(conn, cur, var_name, tif_path, ruolo, tab_griglia, tab_parcels, parcels)

    conn.close()
    print("\n[OK] Integrazione siccita'/calore completata.")


if __name__ == "__main__":
    main()
