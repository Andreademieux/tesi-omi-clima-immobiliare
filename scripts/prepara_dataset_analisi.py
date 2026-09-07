#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tranche 1 - Costruzione del dataset analitico e trattamento della
pseudo-replicazione (protocollo di studio: unita' statistica, S3;
tipologie edilizie, S5; EDA e missing data, S6; fonti di rischio, S4.2).

Pipeline:
  1. Legge parcels + join con parcels_rischio_incendio,
     parcels_rischio_precipitazioni (rr20mm, variante di robustezza) e
     parcels_rischio_p99 (r99pday, variabile headline precipitazioni
     estreme, S4.2).
  2. Usa detail_json.data.municipality come verita' di base per comune e
     provincia (il campo parcels.comune/province riflette il comune usato
     per la query di raccolta, non necessariamente quello catastale reale:
     confermato un boundary-leakage di 182 particelle la cui provincia vera
     e' fuori dalle 4 target, e di ~2.290 particelle con comune vero diverso
     nella stessa provincia). Le particelle fuori dalle 4 province target
     vengono escluse.
  3. Costruisce gli indicatori di esposizione al rischio a livello di
     particella (si veda la tabella fonti in fondo allo script).
  4. Esplode le voci di valutazione OMI (una particella puo' avere piu'
     tipologie edilizie nella stessa zona) in formato lungo
     particella x tipologia x condizione.
  5. Salva il dataset a livello di particella (output/particelle.csv, uso
     esplorativo/sensitivity) e il dataset aggregato all'unita' statistica
     corretta provincia x comune x zona OMI x tipologia x condizione
     (output/celle.csv, dataset primario per la stima).

Uso:
    python prepara_dataset_analisi.py
"""
import json
import math
import os
import sqlite3
from collections import Counter

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "zornade_data.db")
OUT_DIR = os.path.join(BASE, "output")

TARGET_PROVINCES = {"Milano", "Napoli", "Roma", "Torino"}

FLOOD_ORD = {"LPH": 1, "MPH": 2, "HPH": 3}   # PGRA: bassa/media/alta probabilita'
LANDSLIDE_ORD = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}   # PAI, esclusa AA (natura diversa)

# Coordinate del capoluogo di provincia (centro citta'), per un controllo di
# distanza dal CBD indipendente da fonti esterne (S4.3 variabili territoriali).
CAPOLUOGO = {
    "Milano": (45.4642, 9.1900),
    "Napoli": (40.8518, 14.2681),
    "Roma": (41.9028, 12.4964),
    "Torino": (45.0703, 7.6869),
}


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def norm_apostrophe(s):
    return s.replace("`", "'").strip() if s else s


def load_rows():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT p.fid, p.comune AS comune_raccolta, p.province AS provincia_raccolta,
               p.area_m2, p.lat, p.lng, p.detail_json,
               fi.fire_score, fi.fire_score_near, fi.fire_near_dist_km,
               pr.rr20mm_mean_1995_2019, pr.dist_km AS precip_dist_km,
               p99.r99pday_mean_1995_2019, p99.dist_km AS p99_dist_km
        FROM parcels p
        LEFT JOIN parcels_rischio_incendio fi ON p.fid = fi.fid
        LEFT JOIN parcels_rischio_precipitazioni pr ON p.fid = pr.fid
        LEFT JOIN parcels_rischio_p99 p99 ON p.fid = p99.fid
    """)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    conn.close()
    return rows, cols


def build_particle_df(rows, cols):
    idx = {c: i for i, c in enumerate(cols)}
    records = []
    dropped_outside_target = Counter()
    comune_corretto = 0

    for r in rows:
        dj = json.loads(r[idx["detail_json"]])["data"]
        muni = dj.get("municipality") or {}
        true_comune = norm_apostrophe(muni.get("name"))
        true_prov = muni.get("province")
        coll_prov = r[idx["provincia_raccolta"]]
        coll_comune = r[idx["comune_raccolta"]]

        if true_prov not in TARGET_PROVINCES:
            dropped_outside_target[(coll_prov, true_prov)] += 1
            continue
        if true_comune and coll_comune and true_comune.upper() != coll_comune.strip().upper():
            comune_corretto += 1

        risk = dj.get("risk") or {}
        sub = dj.get("subsidence") or {}
        lc = dj.get("land_cover") or {}
        ce_list = dj.get("coastal_erosion") or []
        val_list = dj.get("valuation") or []

        flood_level = risk.get("flood_level")
        flood_ord = FLOOD_ORD.get(flood_level, 0)
        landslide_level = risk.get("landslide_level")
        landslide_aa = 1 if landslide_level == "AA" else 0
        landslide_ord = LANDSLIDE_ORD.get(landslide_level, 0)

        fire_direct = r[idx["fire_score"]]
        fire_near = r[idx["fire_score_near"]]
        fire_score = fire_direct if fire_direct is not None else fire_near

        ce_dists = [c["distance_m"] for c in ce_list if c.get("distance_m") is not None]
        ce_changes = [abs(c["avg_change_m_year"]) for c in ce_list if c.get("avg_change_m_year") is not None]

        lat, lng = r[idx["lat"]], r[idx["lng"]]
        cap = CAPOLUOGO.get(true_prov)
        dist_cbd_km = haversine_km(lat, lng, cap[0], cap[1]) if (cap and lat is not None and lng is not None) else None

        base = dict(
            fid=r[idx["fid"]],
            comune_raccolta=coll_comune, provincia_raccolta=coll_prov,
            comune=true_comune, provincia=true_prov,
            area_m2=r[idx["area_m2"]], lat=lat, lng=lng,
            dist_cbd_km=dist_cbd_km,
            seismic_zone=risk.get("seismic_zone"),
            pga=risk.get("pga"),
            flood_level=flood_level, flood_ord=flood_ord,
            flood_any=1 if flood_ord > 0 else 0,
            landslide_level=landslide_level, landslide_ord=landslide_ord,
            landslide_any=1 if landslide_ord > 0 else 0,
            landslide_aa=landslide_aa,
            subsidence_velocity=sub.get("velocity_mm_year"),
            subsidence_risk_class=sub.get("risk_class"),
            fire_score=fire_score,
            fire_is_direct=1 if fire_direct is not None else 0,
            fire_near_dist_km=r[idx["fire_near_dist_km"]],
            rr20mm=r[idx["rr20mm_mean_1995_2019"]],
            precip_dist_km=r[idx["precip_dist_km"]],
            r99pday=r[idx["r99pday_mean_1995_2019"]],
            p99_dist_km=r[idx["p99_dist_km"]],
            coastal_dist_m=min(ce_dists) if ce_dists else None,
            coastal_change_m_year=max(ce_changes) if ce_changes else None,
            is_coastal_zone=1 if ce_list else 0,
            land_cover_class=lc.get("class"),
            is_urban=1 if lc.get("class") == "Superfici artificiali" else 0,
        )

        if not val_list:
            rec = dict(base)
            rec.update(zone=None, zone_description=None, fascia=None, microzona=None,
                       property_type=None, condition=None,
                       price_m2=None, rental_m2=None, price_m2_prev=None)
            records.append(rec)
            continue

        for v in val_list:
            purch = v.get("purchase") or {}
            rent = v.get("rental") or {}
            prev_purch = (v.get("previous_semester") or {}).get("purchase") or {}

            def midpoint(d):
                if d.get("min_eur_m2") is not None and d.get("max_eur_m2") is not None:
                    return (d["min_eur_m2"] + d["max_eur_m2"]) / 2.0
                return None

            rec = dict(base)
            rec.update(
                zone=v.get("zone"), zone_description=v.get("zone_description"),
                fascia=v.get("fascia"), microzona=v.get("microzona"),
                property_type=v.get("property_type"), condition=v.get("condition"),
                price_m2=midpoint(purch), rental_m2=midpoint(rent),
                price_m2_prev=midpoint(prev_purch),
            )
            records.append(rec)

    df = pd.DataFrame.from_records(records)
    return df, dropped_outside_target, comune_corretto


def load_cri():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT provincia_nome AS provincia, cri_2021_2050_med, cri_2071_2100_med, aci "
        "FROM climate_risk_index_nuts3", conn)
    conn.close()
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Carico particelle e join con rischio incendio/precipitazioni...")
    rows, cols = load_rows()
    print(f"  particelle totali nel DB: {len(rows):,}")

    df, dropped, comune_corretto = build_particle_df(rows, cols)
    n_dropped = sum(dropped.values())
    print(f"\nParticelle escluse perche' la provincia catastale vera e' fuori dalle "
          f"4 province target (boundary leakage): {n_dropped}")
    for (cp, tp), n in dropped.most_common():
        print(f"    raccolta come {cp} -> vera provincia {tp}: {n}")
    print(f"Comune corretto (provincia raccolta = provincia vera, ma comune diverso): "
          f"{comune_corretto} occorrenze (particella o riga di valutazione)")

    n_parcels = df["fid"].nunique()
    n_rows = len(df)
    n_with_val = df.loc[df["property_type"].notna(), "fid"].nunique()
    print(f"\nParticelle uniche nel dataset pulito: {n_parcels:,}")
    print(f"Righe (particella x voce di valutazione): {n_rows:,}")
    print(f"Particelle con almeno una quotazione OMI: {n_with_val:,} "
          f"({100*n_with_val/n_parcels:.1f}%)")

    df_cri = load_cri()
    df = df.merge(df_cri, on="provincia", how="left")

    particelle_path = os.path.join(OUT_DIR, "particelle.csv")
    df.to_csv(particelle_path, index=False, encoding="utf-8")
    print(f"\nSalvato dataset a livello di particella: {particelle_path} "
          f"({len(df):,} righe)")

    # --- Aggregazione all'unita' statistica corretta ---------------------
    df_val = df[df["property_type"].notna()].copy()

    agg_spec = {
        "price_m2": "mean",
        "price_m2_prev": "mean",
        "rental_m2": "mean",
        "area_m2": "mean",
        "lat": "mean", "lng": "mean",
        "dist_cbd_km": "mean",
        "seismic_zone": "mean",
        "pga": "mean",
        "flood_ord": "mean",
        "flood_any": "mean",
        "landslide_ord": "mean",
        "landslide_any": "mean",
        "landslide_aa": "mean",
        "subsidence_velocity": "mean",
        "subsidence_risk_class": "mean",
        "fire_score": "mean",
        "fire_is_direct": "mean",
        "rr20mm": "mean",
        "r99pday": "mean",
        "precip_dist_km": "mean",
        "p99_dist_km": "mean",
        "coastal_dist_m": "mean",
        "coastal_change_m_year": "mean",
        "is_coastal_zone": "mean",
        "is_urban": "mean",
        "cri_2021_2050_med": "first",
        "cri_2071_2100_med": "first",
        "aci": "first",
        "zone_description": "first",
        "microzona": "first",
    }
    group_cols = ["provincia", "comune", "zone", "property_type", "condition", "fascia"]

    cell = df_val.groupby(group_cols, dropna=False).agg(agg_spec)
    cell["n_particelle"] = df_val.groupby(group_cols, dropna=False).size()
    cell["price_m2_std"] = df_val.groupby(group_cols, dropna=False)["price_m2"].std()
    cell = cell.reset_index()

    celle_path = os.path.join(OUT_DIR, "celle.csv")
    cell.to_csv(celle_path, index=False, encoding="utf-8")
    print(f"Salvato dataset aggregato a livello di cella: {celle_path} "
          f"({len(cell):,} celle)")

    print("\n--- Riepilogo copertura per provincia (particelle) ---")
    cov = df.groupby("provincia").agg(
        particelle=("fid", "nunique"),
        con_omi=("fid", lambda s: df.loc[s.index][df.loc[s.index, "property_type"].notna()]["fid"].nunique()),
    )
    print(cov)

    print("\n--- Celle per provincia ---")
    print(cell.groupby("provincia").size())

    print("\n--- Distribuzione std prezzo intra-cella (atteso ~0) ---")
    nz = (cell["price_m2_std"].fillna(0) > 1e-6).sum()
    print(f"Celle con std prezzo > 0: {nz} / {len(cell)} "
          f"({100*nz/len(cell):.1f}%) — cfr. nota metodologica nel report "
          f"su compresenza di piu' zone/descrizioni OMI nello stesso raggruppamento.")

    print("\n[OK] Tranche 1 - dataset pronto.")


if __name__ == "__main__":
    main()
