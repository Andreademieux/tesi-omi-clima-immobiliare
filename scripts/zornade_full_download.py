#!/usr/bin/env python3
"""
Download ESAUSTIVO delle particelle catastali arricchite di Zornade
per le province di Milano, Napoli, Torino, Roma.

Rate limit account: 10.000 richieste/ora.
Lo script è RESUMABLE: interrompilo con Ctrl+C e rilancialo, riprende
da dove era rimasto. Tutto viene salvato in un database SQLite
(zornade_data.db) nella cartella corrente.

Fasi:
  collect  -> elenca tutti i fid delle particelle di ogni comune
  details  -> scarica il profilo arricchito completo di ogni fid
  export   -> esporta i dati scaricati finora in CSV per l'analisi
  both     -> collect poi details (default)

Uso:
    $env:ZORNADE_API_KEY=zrn_d0ef8a8c93795d15a883ff8b1660561dafb936f111e07bbbb271070a5f272d6f
    python zornade_full_download.py --phase collect
    python zornade_full_download.py --phase details
    python zornade_full_download.py --phase export
"""

import os
import sys
import json
import time
import sqlite3
import argparse
from datetime import datetime

import requests

API_BASE = "https://wupqwfqjfpwrapgnogjv.supabase.co/functions/v1/api-v2/api/v2"
DB_PATH = "zornade_data.db"
TARGET_PROVINCES = ["Milano", "Napoli", "Torino", "Roma"]

# Margine di sicurezza sotto le 10.000 richieste/ora dichiarate dall'account
MIN_SECONDS_BETWEEN_REQUESTS = 0.4   # ~9000 richieste/ora
SEARCH_PAGE_SIZE = 200
DETAIL_SECTIONS = "basic,cadastral,address,risk,subsidence,land_cover,land_use,valuation,coastal_erosion,cultural_heritage,poi"


def get_api_key() -> str:
    key = os.environ.get("ZORNADE_API_KEY")
    if not key:
        sys.exit("ERRORE: variabile d'ambiente ZORNADE_API_KEY non impostata.")
    return key


class RateLimitedClient:
    def __init__(self, api_key: str):
        self.headers = {"x-api-key": api_key}
        self.last_request_time = 0.0

    def get(self, path: str, params: dict | None = None, retries: int = 5):
        url = f"{API_BASE}{path}"
        for attempt in range(1, retries + 1):
            elapsed = time.time() - self.last_request_time
            if elapsed < MIN_SECONDS_BETWEEN_REQUESTS:
                time.sleep(MIN_SECONDS_BETWEEN_REQUESTS - elapsed)

            resp = requests.get(url, headers=self.headers, params=params, timeout=30)
            self.last_request_time = time.time()

            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError:
                    print(f"  Risposta 200 non-JSON: {resp.text[:200]}")
                    return None

            if resp.status_code == 429:
                wait = int(resp.headers.get("x-ratelimit-reset", 60))
                print(f"  Rate limit (429), attendo {wait}s...")
                time.sleep(max(wait, 10))
                continue

            if resp.status_code == 401:
                sys.exit("ERRORE 401: API key non valida.")

            print(f"  Tentativo {attempt}/{retries} fallito ({resp.status_code}): {resp.text[:200]}")
            time.sleep(5)

        print(f"  ERRORE: richiesta fallita definitivamente: {url} {params}")
        return None


def init_db(conn: sqlite3.Connection):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS comuni (
        name TEXT,
        province TEXT,
        collect_done INTEGER DEFAULT 0,
        parcels_found INTEGER DEFAULT 0,
        PRIMARY KEY (name, province)
    );
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
    );
    """)
    conn.commit()


def fetch_municipalities(client: RateLimitedClient, province: str) -> list[dict]:
    data = client.get("/admin/municipalities", params={"province": province})
    return data.get("data", []) if data else []


def phase_collect(conn: sqlite3.Connection, client: RateLimitedClient):
    cur = conn.cursor()

    for province in TARGET_PROVINCES:
        row = cur.execute(
            "SELECT COUNT(*) FROM comuni WHERE province=?", (province,)
        ).fetchone()
        if row[0] == 0:
            print(f"Recupero elenco comuni di {province}...")
            comuni = fetch_municipalities(client, province)
            for c in comuni:
                name = c.get("name") or c.get("municipality_name")
                cur.execute(
                    "INSERT OR IGNORE INTO comuni (name, province) VALUES (?, ?)",
                    (name, province),
                )
            conn.commit()
            print(f"  -> {len(comuni)} comuni trovati per {province}.")

    todo = cur.execute(
        "SELECT name, province FROM comuni WHERE collect_done=0"
    ).fetchall()
    print(f"\nComuni da processare (fase collect): {len(todo)}")

    for i, (comune, province) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] Raccolgo particelle: {comune} ({province})")
        offset = 0
        total_found = 0
        while True:
            data = client.get(
                "/parcels/search",
                params={"comune": comune, "limit": SEARCH_PAGE_SIZE, "offset": offset},
            )
            if data is None:
                print(f"  ERRORE su {comune}, salto (riprova al prossimo run).")
                break

            rows = data.get("data", [])
            if not rows:
                break

            for r in rows:
                fid = r.get("fid")
                centroid = r.get("centroid") or {}
                lat = centroid.get("lat") or centroid.get("latitude")
                lng = centroid.get("lng") or centroid.get("longitude")
                cur.execute(
                    """INSERT OR IGNORE INTO parcels
                       (fid, comune, province, label, area_m2, lat, lng)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (fid, comune, province, r.get("label"), r.get("area_m2"), lat, lng),
                )
            conn.commit()
            total_found += len(rows)

            if len(rows) < SEARCH_PAGE_SIZE:
                break
            offset += SEARCH_PAGE_SIZE

        cur.execute(
            "UPDATE comuni SET collect_done=1, parcels_found=? WHERE name=? AND province=?",
            (total_found, comune, province),
        )
        conn.commit()
        print(f"  -> {total_found} particelle trovate e salvate.")

    total = cur.execute("SELECT COUNT(*) FROM parcels").fetchone()[0]
    print(f"\nFase collect completata. Totale particelle in database: {total}")
    est_hours = total / 9000
    print(f"Stima fase 'details' a questo rate limit: ~{est_hours:.2f} ore")


def phase_details(conn: sqlite3.Connection, client: RateLimitedClient):
    cur = conn.cursor()
    total_todo = cur.execute(
        "SELECT COUNT(*) FROM parcels WHERE detail_json IS NULL"
    ).fetchone()[0]
    print(f"Particelle senza dettaglio scaricato: {total_todo}")

    done = 0
    while True:
        row = cur.execute(
            "SELECT fid FROM parcels WHERE detail_json IS NULL LIMIT 1"
        ).fetchone()
        if row is None:
            break
        fid = row[0]

        data = client.get(f"/parcels/{fid}", params={"include": DETAIL_SECTIONS})
        if data is None:
            cur.execute(
                "UPDATE parcels SET detail_json=?, detail_downloaded_at=? WHERE fid=?",
                (json.dumps({"error": "download_failed"}), datetime.utcnow().isoformat(), fid),
            )
        else:
            cur.execute(
                "UPDATE parcels SET detail_json=?, detail_downloaded_at=? WHERE fid=?",
                (json.dumps(data), datetime.utcnow().isoformat(), fid),
            )
        conn.commit()

        done += 1
        if done % 100 == 0:
            remaining = total_todo - done
            print(f"  {done}/{total_todo} scaricate (rimangono {remaining})...")

    print("Fase details completata.")


def phase_export(conn: sqlite3.Connection):
    import csv
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT fid, comune, province, detail_json FROM parcels WHERE detail_json IS NOT NULL"
    ).fetchall()

    out_path = "zornade_export_completo.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = None
        for fid, comune, province, detail_raw in rows:
            try:
                detail = json.loads(detail_raw).get("data", {})
            except (json.JSONDecodeError, AttributeError):
                continue

            flat = {
                "fid": fid,
                "comune": comune,
                "province": province,
                "seismic_risk": (detail.get("risks") or {}).get("seismic_risk"),
                "flood_risk": (detail.get("risks") or {}).get("flood_risk"),
                "landslide_risk": (detail.get("risks") or {}).get("landslide_risk"),
                "subsidence_risk_class": (detail.get("subsidence") or {}).get("risk_class"),
                "subsidence_velocity": (detail.get("subsidence") or {}).get("velocity"),
                "omi_purchase_min": (detail.get("valuation") or {}).get("purchase_min"),
                "omi_purchase_max": (detail.get("valuation") or {}).get("purchase_max"),
                "omi_rental_min": (detail.get("valuation") or {}).get("rental_min"),
                "omi_rental_max": (detail.get("valuation") or {}).get("rental_max"),
                "coastal_erosion": (detail.get("coastal_erosion") or {}).get("risk_level"),
                "area_m2": (detail.get("physical") or {}).get("footprint_sqm"),
            }

            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
                writer.writeheader()
            writer.writerow(flat)

    print(f"Esportato: {out_path} ({len(rows)} particelle)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["collect", "details", "export", "both"], default="both")
    args = parser.parse_args()

    api_key = get_api_key()
    client = RateLimitedClient(api_key)
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    if args.phase in ("collect", "both"):
        phase_collect(conn, client)
    if args.phase in ("details", "both"):
        phase_details(conn, client)
    if args.phase == "export":
        phase_export(conn)

    conn.close()


if __name__ == "__main__":
    main()