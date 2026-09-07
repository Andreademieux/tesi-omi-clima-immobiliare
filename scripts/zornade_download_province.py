#!/usr/bin/env python3
"""
Download ESAUSTIVO delle particelle catastali arricchite di Zornade
per le province di Milano, Napoli, Torino, Roma.

Rate limit account: 10.000 richieste/ora.
Lo script è RESUMABLE: interrompilo con Ctrl+C e rilancialo, riprende
da dove era rimasto. Tutto viene salvato in un database SQLite
(zornade_data.db) nella cartella corrente.

Fasi:
  collect        -> elenca tutti i fid delle particelle di ogni comune
  details        -> scarica il profilo arricchito completo di ogni fid
  status         -> mostra lo stato di avanzamento + un esempio (no API key)
  export         -> esporta i dati scaricati finora in CSV per l'analisi
  reset-collect  -> rimette a 0 collect_done sui comuni (ripete la raccolta fid)
  reset-details  -> cancella tutti i detail_json scaricati (ripete i dettagli)
  reset-errors   -> cancella SOLO i detail_json falliti (download_failed)
  reset-all      -> ricrea il database da zero (CANCELLA TUTTO)
  both           -> collect poi details (default)

Uso:
    $env:ZORNADE_API_KEY=zrn_e7eb1bf4a8517fd040ae54f666e5e8e97dac9affc8028c7b3e529125c3ce1e7e
    python zornade_full_download.py --phase collect
    python zornade_full_download.py --phase details
    python zornade_full_download.py --phase status
    python zornade_full_download.py --phase export
"""

import os
import sys
import json
import time
import sqlite3
import argparse
from datetime import datetime, timezone, timedelta

import requests

API_BASE = "https://wupqwfqjfpwrapgnogjv.supabase.co/functions/v1/api-v2/api/v2"
DB_PATH = "zornade_data.db"
TARGET_PROVINCES = ["Milano", "Napoli", "Torino", "Roma"]

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
        attempt = 0
        while True:
            attempt += 1
            elapsed = time.time() - self.last_request_time
            if elapsed < MIN_SECONDS_BETWEEN_REQUESTS:
                time.sleep(MIN_SECONDS_BETWEEN_REQUESTS - elapsed)

            try:
                resp = requests.get(url, headers=self.headers, params=params, timeout=30)
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                print("  Connessione internet assente, riprovo tra 30s...")
                time.sleep(30)
                continue

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
            if attempt >= retries:
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

            offset += len(rows)
            if len(rows) == 0:
                break

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
        now_iso = datetime.now(timezone.utc).isoformat()
        if data is None:
            cur.execute(
                "UPDATE parcels SET detail_json=?, detail_downloaded_at=? WHERE fid=?",
                (json.dumps({"error": "download_failed"}), now_iso, fid),
            )
        else:
            cur.execute(
                "UPDATE parcels SET detail_json=?, detail_downloaded_at=? WHERE fid=?",
                (json.dumps(data), now_iso, fid),
            )
        conn.commit()

        done += 1
        if done % 100 == 0:
            remaining = total_todo - done
            print(f"  {done}/{total_todo} scaricate (rimangono {remaining})...")

    print("Fase details completata.")


def phase_status(conn: sqlite3.Connection):
    cur = conn.cursor()

    n_comuni = cur.execute("SELECT COUNT(*) FROM comuni").fetchone()[0]
    n_comuni_done = cur.execute("SELECT COUNT(*) FROM comuni WHERE collect_done=1").fetchone()[0]
    n_parcels = cur.execute("SELECT COUNT(*) FROM parcels").fetchone()[0]
    n_details = cur.execute("SELECT COUNT(*) FROM parcels WHERE detail_json IS NOT NULL").fetchone()[0]
    n_errors = cur.execute(
        "SELECT COUNT(*) FROM parcels WHERE detail_json LIKE '%download_failed%'"
    ).fetchone()[0]

    print("=" * 60)
    print("STATO AVANZAMENTO")
    print("=" * 60)
    print(f"Comuni: {n_comuni_done}/{n_comuni} completati (fase collect)")
    print(f"Particelle totali raccolte: {n_parcels}")
    print(f"Particelle con dettaglio scaricato: {n_details}/{n_parcels}")
    print(f"  di cui in errore: {n_errors}")

    print("\n--- Per provincia ---")
    for row in cur.execute(
        """SELECT province, COUNT(*),
                  SUM(CASE WHEN detail_json IS NOT NULL THEN 1 ELSE 0 END)
           FROM parcels GROUP BY province"""
    ):
        province, tot, done = row
        print(f"  {province}: {done or 0}/{tot} particelle con dettaglio")

    one_hour_ago = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    recent = cur.execute(
        "SELECT COUNT(*) FROM parcels WHERE detail_downloaded_at > ?", (one_hour_ago,)
    ).fetchone()[0]
    if recent > 0:
        remaining = n_parcels - n_details
        eta_hours = remaining / recent
        print(f"\nVelocità osservata ultima ora: {recent} particelle/ora")
        print(f"Stima tempo rimanente ai ritmi attuali: ~{eta_hours:.1f} ore (~{eta_hours/24:.1f} giorni)")

    print("\n--- Esempio di dettaglio scaricato (JSON) ---")
    row = cur.execute(
        "SELECT fid, comune, detail_json FROM parcels "
        "WHERE detail_json IS NOT NULL AND detail_json NOT LIKE '%download_failed%' "
        "LIMIT 1"
    ).fetchone()
    if row:
        fid, comune, detail_raw = row
        print(f"Particella fid={fid} ({comune}):")
        print(json.dumps(json.loads(detail_raw), indent=2, ensure_ascii=False))
    else:
        print("Nessun dettaglio ancora scaricato con successo.")
    print("=" * 60)


def phase_export(conn: sqlite3.Connection):
    import csv
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT fid, comune, province, detail_json FROM parcels WHERE detail_json IS NOT NULL"
    ).fetchall()

    out_path = "zornade_export_completo.csv"
    n_written = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = None
        for fid, comune, province, detail_raw in rows:
            try:
                parsed = json.loads(detail_raw)
                if "error" in parsed:
                    continue
                detail = parsed.get("data", {})
            except (json.JSONDecodeError, AttributeError):
                continue

            risk = detail.get("risk") or {}
            subsidence = detail.get("subsidence") or {}
            valuation_list = detail.get("valuation") or []
            coastal = detail.get("coastal_erosion") or []

            residential = next(
                (v for v in valuation_list if v.get("property_type") == "Abitazioni civili"),
                None,
            )

            all_purchase_mins = [v["purchase"]["min_eur_m2"] for v in valuation_list if v.get("purchase")]
            all_purchase_maxs = [v["purchase"]["max_eur_m2"] for v in valuation_list if v.get("purchase")]

            flat = {
                "fid": fid,
                "comune": comune,
                "province": province,
                "area_m2": detail.get("area_m2"),
                "seismic_zone": risk.get("seismic_zone"),
                "pga": risk.get("pga"),
                "flood_level": risk.get("flood_level"),
                "landslide_level": risk.get("landslide_level"),
                "subsidence_velocity_mm_year": subsidence.get("velocity_mm_year"),
                "subsidence_risk_class": subsidence.get("risk_class"),
                "subsidence_risk_label": subsidence.get("risk_label"),
                "subsidence_risk_index": subsidence.get("risk_index"),
                "omi_zone": residential.get("zone") if residential else None,
                "omi_zone_description": residential.get("zone_description") if residential else None,
                "residential_purchase_min_eur_m2": residential["purchase"]["min_eur_m2"] if residential else None,
                "residential_purchase_max_eur_m2": residential["purchase"]["max_eur_m2"] if residential else None,
                "residential_rental_min_eur_m2": residential["rental"]["min_eur_m2"] if residential else None,
                "residential_rental_max_eur_m2": residential["rental"]["max_eur_m2"] if residential else None,
                "avg_purchase_min_all_types": sum(all_purchase_mins) / len(all_purchase_mins) if all_purchase_mins else None,
                "avg_purchase_max_all_types": sum(all_purchase_maxs) / len(all_purchase_maxs) if all_purchase_maxs else None,
                "coastal_erosion_present": len(coastal) > 0,
                "n_property_types_in_zone": len(valuation_list),
            }

            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
                writer.writeheader()
            writer.writerow(flat)
            n_written += 1

    print(f"Esportato: {out_path} ({n_written} particelle valide su {len(rows)} con dettaglio)")


def phase_reset_collect(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("UPDATE comuni SET collect_done=0")
    conn.commit()
    n = cur.execute("SELECT COUNT(*) FROM comuni").fetchone()[0]
    print(f"Reset completato: {n} comuni rimessi a collect_done=0.")
    print("Le particelle già trovate NON vengono cancellate (INSERT OR IGNORE eviterà duplicati).")
    print("Rilancia --phase collect per recuperare eventuali pagine mancanti.")


def phase_reset_details(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("UPDATE parcels SET detail_json=NULL, detail_downloaded_at=NULL")
    conn.commit()
    n = cur.execute("SELECT COUNT(*) FROM parcels").fetchone()[0]
    print(f"Reset completato: {n} particelle rimesse senza dettaglio.")
    print("Rilancia --phase details per riscaricare tutto da capo.")


def phase_reset_errors(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute(
        "UPDATE parcels SET detail_json=NULL, detail_downloaded_at=NULL "
        "WHERE detail_json LIKE '%download_failed%'"
    )
    n = cur.rowcount
    conn.commit()
    print(f"Reset completato: {n} particelle in errore rimesse senza dettaglio.")
    print("Rilancia --phase details per ritentare solo quelle fallite.")


def phase_reset_all():
    if os.path.exists(DB_PATH):
        confirm = input(
            f"ATTENZIONE: questo cancella DEFINITIVAMENTE '{DB_PATH}' e tutti i dati scaricati.\n"
            f"Scrivi 'CONFERMA' per procedere: "
        )
        if confirm.strip() != "CONFERMA":
            print("Annullato. Nessuna modifica effettuata.")
            return
        os.remove(DB_PATH)
        print(f"Database '{DB_PATH}' cancellato. Verrà ricreato vuoto al prossimo run.")
    else:
        print("Nessun database esistente da cancellare.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=[
            "collect", "details", "status", "export",
            "reset-collect", "reset-details", "reset-errors", "reset-all",
            "both",
        ],
        default="both",
    )
    args = parser.parse_args()

    # reset-all non tocca nemmeno la connessione, gestisce il file direttamente
    if args.phase == "reset-all":
        phase_reset_all()
        return

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    # Queste fasi non richiedono la API key: lavorano solo sul database locale
    if args.phase == "status":
        phase_status(conn)
        conn.close()
        return
    if args.phase == "export":
        phase_export(conn)
        conn.close()
        return
    if args.phase == "reset-collect":
        phase_reset_collect(conn)
        conn.close()
        return
    if args.phase == "reset-details":
        phase_reset_details(conn)
        conn.close()
        return
    if args.phase == "reset-errors":
        phase_reset_errors(conn)
        conn.close()
        return

    api_key = get_api_key()
    client = RateLimitedClient(api_key)

    if args.phase in ("collect", "both"):
        phase_collect(conn, client)
    if args.phase in ("details", "both"):
        phase_details(conn, client)

    conn.close()


if __name__ == "__main__":
    main()