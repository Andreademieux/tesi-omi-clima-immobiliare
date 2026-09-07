#!/usr/bin/env python3
"""
Integra nel database Zornade l'indice di rischio climatico nazionale
italiano (CRI) e l'indice di capacita' adattiva (ACI), a livello di
provincia (NUTS3), dalla fonte:

  Mysiak J, Torresan S, Bosello F, Mistry M, Amadio M, Marzi S, Furlan E,
  Sperotto A. "Climate risk index for Italy." Philosophical Transactions
  of the Royal Society A. 2018;376(2121):20170305.
  DOI: 10.1098/rsta.2017.0305
  Materiale supplementare: fonte_mysiak2018_supp1_metodologia.docx,
  fonte_mysiak2018_supp2_dati_province.xlsx

Questa fonte copre TUTTE e 4 le province campionate (Torino, Milano, Roma,
Napoli) senza dati mancanti e puo' quindi essere integrata direttamente nei
modelli. (Anche il rischio incendio, dalla versione EFFIS/JRC integrata con
integra_rischio_incendio.py, copre ora tutte le province.)

Nota metodologica: il CRI e' un indice PROIETTATO al futuro (orizzonti
2021-2050 e 2071-2100 rispetto a un periodo di riferimento), non una
misura del rischio attuale; MED/MIN/MAX sono la mediana e il range
attraverso l'ensemble di modelli climatici usato dagli autori (si veda
il docx per la metodologia completa: hazard x esposizione x sensibilita',
aggregato con SAW e OWA, poi combinato con la capacita' adattiva per
ottenere il CRI finale).

Uso:
    python integra_climate_risk_index.py
"""
import sqlite3
import csv

DB_PATH = "zornade_data.db"
CSV_PATH = "climate_risk_index_italia_province.csv"


def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.executescript("""
        DROP TABLE IF EXISTS climate_risk_index_nuts3;
        CREATE TABLE climate_risk_index_nuts3 (
            istat_code INTEGER,
            provincia_sigla TEXT,
            provincia_nome TEXT,
            nuts3_code TEXT,
            cri_2021_2050_max REAL,
            cri_2021_2050_min REAL,
            cri_2021_2050_med REAL,
            cri_2071_2100_max REAL,
            cri_2071_2100_min REAL,
            cri_2071_2100_med REAL,
            cri_owa_2021_2050_max REAL,
            cri_owa_2021_2050_min REAL,
            cri_owa_2021_2050_med REAL,
            cri_owa_2071_2100_max REAL,
            cri_owa_2071_2100_min REAL,
            cri_owa_2071_2100_med REAL,
            aci REAL,
            PRIMARY KEY (provincia_nome)
        );
    """)

    n_rows = 0
    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = ["istat_code", "provincia_sigla", "provincia_nome", "nuts3_code",
                "cri_2021_2050_max", "cri_2021_2050_min", "cri_2021_2050_med",
                "cri_2071_2100_max", "cri_2071_2100_min", "cri_2071_2100_med",
                "cri_owa_2021_2050_max", "cri_owa_2021_2050_min", "cri_owa_2021_2050_med",
                "cri_owa_2071_2100_max", "cri_owa_2071_2100_min", "cri_owa_2071_2100_med",
                "aci"]
        placeholders = ", ".join(["?"] * len(cols))
        for row in reader:
            cur.execute(
                f"INSERT OR IGNORE INTO climate_risk_index_nuts3 ({', '.join(cols)}) VALUES ({placeholders})",
                [row[c] for c in cols],
            )
            n_rows += 1
    conn.commit()
    print(f"Inserite {n_rows} province in climate_risk_index_nuts3.")

    print("\n--- Le 4 province target ---")
    for r in cur.execute("""
        SELECT provincia_nome, cri_2021_2050_med, cri_2071_2100_med, aci
        FROM climate_risk_index_nuts3
        WHERE provincia_nome IN ('Torino','Milano','Roma','Napoli')
        ORDER BY cri_2021_2050_med DESC
    """):
        print(f"  {r[0]}: CRI 2021-2050 (med)={r[1]:.3f}  CRI 2071-2100 (med)={r[2]:.3f}  ACI={r[3]:.3f}")

    conn.close()


if __name__ == "__main__":
    main()
