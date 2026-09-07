"""
Modello multiplo: impatto del rischio ambientale sulle valutazioni immobiliari.
Fonte dati: zornade_data.db (provincia di Napoli, unica con detail_json popolato).

Variabile dipendente: prezzo medio OMI (EUR/m2) per "Abitazioni civili",
media tra min e max della fascia di valutazione della zona OMI di appartenenza.

Regressori:
  - seismic_zone      (1=piu' a rischio ... 4=meno a rischio, scala INGV)
  - pga               (peak ground acceleration, continuo)
  - subsidence_risk_class   (1..5, classe di rischio subsidenza)
  - subsidence_velocity_mm_year (velocita' di abbassamento/sollevamento suolo)
  - area_m2           (controllo dimensione particella)
  - comune            (fixed effect, per isolare il rischio dalla localizzazione)
"""
import sqlite3
import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

DB_PATH = "zornade_data.db"
PROPERTY_TYPE = "Abitazioni civili"


def load_records():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT fid, comune, area_m2, detail_json FROM parcels "
        "WHERE province='Napoli' AND detail_json IS NOT NULL"
    )
    rows = cur.fetchall()
    conn.close()

    records = []
    for fid, comune, area_m2, dj in rows:
        d = json.loads(dj)["data"]
        risk = d.get("risk") or {}
        sub = d.get("subsidence") or {}
        val = d.get("valuation") or []

        prices = [
            (v["purchase"]["min_eur_m2"] + v["purchase"]["max_eur_m2"]) / 2
            for v in val
            if v.get("property_type") == PROPERTY_TYPE and v.get("purchase")
        ]
        if not prices:
            continue
        price_m2 = float(np.mean(prices))

        seismic_zone = risk.get("seismic_zone")
        pga = risk.get("pga")
        sub_class = sub.get("risk_class")
        sub_vel = sub.get("velocity_mm_year")

        if None in (seismic_zone, pga, sub_class, sub_vel, area_m2):
            continue

        records.append({
            "fid": fid,
            "comune": comune,
            "price_m2": price_m2,
            "seismic_zone": seismic_zone,
            "pga": pga,
            "subsidence_risk_class": sub_class,
            "subsidence_velocity": sub_vel,
            "area_m2": area_m2,
        })

    return pd.DataFrame.from_records(records)


def main():
    df = load_records()
    print(f"Osservazioni utilizzabili: {len(df)}")
    print(f"Comuni distinti: {df['comune'].nunique()}")
    print()
    print(df[["price_m2", "seismic_zone", "pga", "subsidence_risk_class",
              "subsidence_velocity", "area_m2"]].describe())
    print()

    # log-price per interpretare i coefficienti come effetti percentuali
    df["log_price_m2"] = np.log(df["price_m2"])
    df["log_area_m2"] = np.log(df["area_m2"])

    # --- Modello 1: pooled OLS, nessun controllo di localizzazione ---
    m1 = smf.ols(
        "log_price_m2 ~ seismic_zone + pga + subsidence_risk_class + "
        "subsidence_velocity + log_area_m2",
        data=df,
    ).fit(cov_type="HC1")

    # --- Modello 2: con fixed effect di comune (isola il rischio dalla localita') ---
    m2 = smf.ols(
        "log_price_m2 ~ seismic_zone + pga + subsidence_risk_class + "
        "subsidence_velocity + log_area_m2 + C(comune)",
        data=df,
    ).fit(cov_type="cluster", cov_kwds={"groups": df["comune"]})

    print("=" * 70)
    print("MODELLO 1 - Pooled OLS (nessun controllo di localizzazione)")
    print("=" * 70)
    print(m1.summary())

    print()
    print("=" * 70)
    print("MODELLO 2 - OLS con Fixed Effects per comune (robust clustered SE)")
    print("=" * 70)
    # Stampo solo i coefficienti di interesse, non le ~90 dummy di comune
    coef_table = m2.summary2().tables[1]
    vars_of_interest = ["Intercept", "seismic_zone", "pga",
                         "subsidence_risk_class", "subsidence_velocity",
                         "log_area_m2"]
    print(coef_table.loc[vars_of_interest])
    print(f"\nR-squared: {m2.rsquared:.4f}   R-squared adj: {m2.rsquared_adj:.4f}")
    print(f"N osservazioni: {int(m2.nobs)}   N comuni (FE): {df['comune'].nunique()}")

    df.to_csv("napoli_risk_valuation_dataset.csv", index=False)
    print("\nDataset esportato in napoli_risk_valuation_dataset.csv")


if __name__ == "__main__":
    main()
