#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integra cdd (siccita', headline) e su95p (calore, headline) nel dataset
OMI a livello di cella, replicando la stessa aggregazione (media) usata da
prepara_dataset_analisi.py per gli altri rischi, poi ristima A0/A1/A2 con
questi due rischi aggiunti alla famiglia headline (S4.2, "eventi meteo
estremi"), sulla stessa unita' statistica (cella) e con lo stesso standard
di reporting (S8) del modello primario.

Non sovrascrive gli output della Tranche 2 primaria (m1-m6): scrive un
appendice separato (output/modelli/omi_calore_*).

Uso:
    python integra_calore_in_omi.py
"""
import io
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "zornade_data.db")
OUT = os.path.join(BASE, "output")
MOD = os.path.join(OUT, "modelli")

RISKS_BASE = ["flood_ord", "landslide_ord", "pga", "sub_sink", "fire_score",
              "r99pday", "is_coastal_zone"]
RISKS = RISKS_BASE + ["cdd", "su95p"]
# Correzione di Holm per famiglia (standard ufficiale, non su tutti i 9
# insieme): il rischio fisico (danno diretto) e il rischio di comfort/costo
# energetico sono due ipotesi di ricerca distinte, cfr. 05_modelli_livello1.py
# nell'estensione immobiliare.it.
FAMIGLIE_RISCHIO = {"fisico": RISKS_BASE, "comfort": ["cdd", "su95p"]}
CONTROLS_A0 = "C(property_type) + C(condition)"
CONTROLS_LOC = "C(fascia) + dist_cbd_km + is_urban"


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except ValueError:
                pass


def sec(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def fit_ols(formula, df):
    return smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["comune"]})


def coef_table(res, df, spec_name):
    rows = []
    ci = res.conf_int()
    holm_by_var = {}
    for fam_vars in FAMIGLIE_RISCHIO.values():
        pvals = [res.pvalues[v] for v in fam_vars]
        holm = multipletests(pvals, method="holm")[1]
        holm_by_var.update(zip(fam_vars, holm))
    for v in RISKS:
        b = res.params[v]
        sd = df[v].std()
        fam = next(f for f, vs in FAMIGLIE_RISCHIO.items() if v in vs)
        rows.append({
            "spec": spec_name, "rischio": v, "famiglia": fam, "beta": b,
            "ci95_low": ci.loc[v, 0], "ci95_high": ci.loc[v, 1],
            "p_grezzo": res.pvalues[v], "p_holm": holm_by_var[v],
            "pct_per_1sd": 100 * (np.exp(b * sd) - 1),
        })
    return pd.DataFrame(rows)


def main():
    log = open(os.path.join(MOD, "omi_calore_summary.txt"), "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log)

    sec("0. INTEGRAZIONE cdd/su95p A LIVELLO DI PARTICELLA -> CELLA")
    particelle = pd.read_csv(os.path.join(OUT, "particelle.csv"), low_memory=False)
    conn = sqlite3.connect(DB_PATH)
    cdd = pd.read_sql("SELECT fid, cdd FROM parcels_rischio_siccita_cdd", conn)
    su95p = pd.read_sql("SELECT fid, su95p FROM parcels_rischio_calore_su95p", conn)
    conn.close()
    particelle = particelle.merge(cdd, on="fid", how="left").merge(su95p, on="fid", how="left")
    print(f"Particelle: {len(particelle):,}  con cdd: {particelle['cdd'].notna().sum():,}  "
          f"con su95p: {particelle['su95p'].notna().sum():,}")

    group_cols = ["provincia", "comune", "zone", "property_type", "condition", "fascia"]
    df_val = particelle[particelle["property_type"].notna()].copy()
    cell_calore = df_val.groupby(group_cols, dropna=False).agg(
        cdd=("cdd", "mean"), su95p=("su95p", "mean")).reset_index()
    print(f"Celle con cdd/su95p aggregato: {len(cell_calore):,}")

    cell = pd.read_csv(os.path.join(OUT, "celle.csv"))
    print(f"Celle originali (celle.csv): {len(cell):,}")
    cell = cell.merge(cell_calore, on=group_cols, how="left")
    print(f"Celle dopo merge, con cdd non-mancante: {cell['cdd'].notna().sum():,} / {len(cell):,}")
    cell.to_csv(os.path.join(OUT, "celle_con_calore.csv"), index=False)

    # ------------------------------------------------------------------
    sec("1. CAMPIONE E SPECIFICHE (stessa logica A0/A1/A2 del modello primario)")
    cell["sub_sink"] = -cell["subsidence_velocity"]
    cell["log_price"] = np.log(cell["price_m2"])
    model_vars = (["log_price", "price_m2", "comune", "provincia", "zone",
                   "property_type", "condition", "fascia", "dist_cbd_km",
                   "is_urban", "n_particelle"] + RISKS)
    df = cell.dropna(subset=[v for v in model_vars if v != "zone"]).copy()
    print(f"Celle totali: {len(cell)}  campione di stima (listwise): {len(df)} "
          f"({100*len(df)/len(cell):.1f}%)")
    print(f"Missing aggiuntivo per cdd/su95p: {cell['cdd'].isna().sum()} celle "
          f"(zone OMI senza alcuna particella con dato climatico attribuito)")

    risk_str = " + ".join(RISKS)
    f_a0 = f"log_price ~ {risk_str} + {CONTROLS_A0}"
    f_a1 = f"log_price ~ {risk_str} + {CONTROLS_A0} + {CONTROLS_LOC} + C(provincia)"
    f_a2 = f"log_price ~ {risk_str} + {CONTROLS_A0} + {CONTROLS_LOC} + C(comune)"

    res_a0 = fit_ols(f_a0, df)
    res_a1 = fit_ols(f_a1, df)
    res_a2 = fit_ols(f_a2, df)

    tabs = [coef_table(r, df, nm) for r, nm in [(res_a0, "A0"), (res_a1, "A1"), (res_a2, "A2")]]
    coefs = pd.concat(tabs, ignore_index=True)
    coefs.round(5).to_csv(os.path.join(MOD, "omi_calore_coefficienti.csv"), index=False)

    sec("2. COEFFICIENTI cdd/su95p (traiettoria A0->A1->A2)")
    for nm in ["A0", "A1", "A2"]:
        t = coefs[(coefs["spec"] == nm) & (coefs["rischio"].isin(["cdd", "su95p"]))]
        print(f"\n--- {nm} ---")
        print(t[["rischio", "beta", "ci95_low", "ci95_high", "p_grezzo", "p_holm",
                 "pct_per_1sd"]].round(4).to_string(index=False))

    print(f"\nR2: A0={res_a0.rsquared:.3f} -> A1={res_a1.rsquared:.3f} -> A2={res_a2.rsquared:.3f}")
    print(f"N={int(res_a2.nobs)}, comuni={df['comune'].nunique()}")

    sec("3. QUOTA DI VARIANZA WITHIN-COMUNE (S16.1) — cdd, su95p vs gli altri")
    within_share = {}
    for v in RISKS:
        tot = df[v].var()
        within = df.groupby("comune")[v].transform(lambda s: s - s.mean()).var()
        within_share[v] = within / tot if tot > 0 else np.nan
    ws = pd.Series(within_share)
    print((100 * ws).round(1).astype(str) + "%")
    ws.to_csv(os.path.join(MOD, "omi_calore_within_variance.csv"))

    sec("4. VIF (cdd, su95p vs famiglia esistente)")
    Xv = sm.add_constant(df[RISKS + ["dist_cbd_km", "is_urban"]].copy())
    vif = pd.Series([variance_inflation_factor(Xv.values, i) for i in range(1, Xv.shape[1])],
                     index=Xv.columns[1:], name="VIF_raw")
    print(vif.round(2))

    print("\n[OK] Integrazione calore/siccita' in OMI completata.")
    sys.stdout = sys.__stdout__
    log.close()


if __name__ == "__main__":
    main()
