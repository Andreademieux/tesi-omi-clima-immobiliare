#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tranche 1 - Analisi esplorativa dei dati (protocollo di studio: S6),
copertura campionaria e selection bias (S3), natura del dato di prezzo (S2),
pesi campionari (S7.2), tabella fonti (S4.2).

Input:  output/particelle.csv (livello particella x voce di valutazione),
        output/celle.csv (unita' statistica primaria).
Output: output/eda/*.csv (tabelle), output/eda/*.png (figure),
        output/eda/eda_summary.txt (log leggibile con tutti i numeri citati
        nel report).

Uso:
    py analisi_esplorativa.py
"""
import os
import sys
import io

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sps
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")
EDA = os.path.join(OUT, "eda")
os.makedirs(EDA, exist_ok=True)

RISK_VARS = ["pga", "seismic_zone", "flood_any", "flood_ord", "landslide_any",
             "landslide_ord", "landslide_aa", "subsidence_velocity",
             "subsidence_risk_class", "fire_score", "r99pday", "rr20mm",
             "is_coastal_zone"]
RISK_LABELS = {
    "pga": "PGA (sisma)", "seismic_zone": "Zona sismica (1=alta,4=bassa)",
    "flood_any": "Alluvione: quota in perimetro PGRA", "flood_ord": "Alluvione: classe ordinale",
    "landslide_any": "Frana: quota in perimetro PAI", "landslide_ord": "Frana: classe ordinale",
    "landslide_aa": "Frana: aree attenzione AA", "subsidence_velocity": "Subsidenza: vel. mm/anno",
    "subsidence_risk_class": "Subsidenza: classe rischio", "fire_score": "Incendio: score EFFIS",
    "r99pday": "Precip. estreme: r99pday (headline)", "rr20mm": "Precip. estreme: rr20mm (robustezza)",
    "is_coastal_zone": "Erosione costiera: zona costiera",
}


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)
    def flush(self):
        for st in self.streams:
            st.flush()


def sec(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    log = open(os.path.join(EDA, "eda_summary.txt"), "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log)

    part = pd.read_csv(os.path.join(OUT, "particelle.csv"), low_memory=False)
    cell = pd.read_csv(os.path.join(OUT, "celle.csv"))

    # ------------------------------------------------------------------ 1
    sec("1. STRUTTURA DEL DATASET")
    n_fid = part["fid"].nunique()
    has_val = part.groupby("fid")["property_type"].apply(lambda s: s.notna().any())
    print(f"Particelle uniche (pulite, 4 province): {n_fid:,}")
    print(f"Righe particella x voce OMI: {len(part):,}")
    print(f"Particelle con >=1 quotazione OMI: {has_val.sum():,} ({100*has_val.mean():.1f}%)")
    print(f"Celle (provincia x comune x zona x tipologia x condizione x fascia): {len(cell):,}")
    print(f"Comuni distinti: {part['comune'].nunique()}   "
          f"Zone OMI distinte (comune,zona): {cell.groupby(['comune','zone']).ngroups}")

    t = cell.groupby("provincia").agg(
        celle=("price_m2", "size"), comuni=("comune", "nunique"),
        particelle_medie_per_cella=("n_particelle", "mean"),
        prezzo_medio=("price_m2", "mean"), prezzo_mediano=("price_m2", "median"))
    print("\nPer provincia (celle):")
    print(t.round(2))
    t.round(3).to_csv(os.path.join(EDA, "t1_struttura_province.csv"))

    t = (cell.pivot_table(index="property_type", columns="provincia",
                          values="price_m2", aggfunc="size", fill_value=0)
         .assign(TOT=lambda d: d.sum(axis=1)).sort_values("TOT", ascending=False))
    print("\nCelle per tipologia x provincia:")
    print(t)
    t.to_csv(os.path.join(EDA, "t2_celle_tipologia_provincia.csv"))

    print("\nCondition (celle):", cell["condition"].value_counts().to_dict())
    print("Fascia (celle):", cell["fascia"].value_counts().to_dict())

    # ------------------------------------------------------------------ 2
    sec("2. NATURA DEL DATO DI PREZZO E DIMENSIONE TEMPORALE (S2)")
    print("Quotazioni OMI: intervalli min-max EUR/m2 per zona x tipologia x stato;")
    print("il prezzo analizzato e' il punto medio dell'intervallo. Vintage: download")
    print("2026-07-09/11 -> semestre corrente piu' recente pubblicato, con campo")
    print("previous_semester come semestre precedente (2 periodi).")
    both = cell.dropna(subset=["price_m2", "price_m2_prev"])
    chg = (both["price_m2"] != both["price_m2_prev"])
    print(f"\nCelle con entrambi i semestri: {len(both):,}")
    print(f"Celle con prezzo variato tra i 2 semestri: {chg.sum():,} ({100*chg.mean():.2f}%)")
    dl = (np.log(both["price_m2"]) - np.log(both["price_m2_prev"]))
    print(f"Delta log prezzo: media={dl.mean():.5f}  sd={dl.std():.5f}  "
          f"p1={dl.quantile(.01):.4f}  p99={dl.quantile(.99):.4f}")
    by_prov = both.assign(chg=chg).groupby("provincia")["chg"].mean()
    print("Quota celle con variazione, per provincia:")
    print((100 * by_prov).round(2).astype(str) + "%")
    has_rent = cell["rental_m2"].notna()
    print(f"\nCelle con canone di locazione disponibile: {has_rent.sum():,} "
          f"({100*has_rent.mean():.1f}%) — utilizzabile come confronto (S2)")

    # ------------------------------------------------------------------ 3
    sec("3. COPERTURA CAMPIONARIA E SELECTION BIAS (S3)")
    pl = part.drop_duplicates("fid").copy()
    pl["has_omi"] = pl["fid"].map(has_val).astype(int)
    cov = pl.groupby("provincia")["has_omi"].agg(["size", "sum", "mean"])
    cov.columns = ["particelle", "con_omi", "quota_con_omi"]
    print("Copertura OMI per provincia (particelle):")
    print(cov.assign(quota_con_omi=lambda d: (100*d["quota_con_omi"]).round(1)))
    cov.to_csv(os.path.join(EDA, "t3_copertura_omi_provincia.csv"))

    print("\nQuota con OMI per classe di urbanizzazione (land cover CLC):")
    print((100 * pl.groupby("is_urban")["has_omi"].mean()).round(1))

    # differenze standardizzate rischio: con vs senza OMI
    rows = []
    for v in RISK_VARS:
        a = pl.loc[pl["has_omi"] == 1, v].dropna()
        b = pl.loc[pl["has_omi"] == 0, v].dropna()
        if len(a) < 30 or len(b) < 30:
            continue
        sd_p = np.sqrt((a.var() + b.var()) / 2)
        smd = (a.mean() - b.mean()) / sd_p if sd_p > 0 else np.nan
        rows.append({"variabile": v, "media_con_omi": a.mean(), "media_senza_omi": b.mean(),
                     "SMD": smd})
    smd_df = pd.DataFrame(rows).set_index("variabile")
    print("\nDifferenze standardizzate (SMD) del rischio, particelle con vs senza OMI")
    print("(|SMD|>0.10 = squilibrio non trascurabile, >0.25 = marcato):")
    print(smd_df.round(3))
    smd_df.round(4).to_csv(os.path.join(EDA, "t4_selection_smd.csv"))

    # logit descrittivo has_omi ~ rischi + FE provincia (SE cluster comune)
    lv = ["pga", "flood_any", "landslide_any", "fire_score", "r99pday",
          "subsidence_velocity", "is_coastal_zone", "is_urban"]
    dfl = pl[["has_omi", "comune", "provincia"] + lv].dropna().copy()
    for v in lv:
        s = dfl[v].std()
        if s > 0:
            dfl[v + "_z"] = (dfl[v] - dfl[v].mean()) / s
    zs = [v + "_z" for v in lv if v + "_z" in dfl]
    logit = smf.logit("has_omi ~ " + " + ".join(zs) + " + C(provincia)", data=dfl)
    try:
        rl = logit.fit(disp=0, cov_type="cluster", cov_kwds={"groups": dfl["comune"]})
        tab = pd.DataFrame({"coef": rl.params, "p": rl.pvalues,
                            "OR": np.exp(rl.params)})
        tab = tab.loc[[i for i in tab.index if i.endswith("_z")]]
        print("\nLogit descrittivo P(has_omi) ~ rischi standardizzati + FE provincia")
        print("(SE cluster comune; OR per +1 sd del rischio):")
        print(tab.round(3))
        tab.round(4).to_csv(os.path.join(EDA, "t5_selection_logit.csv"))
    except Exception as e:
        print("Logit non stimabile:", e)

    # ------------------------------------------------------------------ 4
    sec("4. DISTRIBUZIONE DEI PREZZI (celle)")
    d = cell["price_m2"].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
    print("price_m2 (EUR/m2):")
    print(d.round(1))
    lp = np.log(cell["price_m2"])
    print(f"\nlog(price): skewness={sps.skew(lp.dropna()):.3f}  "
          f"kurtosi eccesso={sps.kurtosis(lp.dropna()):.3f}   "
          f"(livelli: skew={sps.skew(cell['price_m2'].dropna()):.3f})")
    t = cell.groupby("property_type")["price_m2"].describe(percentiles=[.5])[
        ["count", "mean", "50%", "std", "min", "max"]].sort_values("count", ascending=False)
    print("\nPrezzo per tipologia:")
    print(t.round(0))
    t.round(1).to_csv(os.path.join(EDA, "t6_prezzi_tipologia.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for prov, g in cell.groupby("provincia"):
        axes[0].hist(g["price_m2"], bins=60, alpha=0.45, label=prov)
        axes[1].hist(np.log(g["price_m2"]), bins=60, alpha=0.45, label=prov)
    axes[0].set_title("price_m2 (EUR/m2)"); axes[1].set_title("log(price_m2)")
    axes[0].legend(fontsize=8)
    for ax in axes: ax.set_ylabel("celle")
    fig.tight_layout(); fig.savefig(os.path.join(EDA, "f1_distribuzione_prezzi.png"), dpi=150)
    plt.close(fig)

    # outlier univariati
    q1, q99 = cell["price_m2"].quantile([.01, .99])
    n_out = ((cell["price_m2"] < q1) | (cell["price_m2"] > q99)).sum()
    print(f"\nOutlier oltre p1/p99 ({q1:.0f} / {q99:.0f} EUR/m2): {n_out} celle "
          f"({100*n_out/len(cell):.1f}%) — gestiti in robustezza con winsorizzazione (S17)")
    ext = cell.nlargest(5, "price_m2")[["provincia", "comune", "zone", "property_type", "price_m2"]]
    print("Top-5 prezzi:")
    print(ext.to_string(index=False))

    # ------------------------------------------------------------------ 5
    sec("5. VARIABILI DI RISCHIO: DISTRIBUZIONI (celle)")
    t = cell[RISK_VARS].describe(percentiles=[.05, .5, .95]).T[
        ["count", "mean", "std", "min", "5%", "50%", "95%", "max"]]
    t.index = [RISK_LABELS.get(i, i) for i in t.index]
    print(t.round(3))
    t.round(4).to_csv(os.path.join(EDA, "t7_rischi_descrittive.csv"))

    print("\nMedie di rischio per provincia:")
    t = cell.groupby("provincia")[RISK_VARS].mean()
    print(t.round(3).T)
    t.round(4).to_csv(os.path.join(EDA, "t8_rischi_per_provincia.csv"))

    print(f"\nQuote celle esposte: flood_any>0: {100*(cell['flood_any']>0).mean():.1f}%  "
          f"landslide_any>0: {100*(cell['landslide_any']>0).mean():.1f}%  "
          f"coastal: {100*(cell['is_coastal_zone']>0).mean():.1f}%")

    # ------------------------------------------------------------------ 6
    sec("6. CORRELAZIONI TRA RISCHI E CON LA LOCALIZZAZIONE (celle)")
    cv = RISK_VARS + ["dist_cbd_km", "is_urban"]
    sub = cell[cv].dropna(how="all")
    corr_s = sub.corr(method="spearman")
    corr_s.round(3).to_csv(os.path.join(EDA, "t9_correlazioni_spearman.csv"))
    print("Spearman (estratto — coppie con |rho|>0.4, escluse coppie derivate):")
    derived = {("flood_any", "flood_ord"), ("landslide_any", "landslide_ord"),
               ("pga", "seismic_zone"), ("subsidence_velocity", "subsidence_risk_class")}
    seen = []
    for i in corr_s.index:
        for j in corr_s.columns:
            if i < j and abs(corr_s.loc[i, j]) > 0.4 and (i, j) not in derived and (j, i) not in derived:
                seen.append((i, j, corr_s.loc[i, j]))
    for i, j, r in sorted(seen, key=lambda x: -abs(x[2])):
        print(f"  {i:<22} x {j:<22} rho={r:+.3f}")

    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(corr_s.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(cv))); ax.set_xticklabels(cv, rotation=90, fontsize=7)
    ax.set_yticks(range(len(cv))); ax.set_yticklabels(cv, fontsize=7)
    fig.colorbar(im, shrink=0.8); ax.set_title("Correlazioni Spearman (celle)")
    fig.tight_layout(); fig.savefig(os.path.join(EDA, "f2_corr_rischi.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------ 7
    sec("7. DATI MANCANTI (celle) E TRATTAMENTO (S6)")
    miss = cell.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    print("Quota di missing per variabile (solo variabili con missing >0):")
    print((100 * miss).round(2).astype(str) + "%")
    miss.round(4).to_csv(os.path.join(EDA, "t10_missing.csv"))
    print("\nNote sul meccanismo:")
    print(" - coastal_dist_m / coastal_change_m_year: missing STRUTTURALE (particelle non")
    print("   costiere) -> non imputati; l'esposizione costiera entra con is_coastal_zone.")
    print(" - fire_score: celle EFFIS non bruciabili (urbano denso) -> gia' riempito con")
    print("   la cella bruciabile piu' vicina (fire_score_near); quota diretta vs near sotto.")
    print(" - subsidenza/pga: missing tecnico raro -> listwise deletion nel modello,")
    print("   sensibilita' se >5%.")
    print(f"\nfire_is_direct (quota particelle della cella con dato EFFIS diretto): "
          f"media={cell['fire_is_direct'].mean():.3f}")
    fna = cell["fire_score"].isna().mean()
    print(f"Celle senza fire_score anche dopo nearest: {100*fna:.2f}%")

    # ------------------------------------------------------------------ 8
    sec("8. ERRORE DI MISURA / DISTANZE DI ATTRIBUZIONE (S11)")
    pq = part.drop_duplicates("fid")
    for v, lab in [("precip_dist_km", "rr20mm (E-OBS 0.1 grad)"),
                   ("p99_dist_km", "r99pday (E-OBS 0.1 grad)"),
                   ("fire_near_dist_km", "fire_score se attribuito da cella near (EFFIS ~12km)")]:
        s = pq[v].dropna()
        if v == "fire_near_dist_km":
            s = pq.loc[pq["fire_is_direct"] == 0, v].dropna()
        print(f"{lab:<52} media={s.mean():6.2f} km  mediana={s.median():6.2f}  p95={s.quantile(.95):6.2f}")

    # ------------------------------------------------------------------ 9
    sec("9. PSEUDO-REPLICAZIONE E PESI (S3, S7.2)")
    nz = (cell["price_m2_std"].fillna(0) > 1e-6)
    print(f"Celle con std prezzo intra-cella > 0: {nz.sum()} / {len(cell)} ({100*nz.mean():.1f}%)")
    if nz.any():
        print("  (dovute a compresenza di piu' microzone/descrizioni nella stessa zona OMI;")
        print("   il prezzo di cella e' la media -> impatto trascurabile)")
    print("\nDistribuzione n_particelle per cella (rilevante per i pesi, S7.2):")
    print(cell["n_particelle"].describe(percentiles=[.05, .25, .5, .75, .95]).round(1))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(cell["n_particelle"], bins=80)
    ax.set_xlabel("particelle per cella"); ax.set_ylabel("celle")
    fig.tight_layout(); fig.savefig(os.path.join(EDA, "f3_n_particelle_cella.png"), dpi=150)
    plt.close(fig)

    # mappa scatter
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, prov in zip(axes.ravel(), ["Milano", "Napoli", "Roma", "Torino"]):
        g = cell[cell["provincia"] == prov]
        s = ax.scatter(g["lng"], g["lat"], c=np.log(g["price_m2"]), s=6, cmap="viridis")
        ax.set_title(prov, fontsize=10)
        fig.colorbar(s, ax=ax, shrink=0.8, label="log prezzo")
    fig.suptitle("Celle: posizione e log prezzo", y=0.995)
    fig.tight_layout(); fig.savefig(os.path.join(EDA, "f4_mappa_prezzi.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------ 10
    sec("10. TABELLA FONTI DATI DI RISCHIO (S4.2)")
    fonti = pd.DataFrame([
        ["Alluvione (flood_level LPH/MPH/HPH)", "PGRA - Piani Gestione Rischio Alluvioni (ISPRA/Distretti), via API Zornade", "vintage PGRA vigente (II ciclo)", "perimetri poligonali", "punto-in-poligono sulla particella"],
        ["Frana (landslide_level P1-P4, AA)", "PAI - Piani Assetto Idrogeologico (ISPRA/IdroGEO), via API Zornade", "vintage PAI vigente", "perimetri poligonali", "punto-in-poligono sulla particella"],
        ["Sisma (seismic_zone, pga)", "Classificazione sismica comunale + PGA INGV MPS04, via API Zornade", "zonazione vigente 2026", "comune / griglia INGV", "attribuzione comunale / interpolazione puntuale"],
        ["Subsidenza (velocity, risk_class)", "Interferometria satellitare (EGMS o equivalente), via API Zornade", "ultimo ciclo disponibile", "punti/celle locali", "attribuzione alla particella"],
        ["Erosione costiera (distanza, variazione)", "Linee di costa ISPRA/regionali, via API Zornade", "vigente", "transetti costieri", "distanza particella-transetto"],
        ["Incendio (fire_score)", "EFFIS Wildfire Risk Assessment (JRC/Copernicus), download 2026-07-12", "WRA armonizzato (rank terzili)", "EURO-CORDEX ~12 km", "cella contenente; se non bruciabile, cella bruciabile piu' vicina"],
        ["Precipitazioni estreme HEADLINE (r99pday)", "Copernicus C3S_430, CMCC-KNMI, E-OBS, v1.0, download 2026-07-21", "climatologia 1995-2019 (25 anni)", "griglia regolare 0.1 grad (~11 km)", "nearest-neighbor valido + dist_km"],
        ["Precipitazioni estreme robustezza (rr20mm)", "Copernicus C3S_430, CMCC-KNMI, E-OBS, v1.0, download 2026-07-21", "climatologia 1995-2019 (25 anni)", "griglia regolare 0.1 grad (~11 km)", "nearest-neighbor valido + dist_km"],
        ["Quotazioni immobiliari (price_m2)", "OMI - Osservatorio Mercato Immobiliare (Agenzia Entrate), via API Zornade, download 2026-07-09/11", "semestre corrente + precedente", "zona OMI x tipologia x stato", "midpoint intervallo min-max"],
    ], columns=["Variabile", "Fonte", "Vintage", "Risoluzione nativa", "Attribuzione"])
    print(fonti.to_string(index=False, max_colwidth=60))
    fonti.to_csv(os.path.join(EDA, "t11_fonti_rischio.csv"), index=False)

    print("\n[OK] EDA completata. Output in output/eda/")
    sys.stdout = sys.__stdout__
    log.close()


if __name__ == "__main__":
    main()
