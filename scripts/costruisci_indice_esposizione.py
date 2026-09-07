#!/usr/bin/env python3
"""
Costruisce l'indice di esposizione fisica al rischio climatico-ambientale
del patrimonio immobiliare, a livello di zona OMI (provincia x comune x
zona), per le quattro province gia' coperte dall'analisi primaria
(Milano, Napoli, Roma, Torino).

Disegno concordato (vedi discussione in tesi, Cap. 3):
  - Due sotto-indici, non uno solo:
      Indice A = rischio da evento/danno diretto (7 componenti):
                 alluvione, frana, sisma (pga), subsidenza, incendio,
                 precipitazioni estreme, costa
      Indice B = rischio climatico cronico (2 componenti):
                 calore estremo (su95p), siccita' (cdd)
    stessa partizione gia' usata nei report per la correzione di Holm.
  - Pesi uguali entro ciascun sotto-indice (VIF basso tra i rischi,
    nessuna base statistica per pesarli diversamente); nota SAW/OWA di
    Mysiak et al. (2018) come precedente citabile.
  - Sisma: si usa pga (continua), non seismic_zone (ordinale invertita
    per convenzione italiana: 1=piu' grave, 4=meno grave) per evitare
    di contare due volte lo stesso fenomeno e il rischio di segno.
  - Costa: si usa is_coastal_zone (gia' una quota continua [0,1] di
    particelle costiere per cella), non coastal_dist_m (95,7% mancante
    a livello di cella, quindi impraticabile).
  - Esposizione monetaria: NON un ammontare aggregato di zona. Il
    database campiona 100 particelle per comune (non e' un censimento),
    quindi n_particelle riflette la densita' di campionamento, non lo
    stock reale di unita' immobiliari: moltiplicarlo per il prezzo
    produrrebbe un aggregato fuorviante. Si riporta invece un valore
    "per unita' tipo": prezzo OMI residenziale di zona x superficie
    convenzionale di un'abitazione media (parametro esplicito,
    default 90 mq, fonte ISTAT/Banca d'Italia) x grado di esposizione.
    Interpretazione: quanto valore avrebbe, ponderato per il grado di
    rischio, un'abitazione media in quella zona - un indicatore di
    intensita' per unita' di stock, non una stima di ricchezza totale.

Input:  output/celle_con_calore.csv (2.793 celle zona OMI x tipologia x
        stato x fascia, gia' con cdd/su95p integrati)
Output: output/indice_esposizione_zone.csv (una riga per zona OMI)
        output/indice_esposizione_summary.txt (sintesi leggibile)
"""
import pandas as pd
import numpy as np

IN_PATH = "output/celle_con_calore.csv"
OUT_CSV = "output/indice_esposizione_zone.csv"
OUT_SUMMARY = "output/indice_esposizione_summary.txt"

SUPERFICIE_CONVENZIONALE_MQ = 90.0  # abitazione media, parametro esplicito
TIPI_RESIDENZIALI = ["Abitazioni civili", "Abitazioni di tipo economico", "Ville e Villini"]

RISCHI_A = ["flood_ord", "landslide_ord", "pga", "subsidence_velocity_abs",
            "fire_score", "r99pday", "is_coastal_zone"]
RISCHI_B = ["cdd", "su95p"]


def minmax(s):
    """Normalizza in [0,1] su min/max campionari, ignorando i mancanti."""
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


def main():
    df = pd.read_csv(IN_PATH)
    n_celle_raw = len(df)

    # --- 1. Collassa a livello di zona OMI (provincia x comune x zone) ---
    # I rischi sono gia' pressoche' identici tra le righe tipologia/stato/
    # fascia della stessa zona (verificato: 432/433 zone hanno un solo
    # valore distinto di flood_ord); si prende la media per le poche
    # eccezioni residue.
    key = ["provincia", "comune", "zone"]
    risk_cols_raw = ["flood_ord", "landslide_ord", "pga", "subsidence_velocity",
                      "fire_score", "r99pday", "is_coastal_zone", "cdd", "su95p"]
    zone = df.groupby(key, as_index=False)[risk_cols_raw].mean()

    # numero di particelle campionate per zona (somma sulle sotto-celle;
    # e' un indicatore di densita' campionaria, non di stock reale - vedi
    # avvertenza nel modulo)
    n_part = df.groupby(key, as_index=False)["n_particelle"].sum() \
                .rename(columns={"n_particelle": "n_particelle_campionate"})
    zone = zone.merge(n_part, on=key)

    n_zone = len(zone)

    # --- 2. Prezzo OMI residenziale rappresentativo di zona ---
    resid = df[df["property_type"].isin(TIPI_RESIDENZIALI)]
    prezzo_normale = (resid[resid["condition"] == "NORMALE"]
                       .groupby(key, as_index=False)["price_m2"].mean()
                       .rename(columns={"price_m2": "price_m2_residenziale"}))
    # fallback: se manca NORMALE in una zona, usa la media su tutte le
    # condizioni disponibili per le tipologie residenziali
    prezzo_fallback = (resid.groupby(key, as_index=False)["price_m2"].mean()
                        .rename(columns={"price_m2": "price_m2_residenziale_fallback"}))

    zone = zone.merge(prezzo_normale, on=key, how="left")
    zone = zone.merge(prezzo_fallback, on=key, how="left")
    n_fallback = zone["price_m2_residenziale"].isna().sum()
    zone["price_m2_residenziale"] = zone["price_m2_residenziale"].fillna(
        zone["price_m2_residenziale_fallback"])
    n_no_price = zone["price_m2_residenziale"].isna().sum()
    zone = zone.drop(columns=["price_m2_residenziale_fallback"])

    # --- 3. Trasformazioni e normalizzazione [0,1] ---
    zone["subsidence_velocity_abs"] = zone["subsidence_velocity"].abs()

    # scale regolamentari note (PGRA a 3 classi, PAI a 4 classi)
    zone["n_flood_ord"] = (zone["flood_ord"] / 3.0).clip(0, 1)
    zone["n_landslide_ord"] = (zone["landslide_ord"] / 4.0).clip(0, 1)
    # gia' una quota [0,1]
    zone["n_is_coastal_zone"] = zone["is_coastal_zone"].clip(0, 1)
    # min-max campionario per le variabili continue senza soglia ufficiale
    for c in ["pga", "subsidence_velocity_abs", "fire_score", "r99pday", "cdd", "su95p"]:
        zone[f"n_{c}"] = minmax(zone[c])

    norm_cols_A = ["n_flood_ord", "n_landslide_ord", "n_pga",
                   "n_subsidence_velocity_abs", "n_fire_score", "n_r99pday",
                   "n_is_coastal_zone"]
    norm_cols_B = ["n_cdd", "n_su95p"]

    # media dei componenti disponibili (missing pairwise, quota bassa:
    # subsidenza 1,9%, incendio 0,5% a livello di particella)
    zone["indice_A"] = zone[norm_cols_A].mean(axis=1, skipna=True)
    zone["indice_B"] = zone[norm_cols_B].mean(axis=1, skipna=True)
    zone["indice_generale"] = zone[["indice_A", "indice_B"]].mean(axis=1)

    # sensitivity: pesi raddoppiati su alluvione e sisma dentro l'indice A
    pesi_alt = {"n_flood_ord": 2, "n_landslide_ord": 1, "n_pga": 2,
                "n_subsidence_velocity_abs": 1, "n_fire_score": 1,
                "n_r99pday": 1, "n_is_coastal_zone": 1}
    tot_w = sum(pesi_alt.values())
    zone["indice_A_pesi_alt"] = sum(zone[c] * w for c, w in pesi_alt.items()) / tot_w
    corr_pesi = zone["indice_A"].corr(zone["indice_A_pesi_alt"])
    rank_corr_pesi = zone["indice_A"].rank().corr(zone["indice_A_pesi_alt"].rank())

    # classi di lettura (terzili sul campione)
    for c in ["indice_A", "indice_B", "indice_generale"]:
        zone[f"{c}_classe"] = pd.qcut(zone[c], 3, labels=["Basso", "Medio", "Alto"])

    # --- 4. Valore unitario esposto (per abitazione media) ---
    for c in ["indice_A", "indice_B", "indice_generale"]:
        zone[f"valore_esposto_{c}"] = (
            zone["price_m2_residenziale"] * SUPERFICIE_CONVENZIONALE_MQ * zone[c]
        )

    # --- 5. Diagnostica ---
    corr_AB = zone["indice_A"].corr(zone["indice_B"])

    zone = zone.sort_values(["provincia", "indice_generale"], ascending=[True, False])
    zone.to_csv(OUT_CSV, index=False)

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write("INDICE DI ESPOSIZIONE FISICA AL RISCHIO CLIMATICO-AMBIENTALE\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Celle di input: {n_celle_raw}\n")
        f.write(f"Zone OMI (provincia x comune x zona): {n_zone}\n")
        f.write(f"Zone senza prezzo NORMALE (fallback su altre condizioni): {n_fallback}\n")
        f.write(f"Zone senza alcun prezzo residenziale disponibile: {n_no_price}\n\n")

        f.write("Copertura per provincia:\n")
        f.write(zone.groupby("provincia").size().to_string() + "\n\n")

        f.write("Correlazione Indice A (danno diretto) vs Indice B (cronico): "
                f"{corr_AB:.3f}\n")
        f.write("(bassa attesa: sono dimensioni concettualmente distinte, "
                "coerente con la partizione usata per Holm nei report)\n\n")

        f.write("Sensitivity pesi (Indice A base vs pesi doppi su alluvione/sisma):\n")
        f.write(f"  correlazione di Pearson:  {corr_pesi:.3f}\n")
        f.write(f"  correlazione di rango:    {rank_corr_pesi:.3f}\n\n")

        f.write("Statistiche descrittive degli indici:\n")
        f.write(zone[["indice_A", "indice_B", "indice_generale"]].describe().to_string())
        f.write("\n\n")

        f.write("Top 5 zone per Indice generale, per provincia:\n")
        for prov, g in zone.groupby("provincia"):
            f.write(f"\n-- {prov} --\n")
            top = g.nlargest(5, "indice_generale")[
                ["comune", "zone", "indice_A", "indice_B", "indice_generale",
                 "price_m2_residenziale", "valore_esposto_indice_generale"]]
            f.write(top.to_string(index=False))
            f.write("\n")

    print(f"Fatto. Zone: {n_zone}. Output: {OUT_CSV}, {OUT_SUMMARY}")
    print(f"Corr(A,B) = {corr_AB:.3f}  |  Corr pesi alt (rango) = {rank_corr_pesi:.3f}")
    print(f"Zone senza prezzo residenziale: {n_no_price} su {n_zone}")


if __name__ == "__main__":
    main()
