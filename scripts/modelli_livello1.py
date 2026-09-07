#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tranche 2 - Modelli econometrici di Livello 1 e diagnostica obbligatoria
(protocollo di studio: S7, S7.1, S7.2, S8, S9, S12, S13, S14, S16, S16.1).

Specifiche HEADLINE (S7.1) - variabile dipendente: log(price_m2) di cella
(punto medio dell'intervallo OMI, unita' statistica S3). Rischi (S4.2, forme
grezze/disaggregate):
  flood_ord      quota-media classe PGRA (0-3) della cella
  landslide_ord  quota-media classe PAI P1-P4 (0-4)
  pga            peak ground acceleration (g), INGV
  sub_sink       subsidenza: -velocity_mm_year (positivo = sprofondamento)
  fire_score     score EFFIS aggregato (rank terzili, 0-1... continuo)
  r99pday        precipitazioni estreme: giorni/anno oltre il P99 locale (headline)
  is_coastal_zone  quota particelle della cella in zona di monitoraggio costiero

  A0 (naive):    log(P_c) = a + RISK_c'b + d_tipologia + d_condizione + e_c
  A1 (benchmark provincia+amenita'):
                 log(P_c) = a + RISK_c'b + d_tipologia + d_condizione
                            + d_fascia + g1*dist_cbd_km + g2*is_urban
                            + alpha_provincia + e_c
  A2 (within-comune, MODELLO PRINCIPALE):
                 log(P_c) = a + RISK_c'b + d_tipologia + d_condizione
                            + d_fascia + g1*dist_cbd_km + g2*is_urban
                            + alpha_comune + e_c
  A2w: come A2, WLS con pesi = n_particelle della cella (S7.2).

Errore: e_c eteroschedastico, correlato entro comune -> SE cluster-robust a
livello di comune (numero cluster verificato, S9).

Diagnostica (S16, sempre riportata): VIF raw e within-comune (soglie 5/10),
Breusch-Pagan, Jarque-Bera, RESET (potenze 2-3 del fitted, Wald cluster),
Cook's D (soglia 4/n), Moran's I sui residui (KNN8 a livello di cella e di
zona OMI), test LM lag/error + versioni robuste (spreg), R2/adjR2/AIC/BIC/
RMSE/MAE, conteggio parametri assorbiti dagli FE (S16.1), ICC (decisione
mixed model, S0.1).

Output: output/modelli/ (tabelle CSV, residui per Tranche 3, log completo).
Uso:  py modelli_livello1.py
"""
import os
import sys
import io

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats as sps

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")
MOD = os.path.join(OUT, "modelli")
os.makedirs(MOD, exist_ok=True)

RISKS = ["flood_ord", "landslide_ord", "pga", "sub_sink", "fire_score",
         "r99pday", "is_coastal_zone"]
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


def sec(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def fit_ols(formula, df, weights=None):
    if weights is None:
        model = smf.ols(formula, data=df)
    else:
        model = smf.wls(formula, data=df, weights=weights)
    return model.fit(cov_type="cluster", cov_kwds={"groups": df["comune"]})


def coef_table(res, df, spec_name):
    """Standard di reporting S8: stima, IC95, p grezzo, p Holm (famiglia =
    7 rischi della specifica), magnitudine economica (% per +1 sd e per +1
    unita')."""
    rows = []
    pvals = [res.pvalues[v] for v in RISKS]
    holm = multipletests(pvals, method="holm")[1]
    ci = res.conf_int()
    for v, ph in zip(RISKS, holm):
        b = res.params[v]
        sd = df[v].std()
        rows.append({
            "spec": spec_name, "rischio": v, "beta": b,
            "ci95_low": ci.loc[v, 0], "ci95_high": ci.loc[v, 1],
            "se": res.bse[v], "p_grezzo": res.pvalues[v], "p_holm": ph,
            "sd_var": sd, "pct_per_1sd": 100 * (np.exp(b * sd) - 1),
            "pct_per_1unit": 100 * (np.exp(b) - 1),
        })
    return pd.DataFrame(rows)


def fit_stats(res, df, label):
    resid = res.resid
    return {
        "spec": label, "N": int(res.nobs),
        "cluster_comuni": df["comune"].nunique(),
        "R2": res.rsquared, "adjR2": res.rsquared_adj,
        "AIC": res.aic, "BIC": res.bic,
        "RMSE": float(np.sqrt(np.mean(resid ** 2))),
        "MAE": float(np.mean(np.abs(resid))),
        "k_parametri": int(res.df_model + 1),
        "df_resid": int(res.df_resid),
    }


def indep_columns(X):
    """Seleziona colonne linearmente indipendenti via QR con pivoting."""
    from scipy.linalg import qr
    _, r, piv = qr(X, mode="economic", pivoting=True)
    tol = np.abs(r[0, 0]) * max(X.shape) * np.finfo(float).eps
    rank = int((np.abs(np.diag(r)) > tol).sum())
    keep = sorted(piv[:rank])
    return keep


def main():
    log = open(os.path.join(MOD, "modelli_summary.txt"), "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log)

    cell = pd.read_csv(os.path.join(OUT, "celle.csv"))
    cell["sub_sink"] = -cell["subsidence_velocity"]
    cell["log_price"] = np.log(cell["price_m2"])

    model_vars = (["log_price", "price_m2", "comune", "provincia", "zone",
                   "property_type", "condition", "fascia", "dist_cbd_km",
                   "is_urban", "n_particelle", "lat", "lng"] + RISKS)
    df = cell.dropna(subset=[v for v in model_vars if v != "zone"]).copy()
    df["zone_id"] = df["provincia"] + "|" + df["comune"] + "|" + df["zone"].astype(str)

    sec("0. CAMPIONE DI STIMA E VERIFICA CLUSTER (S9)")
    print(f"Celle totali: {len(cell)}   celle nel campione di stima (listwise su "
          f"variabili modello): {len(df)}  ({100*len(df)/len(cell):.1f}%)")
    print(f"Missing esclusi: {len(cell)-len(df)} celle "
          f"(subsidenza {cell['subsidence_velocity'].isna().sum()}, "
          f"fire {cell['fire_score'].isna().sum()})")
    n_com = df["comune"].nunique()
    n_zone = df["zone_id"].nunique()
    print(f"Cluster: comuni={n_com}  zone OMI={n_zone}  province={df['provincia'].nunique()}")
    print("Livello di clustering primario: COMUNE (>>30 cluster -> SE cluster-robust")
    print("asintotici affidabili; il wild cluster bootstrap NON si attiva a questo")
    print("livello, S9). Il livello provincia (4 cluster) NON e' usato per inferenza;")
    print("le verifiche leave-one-provincia-out sono in Tranche 3 (S10).")

    # ------------------------------------------------------------------
    sec("1. SPECIFICHE HEADLINE (S7.1) - equazioni")
    risk_str = " + ".join(RISKS)
    f_a0 = f"log_price ~ {risk_str} + {CONTROLS_A0}"
    f_a1 = f"log_price ~ {risk_str} + {CONTROLS_A0} + {CONTROLS_LOC} + C(provincia)"
    f_a2 = f"log_price ~ {risk_str} + {CONTROLS_A0} + {CONTROLS_LOC} + C(comune)"
    print("A0:", f_a0)
    print("A1:", f_a1)
    print("A2:", f_a2)
    print("A2w: come A2, WLS pesi=n_particelle (S7.2)")
    print("SE: cluster-robust per comune in tutte le specifiche.")

    res_a0 = fit_ols(f_a0, df)
    res_a1 = fit_ols(f_a1, df)
    res_a2 = fit_ols(f_a2, df)
    res_a2w = fit_ols(f_a2, df, weights=df["n_particelle"])

    tabs = []
    for r, nm in [(res_a0, "A0"), (res_a1, "A1"), (res_a2, "A2"), (res_a2w, "A2w")]:
        tabs.append(coef_table(r, df, nm))
    coefs = pd.concat(tabs, ignore_index=True)
    coefs.round(5).to_csv(os.path.join(MOD, "m1_coefficienti.csv"), index=False)

    stats_rows = [fit_stats(res_a0, df, "A0"), fit_stats(res_a1, df, "A1"),
                  fit_stats(res_a2, df, "A2"), fit_stats(res_a2w, df, "A2w")]
    fs = pd.DataFrame(stats_rows).set_index("spec")
    fs.round(4).to_csv(os.path.join(MOD, "m2_fit.csv"))

    sec("2. COEFFICIENTI DI RISCHIO (S8) - traiettoria A0 -> A1 -> A2")
    for nm in ["A0", "A1", "A2", "A2w"]:
        t = coefs[coefs["spec"] == nm][["rischio", "beta", "ci95_low", "ci95_high",
                                        "p_grezzo", "p_holm", "pct_per_1sd"]]
        print(f"\n--- {nm} ---")
        print(t.round(4).to_string(index=False))
    print("\nFit:")
    print(fs.round(3))

    # figura traiettoria
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(15, 6), sharex=True)
    for ax, v in zip(axes.ravel(), RISKS):
        sub = coefs[(coefs["rischio"] == v) & (coefs["spec"].isin(["A0", "A1", "A2"]))]
        x = np.arange(len(sub))
        ax.errorbar(x, sub["beta"], yerr=[sub["beta"] - sub["ci95_low"],
                                          sub["ci95_high"] - sub["beta"]],
                    fmt="o", capsize=4)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(sub["spec"])
        ax.set_title(v, fontsize=9)
    axes[1, 3].axis("off")
    fig.suptitle("Traiettoria dei coefficienti di rischio (IC95, SE cluster comune)")
    fig.tight_layout()
    fig.savefig(os.path.join(MOD, "f5_traiettoria_coefficienti.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    sec("3. SATURAZIONE FE E GRADI DI LIBERTA' EFFETTIVI (S16.1)")
    print(f"R2: A0={res_a0.rsquared:.3f} -> A1={res_a1.rsquared:.3f} -> A2={res_a2.rsquared:.3f}")
    print(f"A2: parametri totali={int(res_a2.df_model)+1} (di cui ~{n_com-1} dummy comune), "
          f"N={int(res_a2.nobs)}, df residui={int(res_a2.df_resid)}")
    zone_per_com = df.groupby("comune")["zone_id"].nunique()
    print(f"Comuni con 1 sola zona OMI quotata: {(zone_per_com==1).sum()} / {n_com} "
          f"(contribuiscono poco alla variazione within del rischio)")
    within_share = {}
    for v in RISKS:
        tot = df[v].var()
        within = df.groupby("comune")[v].transform(lambda s: s - s.mean()).var()
        within_share[v] = within / tot if tot > 0 else np.nan
    ws = pd.Series(within_share)
    print("\nQuota di varianza del rischio che sopravvive within-comune (A2):")
    print((100 * ws).round(1).astype(str) + "%")
    ws.to_csv(os.path.join(MOD, "m3_within_variance_share.csv"))

    # ------------------------------------------------------------------
    sec("4. MULTICOLLINEARITA' - VIF (S13, soglie: 5 attenzione, 10 critica)")
    Xv = df[RISKS + ["dist_cbd_km", "is_urban"]].copy()
    Xv = sm.add_constant(Xv)
    vif_raw = pd.Series(
        [variance_inflation_factor(Xv.values, i) for i in range(1, Xv.shape[1])],
        index=Xv.columns[1:], name="VIF_raw")
    Xw = df[RISKS + ["dist_cbd_km", "is_urban"]].copy()
    for c in Xw.columns:
        Xw[c] = Xw[c] - df.groupby("comune")[c].transform("mean")
    Xw = sm.add_constant(Xw)
    vif_within = pd.Series(
        [variance_inflation_factor(Xw.values, i) for i in range(1, Xw.shape[1])],
        index=Xw.columns[1:], name="VIF_within_comune")
    vt = pd.concat([vif_raw, vif_within], axis=1)
    print(vt.round(2))
    vt.round(3).to_csv(os.path.join(MOD, "m4_vif.csv"))
    n_crit = (vt["VIF_within_comune"] > 10).sum()
    print(f"\nVariabili con VIF within > 10: {n_crit}  |  > 5: "
          f"{(vt['VIF_within_comune']>5).sum()}")

    # ------------------------------------------------------------------
    sec("5. DIAGNOSTICA RESIDUI A2 (S16)")
    resid = res_a2.resid
    y, X = patsy.dmatrices(f_a2, df, return_type="dataframe")
    bp_lm, bp_p, _, _ = het_breuschpagan(resid, X.values)
    print(f"Breusch-Pagan: LM={bp_lm:.1f}  p={bp_p:.2e}  "
          f"({'eteroschedasticita'' presente' if bp_p<0.05 else 'omoschedasticita'' non rigettata'} "
          f"-> SE cluster-robust gia' adottati)")
    jb_stat, jb_p = sps.jarque_bera(resid)[:2]
    print(f"Jarque-Bera: JB={jb_stat:.1f}  p={jb_p:.2e}  "
          f"skew={sps.skew(resid):.3f}  kurt.ecc={sps.kurtosis(resid):.3f}")

    # RESET (potenze 2 e 3 del fitted, Wald con cov cluster)
    df_r = df.copy()
    fit_std = (res_a2.fittedvalues - res_a2.fittedvalues.mean()) / res_a2.fittedvalues.std()
    df_r["yhat2"] = fit_std ** 2
    df_r["yhat3"] = fit_std ** 3
    res_reset = fit_ols(f_a2 + " + yhat2 + yhat3", df_r)
    w = res_reset.wald_test("yhat2 = 0, yhat3 = 0", scalar=True)
    print(f"RESET (Ramsey, fitted^2, fitted^3; Wald cluster comune): "
          f"stat={float(w.statistic):.2f}  p={float(w.pvalue):.4f}")
    reset_reject = float(w.pvalue) < 0.05

    # Cook's distance (soglia 4/n), con gestione della leva ~1: le celle
    # interamente assorbite da un FE quasi-singleton hanno h_ii -> 1 e Cook
    # numericamente indefinito; vanno segnalate come "assorbite", non come
    # influenti in senso proprio.
    infl = res_a2.get_influence()
    hii = infl.hat_matrix_diag
    cooks = infl.cooks_distance[0]
    n_sing = (hii > 0.99).sum()
    ok = hii <= 0.99
    thr = 4 / len(df)
    n_infl = (cooks[ok] > thr).sum()
    print(f"Osservazioni con leva h_ii>0.99 (assorbite da FE quasi-singleton, "
          f"Cook indefinito): {n_sing}")
    print(f"Cook's D (su h_ii<=0.99): soglia 4/n={thr:.5f}; oltre soglia: {n_infl} "
          f"({100*n_infl/ok.sum():.1f}%); max={np.nanmax(cooks[ok]):.4f}")
    top_infl = df.loc[ok].assign(cooks=cooks[ok]).nlargest(5, "cooks")[
        ["provincia", "comune", "zone", "property_type", "price_m2", "cooks"]]
    print("Top-5 Cook (escluse leve ~1):")
    print(top_infl.round(4).to_string(index=False))
    cells_per_com = df.groupby("comune").size()
    print(f"Comuni con 1 sola cella nel campione (osservazione assorbita dal "
          f"proprio FE): {(cells_per_com==1).sum()}")
    df_out = df.assign(residuo_a2=resid.values, fitted_a2=res_a2.fittedvalues.values,
                       cooks_a2=cooks)
    df_out.to_csv(os.path.join(MOD, "m5_campione_stima_residui.csv"), index=False)

    # ------------------------------------------------------------------
    sec("6. DIPENDENZA SPAZIALE (S12) - Moran e test LM")
    import libpysal
    from esda.moran import Moran

    # (a) livello ZONA OMI: residuo medio di zona, KNN8 su centroidi di zona.
    # Confronto A1 vs A2: gli FE di comune impongono somma ~0 dei residui
    # entro comune -> i vicini della stessa citta' risultano meccanicamente
    # anticorrelati (I negativo atteso come artefatto di demeaning). Il
    # segnale genuino di dipendenza spaziale va cercato (i) nei residui A1
    # e (ii) in una W che esclude i vicini dello stesso comune.
    df_out["residuo_a1"] = res_a1.resid.values
    zres = df_out.groupby("zone_id").agg(res=("residuo_a2", "mean"),
                                         res_a1=("residuo_a1", "mean"),
                                         comune=("comune", "first"),
                                         lat=("lat", "mean"), lng=("lng", "mean"))
    wz = libpysal.weights.KNN.from_array(zres[["lng", "lat"]].values, k=8)
    wz.transform = "r"
    mz = Moran(zres["res"].values, wz, permutations=9999)
    mz_a1 = Moran(zres["res_a1"].values, wz, permutations=9999)
    print(f"(a) Moran's I sui residui medi di ZONA (n={len(zres)}, KNN k=8):")
    print(f"      A1 (FE provincia): I={mz_a1.I:+.4f}  p_perm={mz_a1.p_sim:.4f}")
    print(f"      A2 (FE comune):    I={mz.I:+.4f}  E[I]={mz.EI:.4f}  p_perm={mz.p_sim:.4f}")

    # (a-bis) W tra zone di COMUNI DIVERSI (esclusi i vicini dello stesso
    # comune, poi rinormalizzata): testa la dipendenza spaziale residua non
    # riconducibile all'artefatto within-comune.
    zid = list(zres.index)
    com_arr = zres["comune"].values
    neigh_cross = {}
    for i, ns in wz.neighbors.items():
        keep_n = [j for j in ns if com_arr[j] != com_arr[i]]
        neigh_cross[i] = keep_n
    n_isole = sum(1 for v in neigh_cross.values() if not v)
    wcross = libpysal.weights.W(neigh_cross, silence_warnings=True)
    wcross.transform = "r"
    mask = np.array([len(neigh_cross[i]) > 0 for i in range(len(zid))])
    mzc = Moran(zres["res"].values, wcross, permutations=9999)
    mzc_a1 = Moran(zres["res_a1"].values, wcross, permutations=9999)
    print(f"(a-bis) Moran's I, W solo vicini di ALTRI comuni (isole senza vicini: {n_isole}):")
    print(f"      A1: I={mzc_a1.I:+.4f}  p_perm={mzc_a1.p_sim:.4f}")
    print(f"      A2: I={mzc.I:+.4f}  p_perm={mzc.p_sim:.4f}")

    # (b) livello CELLA: KNN8 (nota: celle della stessa zona condividono il
    # centroide -> la componente within-zona e' gia' gestita dal cluster SE;
    # il test di zona (a) e' il segnale primario di attivazione, S12).
    coords = df_out[["lng", "lat"]].values
    rng = np.random.default_rng(42)
    jit = coords + rng.normal(0, 1e-6, coords.shape)  # rompe i punti coincidenti
    wc = libpysal.weights.KNN.from_array(jit, k=8)
    wc.transform = "r"
    mc = Moran(df_out["residuo_a2"].values, wc, permutations=999)
    print(f"(b) Moran's I sui residui di CELLA (n={len(df_out)}, KNN k=8): "
          f"I={mc.I:.4f}  p_perm={mc.p_sim:.4f}")

    # test LM su A2 (spreg, W di cella)
    lm_ok = False
    try:
        from spreg import OLS as SpregOLS
        Xmat = X.values
        keep = indep_columns(Xmat)
        keep_names = [X.columns[i] for i in keep]
        drop_const = [i for i in keep if X.columns[i] != "Intercept"]
        Xs = Xmat[:, drop_const]
        sp = SpregOLS(np.asarray(y), Xs, w=wc, spat_diag=True, moran=False,
                      name_y="log_price", name_x=[X.columns[i] for i in drop_const])
        print(f"\nTest LM (spreg, W=KNN8 cella, {Xs.shape[1]} regressori):")
        print(f"  LM-lag:          stat={sp.lm_lag[0]:8.2f}  p={sp.lm_lag[1]:.4f}")
        print(f"  LM-error:        stat={sp.lm_error[0]:8.2f}  p={sp.lm_error[1]:.4f}")
        print(f"  LM-lag robusto:  stat={sp.rlm_lag[0]:8.2f}  p={sp.rlm_lag[1]:.4f}")
        print(f"  LM-error robusto:stat={sp.rlm_error[0]:8.2f}  p={sp.rlm_error[1]:.4f}")
        lm_ok = True
        lm_res = {"lm_lag_p": sp.lm_lag[1], "lm_err_p": sp.lm_error[1],
                  "rlm_lag_p": sp.rlm_lag[1], "rlm_err_p": sp.rlm_error[1]}
    except Exception as e:
        print("spreg non disponibile/fallito:", e)
        lm_res = {}

    # ------------------------------------------------------------------
    sec("7. ICC / STRUTTURA GERARCHICA (decisione mixed model, S0.1)")
    for g, lab in [("comune", "comune"), ("zone_id", "zona OMI")]:
        try:
            mm = sm.MixedLM.from_formula("log_price ~ 1", groups=df[g], data=df)
            mr = mm.fit(reml=True)
            var_re = float(mr.cov_re.iloc[0, 0])
            var_e = float(mr.scale)
            icc = var_re / (var_re + var_e)
            print(f"ICC log(price) a livello di {lab}: {icc:.3f}")
        except Exception as e:
            print(f"MixedLM {lab} fallito: {e}")
    print("Nota: la varianza di cluster e' gia' assorbita dagli FE di comune in A2")
    print("-> il mixed model NON si attiva (condizione S0.1: varianza di cluster")
    print("   non gia' catturata da un effetto fisso).")

    # ------------------------------------------------------------------
    sec("8. DECISIONI DI ATTIVAZIONE LIVELLO 2 (S0.1)")
    dec = []
    cross_dep = mzc.p_sim < 0.05
    rlm_sig = lm_ok and (lm_res.get("rlm_lag_p", 1) < 0.05 or lm_res.get("rlm_err_p", 1) < 0.05)
    dec.append(("Modelli spaziali SAR/SEM/SDM",
                cross_dep and rlm_sig,
                f"Moran zona A2 (W cross-comune, al netto dell'artefatto da "
                f"demeaning) I={mzc.I:+.3f} p={mzc.p_sim:.4f}; "
                f"Moran zona A2 (W piena) I={mz.I:+.3f} p={mz.p_sim:.4f}; "
                f"robust LM sig={rlm_sig}"))
    dec.append(("Forme non lineari (spline/GAM/dummy classi)",
                reset_reject, f"RESET p={float(w.pvalue):.4f}"))
    dec.append(("Trattamento multicollinearita' (PCA/Ridge/blocchi)",
                n_crit >= 2, f"variabili con VIF within>10: {n_crit}"))
    dec.append(("Mixed/multilevel model", False,
                "varianza di cluster gia' assorbita dagli FE comune"))
    dec.append(("Machine learning (RF/GBM) sintetico", True,
                "sempre previsto come controllo indipendente (S15)"))
    for nome, att, mot in dec:
        print(f"  [{'ATTIVA' if att else 'non attivare'}] {nome} — {mot}")
    pd.DataFrame(dec, columns=["tecnica", "attivata", "motivazione"]).to_csv(
        os.path.join(MOD, "m6_decisioni_livello2.csv"), index=False)

    print("\n[OK] Tranche 2 completata. Output in output/modelli/")
    sys.stdout = sys.__stdout__
    log.close()


if __name__ == "__main__":
    main()
