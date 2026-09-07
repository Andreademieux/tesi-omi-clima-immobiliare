#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tranche 3b - Estensioni attivate dalla diagnostica (protocollo S0.1) e
analisi complementari:

  1. FORME NON LINEARI (Livello 2, ATTIVATA dal RESET p<0.0001 su A2):
     - diagnosi della fonte del rigetto: spline naturale su dist_cbd_km;
     - dummy per classi di pericolosita': quote di particelle in classe
       PGRA LPH/MPH/HPH (alluvione) e PAI P1-P2/P3-P4 (frana) al posto
       dell'ordinale; quartili di r99pday in A1.
  2. EFFETTI ETEROGENEI PER TIPOLOGIA (S5): interazioni rischio x gruppo
     (Residenziale / Commerciale / Produttivo / Box), effetti marginali per
     gruppo con Holm entro famiglia e test congiunto di uguaglianza.
  3. MACHINE LEARNING SINTETICO (Livello 2, sempre previsto, S15):
     GradientBoosting e RandomForest con GroupKFold per comune (no leakage
     spaziale), delta R2 con/senza rischi, permutation importance out-of-fold,
     PDP e mean|SHAP| per i rischi.
  4. REPRICING DINAMICO PRELIMINARE (S17): delta log prezzo tra i 2 semestri
     OMI disponibili ~ rischi (indizio, non panel).
  5. CANONI E RAPPORTO CANONE/PREZZO (S2): stesso modello su log(canone) e
     su log(canone/prezzo) per discriminare "mercato non prezza" vs
     "strumento non riflette".

Le specifiche con forma funzionale diversa sono headline a se' stanti
(equazioni nel log); le altre sono richiamate per delta (S7.1).

Output: output/estensioni/. Uso: py estensioni_livello2.py
"""
import os
import sys
import io

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")
EST = os.path.join(OUT, "estensioni")
os.makedirs(EST, exist_ok=True)

RISKS = ["flood_ord", "landslide_ord", "pga", "sub_sink", "fire_score",
         "r99pday", "is_coastal_zone"]
C_A0 = "C(property_type) + C(condition)"
C_LOC = "C(fascia) + dist_cbd_km + is_urban"
GRUPPI = {
    "Abitazioni civili": "Residenziale", "Abitazioni di tipo economico": "Residenziale",
    "Ville e Villini": "Residenziale", "Negozi": "Commerciale", "Uffici": "Commerciale",
    "Uffici strutturati": "Commerciale", "Capannoni industriali": "Produttivo",
    "Capannoni tipici": "Produttivo", "Laboratori": "Produttivo",
    "Magazzini": "Produttivo", "Box": "Box", "Posti auto coperti": "Box",
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
            try:
                st.flush()
            except ValueError:
                pass


def sec(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


def fit(formula, d, w=None):
    m = smf.wls(formula, data=d, weights=w) if w is not None else smf.ols(formula, data=d)
    return m.fit(cov_type="cluster", cov_kwds={"groups": d["comune"]})


def load_df():
    cell = pd.read_csv(os.path.join(OUT, "celle.csv"))
    cell["sub_sink"] = -cell["subsidence_velocity"]
    cell["log_price"] = np.log(cell["price_m2"])
    mv = (["log_price", "comune", "provincia", "zone", "property_type",
           "condition", "fascia", "dist_cbd_km", "is_urban", "n_particelle"]
          + RISKS)
    df = cell.dropna(subset=mv).copy().reset_index(drop=True)
    df["zone_id"] = df["provincia"] + "|" + df["comune"] + "|" + df["zone"].astype(str)
    df["gruppo"] = df["property_type"].map(GRUPPI)
    return df


def add_class_shares(df):
    """Quote di particelle per classe PGRA/PAI, ricostruite da particelle.csv
    sulla stessa chiave di cella."""
    part = pd.read_csv(os.path.join(OUT, "particelle.csv"), low_memory=False)
    part = part[part["property_type"].notna()].copy()
    key = ["provincia", "comune", "zone", "property_type", "condition", "fascia"]
    fl = pd.get_dummies(part["flood_level"].fillna("NONE"))
    for c in ["LPH", "MPH", "HPH"]:
        if c not in fl:
            fl[c] = 0.0
    part[["sh_LPH", "sh_MPH", "sh_HPH"]] = fl[["LPH", "MPH", "HPH"]].astype(float)
    ls = part["landslide_level"].fillna("NONE")
    part["sh_P12"] = ls.isin(["P1", "P2"]).astype(float)
    part["sh_P34"] = ls.isin(["P3", "P4"]).astype(float)
    shares = part.groupby(key, dropna=False)[
        ["sh_LPH", "sh_MPH", "sh_HPH", "sh_P12", "sh_P34"]].mean().reset_index()
    return df.merge(shares, on=key, how="left")


def reset_test(res, formula, d):
    d2 = d.copy()
    fs = (res.fittedvalues - res.fittedvalues.mean()) / res.fittedvalues.std()
    d2["yhat2"] = fs ** 2
    d2["yhat3"] = fs ** 3
    rr = fit(formula + " + yhat2 + yhat3", d2)
    w = rr.wald_test("yhat2 = 0, yhat3 = 0", scalar=True)
    return float(w.statistic), float(w.pvalue)


def main():
    log = open(os.path.join(EST, "estensioni_summary.txt"), "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log)

    df = load_df()
    df = add_class_shares(df)
    risk_str = " + ".join(RISKS)
    f_a1 = f"log_price ~ {risk_str} + {C_A0} + {C_LOC} + C(provincia)"
    f_a2 = f"log_price ~ {risk_str} + {C_A0} + {C_LOC} + C(comune)"
    res_a1 = fit(f_a1, df)
    res_a2 = fit(f_a2, df)
    ext_rows = []  # per la curva di specificazione estesa

    # ------------------------------------------------------------------ 1
    sec("1. FORME NON LINEARI (Livello 2, attivata: RESET su A2 p<0.0001)")
    st0, p0 = reset_test(res_a2, f_a2, df)
    print(f"RESET A2 headline (riferimento): stat={st0:.1f}  p={p0:.2e}")

    # (a) spline naturale su dist_cbd (bs df=4): la non linearita' e' nei
    # controlli di localizzazione o nei rischi?
    f_a2s = f_a2.replace("dist_cbd_km", "bs(dist_cbd_km, df=4)")
    res_a2s = fit(f_a2s, df)
    st1, p1 = reset_test(res_a2s, f_a2s, df)
    print(f"\n(a) A2 + spline dist_cbd (bs df=4): RESET stat={st1:.1f}  p={p1:.2e}")
    ci = res_a2s.conf_int()
    print("    Coefficienti di rischio con spline su dist_cbd (invarianza attesa):")
    for v in RISKS:
        sd = df[v].std()
        print(f"      {v:<16} beta={res_a2s.params[v]:+.4f}  "
              f"pct/1sd={100*(np.exp(res_a2s.params[v]*sd)-1):+6.2f}%  "
              f"p={res_a2s.pvalues[v]:.3f}")
        ext_rows.append({"spec": "A2 — spline dist_cbd", "famiglia": {"flood_ord": "flood",
                         "landslide_ord": "frana", "pga": "sisma", "sub_sink": "subsidenza",
                         "fire_score": "incendio", "r99pday": "precip",
                         "is_coastal_zone": "costa"}[v],
                         "variabile": v, "beta": res_a2s.params[v],
                         "p_grezzo": res_a2s.pvalues[v], "sd_var": sd,
                         "pct_per_1sd": 100*(np.exp(res_a2s.params[v]*sd)-1),
                         "pct_ci_low": 100*(np.exp(ci.loc[v, 0]*sd)-1),
                         "pct_ci_high": 100*(np.exp(ci.loc[v, 1]*sd)-1),
                         "N": int(res_a2s.nobs)})

    # (b) log(dist_cbd) come alternativa parsimoniosa
    df["log_dist_cbd"] = np.log1p(df["dist_cbd_km"])
    f_a2l = f_a2.replace("dist_cbd_km", "log_dist_cbd")
    res_a2l = fit(f_a2l, df)
    st2, p2 = reset_test(res_a2l, f_a2l, df)
    print(f"(b) A2 con log(1+dist_cbd): RESET stat={st2:.1f}  p={p2:.2e}")

    # (c) dummy per classi di pericolosita' (alluvione, frana) in A2
    f_cl = f_a2.replace("flood_ord", "sh_LPH + sh_MPH + sh_HPH").replace(
        "landslide_ord", "sh_P12 + sh_P34")
    dcl = df.dropna(subset=["sh_LPH", "sh_MPH", "sh_HPH", "sh_P12", "sh_P34"])
    res_cl = fit(f_cl, dcl)
    print("\n(c) A2 con quote per CLASSE di pericolosita' (0->1 = intera cella in classe):")
    tab = []
    for v in ["sh_LPH", "sh_MPH", "sh_HPH", "sh_P12", "sh_P34"]:
        cii = res_cl.conf_int().loc[v]
        tab.append({"classe": v, "beta": res_cl.params[v],
                    "pct_0a1": 100 * (np.exp(res_cl.params[v]) - 1),
                    "ci_low": cii[0], "ci_high": cii[1], "p": res_cl.pvalues[v]})
    tcl = pd.DataFrame(tab)
    print(tcl.round(4).to_string(index=False))
    tcl.round(5).to_csv(os.path.join(EST, "e2_classi_pericolosita.csv"), index=False)
    wj = res_cl.wald_test("sh_LPH = 0, sh_MPH = 0, sh_HPH = 0", scalar=True)
    print(f"    Test congiunto classi alluvione = 0: stat={float(wj.statistic):.2f} "
          f"p={float(wj.pvalue):.4f}")

    # (d) quartili di r99pday in A1 (monotonicita')
    df["r99_q"] = pd.qcut(df["r99pday"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    f_q = f_a1.replace("r99pday", "C(r99_q)")
    res_q = fit(f_q, df)
    print("\n(d) A1 con quartili di r99pday (base Q1):")
    for lv in ["Q2", "Q3", "Q4"]:
        nm = f"C(r99_q)[T.{lv}]"
        print(f"      {lv}: beta={res_q.params[nm]:+.4f} "
              f"({100*(np.exp(res_q.params[nm])-1):+5.1f}%)  p={res_q.pvalues[nm]:.4f}")
    # (e) eterogeneita' della struttura edonica: interazione tipologia x fascia
    f_a2tf = f_a2 + " + C(property_type):C(fascia)"
    res_tf = fit(f_a2tf, df)
    st3, p3 = reset_test(res_tf, f_a2tf, df)
    print(f"\n(e) A2 + C(property_type):C(fascia): RESET stat={st3:.1f}  p={p3:.2e}")
    print(f"    flood_ord: beta={res_tf.params['flood_ord']:+.4f} "
          f"(p={res_tf.pvalues['flood_ord']:.3f});  r99pday: "
          f"beta={res_tf.params['r99pday']:+.4f} (p={res_tf.pvalues['r99pday']:.3f})")

    print("\nConclusione sez.1: se il RESET resta rigettato in (a),(b),(e) mentre i")
    print("coefficienti di rischio restano invarianti e le forme flessibili sul")
    print("rischio (c),(d) non mostrano pattern, il misfit di forma riguarda la")
    print("struttura edonica dei controlli, non la relazione rischio-prezzo.")

    # ------------------------------------------------------------------ 2
    sec("2. EFFETTI ETEROGENEI PER TIPOLOGIA (S5) — gruppi: Res/Comm/Prod/Box")
    print("Celle per gruppo:", df["gruppo"].value_counts().to_dict())
    het_rows = []
    het_specs = [("A2", f_a2, "flood_ord"), ("A2", f_a2, "sub_sink"),
                 ("A1", f_a1, "r99pday"), ("A1", f_a1, "pga"),
                 ("A1", f_a1, "fire_score"), ("A1", f_a1, "is_coastal_zone")]
    for base, fml, v in het_specs:
        f_int = fml.replace(f"{v} + ", "").replace(f" + {v}", "")
        # interazione pura (niente main effect di gruppo: e' un'aggregazione di
        # property_type e sarebbe collineare con C(property_type) gia' incluso)
        f_int = f_int.replace(
            "log_price ~ ",
            f"log_price ~ {v} + {v}:C(gruppo, Treatment('Residenziale')) + ")
        r = fit(f_int, df)
        sd = df[v].std()
        effs = []
        for g in ["Residenziale", "Commerciale", "Produttivo", "Box"]:
            if g == "Residenziale":
                contrast = v
            else:
                contrast = f"{v} + {v}:C(gruppo, Treatment('Residenziale'))[T.{g}]"
            tt = r.t_test(contrast)
            b = float(np.asarray(tt.effect).ravel()[0])
            se = float(np.asarray(tt.sd).ravel()[0])
            p = float(np.asarray(tt.pvalue).ravel()[0])
            effs.append({"spec": base, "rischio": v, "gruppo": g, "beta": b, "se": se,
                         "p_grezzo": p, "pct_per_1sd": 100*(np.exp(b*sd)-1)})
        ph = multipletests([e["p_grezzo"] for e in effs], method="holm")[1]
        for e, p_ in zip(effs, ph):
            e["p_holm"] = p_
        het_rows += effs
        inter_terms = [t for t in r.params.index if f"{v}:C(gruppo" in t]
        wt = r.wald_test(", ".join(f"{t} = 0" for t in inter_terms), scalar=True)
        print(f"\n{base}, {v}: test congiunto interazioni = 0: "
              f"stat={float(wt.statistic):.2f}  p={float(wt.pvalue):.4f}")
        for e in effs:
            print(f"    {e['gruppo']:<13} pct/1sd={e['pct_per_1sd']:+7.2f}%  "
                  f"p={e['p_grezzo']:.4f}  p_holm={e['p_holm']:.4f}")
    pd.DataFrame(het_rows).round(5).to_csv(
        os.path.join(EST, "e3_interazioni_tipologia.csv"), index=False)

    # ------------------------------------------------------------------ 3
    sec("3. MACHINE LEARNING SINTETICO (S15) — GroupKFold per comune")
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.inspection import permutation_importance, partial_dependence

    feat_ctrl = ["dist_cbd_km", "is_urban"]
    X_all = pd.get_dummies(
        df[RISKS + feat_ctrl + ["property_type", "condition", "fascia", "provincia"]],
        columns=["property_type", "condition", "fascia", "provincia"], drop_first=True)
    X_norisk = X_all.drop(columns=RISKS)
    yv = df["log_price"].values
    groups = df["comune"].values
    gkf = GroupKFold(n_splits=5)

    hgb = HistGradientBoostingRegressor(random_state=0, max_iter=400,
                                        learning_rate=0.06, max_depth=None)
    rf = RandomForestRegressor(n_estimators=400, random_state=0, n_jobs=-1,
                               min_samples_leaf=3)
    scores = {}
    for nm, mdl, X_ in [("HGB con rischi", hgb, X_all), ("HGB senza rischi", hgb, X_norisk),
                        ("RF con rischi", rf, X_all), ("RF senza rischi", rf, X_norisk)]:
        cv = cross_val_score(mdl, X_, yv, groups=groups, cv=gkf, scoring="r2")
        scores[nm] = (cv.mean(), cv.std())
        print(f"{nm:<18} R2 out-of-comune = {cv.mean():.3f} (+/- {cv.std():.3f})")
    d_hgb = scores["HGB con rischi"][0] - scores["HGB senza rischi"][0]
    d_rf = scores["RF con rischi"][0] - scores["RF senza rischi"][0]
    print(f"\nDelta R2 attribuibile ai 7 rischi: HGB={d_hgb:+.4f}  RF={d_rf:+.4f}")
    print(f"(riferimento: R2 out-of-comune del modello OLS A1 e' nell'ordine di ~0.8)")
    pd.DataFrame([{"modello": k, "r2_cv": v[0], "sd": v[1]} for k, v in scores.items()]
                 ).round(4).to_csv(os.path.join(EST, "e4_ml_r2.csv"), index=False)

    # permutation importance out-of-fold (HGB, solo rischi + controlli continui)
    imp_acc = pd.Series(0.0, index=X_all.columns)
    for tr, te in gkf.split(X_all, yv, groups):
        hgb.fit(X_all.iloc[tr], yv[tr])
        pi = permutation_importance(hgb, X_all.iloc[te], yv[te], n_repeats=5,
                                    random_state=0, n_jobs=-1)
        imp_acc += pd.Series(pi.importances_mean, index=X_all.columns)
    imp = (imp_acc / 5).sort_values(ascending=False)
    print("\nPermutation importance out-of-fold (top 12):")
    print(imp.head(12).round(4))
    imp.round(5).to_csv(os.path.join(EST, "e4_ml_importance.csv"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    imp_r = imp[[c for c in imp.index if c in RISKS]]
    ax.barh(imp_r.index[::-1], imp_r.values[::-1])
    ax.set_title("Permutation importance (out-of-comune) — variabili di rischio")
    fig.tight_layout(); fig.savefig(os.path.join(EST, "f8_ml_importance_rischi.png"), dpi=150)
    plt.close(fig)

    # PDP su modello full-sample (descrittivo) per flood_ord, r99pday, pga
    hgb.fit(X_all, yv)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, v in zip(axes, ["flood_ord", "r99pday", "pga"]):
        pd_res = partial_dependence(hgb, X_all, [list(X_all.columns).index(v)],
                                    grid_resolution=30)
        ax.plot(pd_res["grid_values"][0], pd_res["average"][0])
        ax.set_xlabel(v); ax.set_ylabel("PD log(prezzo)")
    fig.suptitle("Partial dependence (HGB full sample, descrittivo)")
    fig.tight_layout(); fig.savefig(os.path.join(EST, "f9_ml_pdp.png"), dpi=150)
    plt.close(fig)

    # mean|SHAP| per i rischi (RF, sottocampione per velocita')
    try:
        import shap
        rf.fit(X_all, yv)
        sub_idx = np.random.default_rng(0).choice(len(X_all), size=600, replace=False)
        expl = shap.TreeExplainer(rf)
        sv = expl.shap_values(X_all.iloc[sub_idx])
        msh = pd.Series(np.abs(sv).mean(axis=0), index=X_all.columns)
        msh_r = msh[[c for c in msh.index if c in RISKS]].sort_values(ascending=False)
        print("\nmean|SHAP| (RF, n=600) — variabili di rischio:")
        print(msh_r.round(4))
        msh.round(5).to_csv(os.path.join(EST, "e4_ml_shap.csv"))
    except Exception as e:
        print("SHAP saltato:", e)

    # ------------------------------------------------------------------ 4
    sec("4. REPRICING DINAMICO PRELIMINARE (S17) — 2 semestri, INDIZIO")
    dr = df[df["price_m2_prev"].notna() & (df["price_m2_prev"] > 0)].copy()
    dr["dlog"] = np.log(dr["price_m2"]) - np.log(dr["price_m2_prev"])
    f_d1 = f"dlog ~ {risk_str} + {C_A0} + {C_LOC} + C(provincia)"
    r_d1 = fit(f_d1, dr)
    f_d2 = f"dlog ~ {risk_str} + {C_A0} + {C_LOC} + C(comune)"
    r_d2 = fit(f_d2, dr)
    print("Delta log prezzo (semestre corrente vs precedente) ~ rischi:")
    rows_d = []
    for v in RISKS:
        sd = dr[v].std()
        rows_d.append({"variabile": v,
                       "beta_A1": r_d1.params[v], "p_A1": r_d1.pvalues[v],
                       "pp_per_sd_A1": 100 * r_d1.params[v] * sd,
                       "beta_A2": r_d2.params[v], "p_A2": r_d2.pvalues[v],
                       "pp_per_sd_A2": 100 * r_d2.params[v] * sd})
    td = pd.DataFrame(rows_d).set_index("variabile")
    print(td.round(4))
    td.round(5).to_csv(os.path.join(EST, "e5_repricing.csv"))
    print(f"N={int(r_d1.nobs)}; solo il ~26% delle celle varia tra i 2 semestri:")
    print("indizio preliminare, esplicitamente NON un test panel (S17, S0.1 L3).")

    # ------------------------------------------------------------------ 5
    sec("5. CANONI E RAPPORTO CANONE/PREZZO (S2) — discriminare le ipotesi")
    dnr = df[df["rental_m2"].notna() & (df["rental_m2"] > 0)].copy()
    dnr["log_rent"] = np.log(dnr["rental_m2"])
    dnr["log_yield"] = dnr["log_rent"] - dnr["log_price"]
    out_rows = []
    for dep, lab in [("log_rent", "log(canone)"), ("log_yield", "log(canone/prezzo)")]:
        for base, tail in [("A1", f"{C_A0} + {C_LOC} + C(provincia)"),
                           ("A2", f"{C_A0} + {C_LOC} + C(comune)")]:
            r = fit(f"{dep} ~ {risk_str} + {tail}", dnr)
            for v in RISKS:
                sd = dnr[v].std()
                out_rows.append({"dip": lab, "spec": base, "variabile": v,
                                 "beta": r.params[v], "p": r.pvalues[v],
                                 "pct_per_1sd": 100 * (np.exp(r.params[v] * sd) - 1)})
    tr_ = pd.DataFrame(out_rows)
    piv = tr_.pivot_table(index="variabile", columns=["dip", "spec"],
                          values=["pct_per_1sd", "p"])
    print(piv.round(3).to_string())
    tr_.round(5).to_csv(os.path.join(EST, "e6_canoni.csv"), index=False)
    print("\nLettura: se il rischio non compare ne' nei prezzi ne' nel rapporto")
    print("canone/prezzo, l'assenza e' uniforme nello strumento OMI (coerente con")
    print("non-prezzatura o con lisciamento amministrativo di entrambi); un rischio")
    print("nel rapporto canone/prezzo indicherebbe tassi di sconto differenziati.")

    pd.DataFrame(ext_rows).round(5).to_csv(
        os.path.join(EST, "e7_specs_estensioni.csv"), index=False)
    print("\n[OK] Tranche 3b completata. Output in output/estensioni/")
    sys.stdout = sys.__stdout__
    log.close()


if __name__ == "__main__":
    main()
