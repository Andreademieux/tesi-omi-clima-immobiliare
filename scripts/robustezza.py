#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tranche 3a - Analisi di robustezza (protocollo: S9, S10, S17, S17.1-S17.4).

Tutte le varianti sono richiamate per DELTA rispetto alle specifiche headline
A1/A2 di modelli_livello1.py (S7.1): stessa equazione salvo la modifica
indicata nel nome della variante.

Contenuti:
  1. Varianti di robustezza (S17): rr20mm al posto di r99pday; dummy/quote
     (flood_any, landslide_any) al posto delle ordinali; seismic_zone al posto
     di pga; subsidence_risk_class al posto di sub_sink; winsorizzazione log
     prezzo p1/p99; trim outlier prezzo p1/p99; pesi n_particelle (A1w);
     aggregazione territoriale alternativa comune x tipologia.
  2. Leave-one-provincia-out (S10) su A1 e A2.
  3. Wild cluster bootstrap a livello di PROVINCIA (4 cluster, pesi Webb,
     null imposto) per i rischi di A1 (S9).
  4. Correzione di Holm globale (S17.1) sulla famiglia primaria:
     7 rischi x 2 scale informative (A1, A2).
  5. Potenza statistica / MDE all'80% (S17.2).
  6. Test di falsificazione con rischio permutato (S17.3): r99pday permutato
     tra comuni (within provincia) in A1; flood_ord permutato tra zone
     (within comune) in A2. Via FWL esatto.
  7. Curva di specificazione (S17.4) sulle specifiche effettivamente stimate.

Output: output/robustezza/. Uso: py robustezza.py
"""
import os
import sys
import io

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")
ROB = os.path.join(OUT, "robustezza")
os.makedirs(ROB, exist_ok=True)

RISKS = ["flood_ord", "landslide_ord", "pga", "sub_sink", "fire_score",
         "r99pday", "is_coastal_zone"]
FAMILY = {"flood_ord": "flood", "flood_any": "flood",
          "landslide_ord": "frana", "landslide_any": "frana",
          "pga": "sisma", "seismic_zone": "sisma",
          "sub_sink": "subsidenza", "subsidence_risk_class": "subsidenza",
          "fire_score": "incendio", "r99pday": "precip", "rr20mm": "precip",
          "is_coastal_zone": "costa"}
C_A0 = "C(property_type) + C(condition)"
C_LOC = "C(fascia) + dist_cbd_km + is_urban"


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


def load_df():
    cell = pd.read_csv(os.path.join(OUT, "celle.csv"))
    cell["sub_sink"] = -cell["subsidence_velocity"]
    cell["log_price"] = np.log(cell["price_m2"])
    mv = (["log_price", "comune", "provincia", "zone", "property_type",
           "condition", "fascia", "dist_cbd_km", "is_urban", "n_particelle"]
          + RISKS)
    df = cell.dropna(subset=mv).copy().reset_index(drop=True)
    df["zone_id"] = df["provincia"] + "|" + df["comune"] + "|" + df["zone"].astype(str)
    return df


def fit(formula, d, w=None):
    m = smf.wls(formula, data=d, weights=w) if w is not None else smf.ols(formula, data=d)
    return m.fit(cov_type="cluster", cov_kwds={"groups": d["comune"]})


def extract(res, d, spec, delta):
    rows = []
    ci = res.conf_int()
    for v in res.params.index:
        if v in FAMILY:
            b = res.params[v]
            sd = d[v].std()
            rows.append({"spec": spec, "delta_vs_headline": delta,
                         "famiglia": FAMILY[v], "variabile": v, "beta": b,
                         "se": res.bse[v], "ci95_low": ci.loc[v, 0],
                         "ci95_high": ci.loc[v, 1], "p_grezzo": res.pvalues[v],
                         "sd_var": sd,
                         "pct_per_1sd": 100 * (np.exp(b * sd) - 1),
                         "pct_ci_low": 100 * (np.exp(ci.loc[v, 0] * sd) - 1),
                         "pct_ci_high": 100 * (np.exp(ci.loc[v, 1] * sd) - 1),
                         "N": int(res.nobs)})
    return rows


def cluster_meat_se(X, e, gid_codes, n_groups):
    """SE cluster-robust: meat = S'S con S = somme di gruppo di X*e."""
    Xe = X * e[:, None]
    S = np.zeros((n_groups, X.shape[1]))
    np.add.at(S, gid_codes, Xe)
    meat = S.T @ S
    N, k = X.shape
    G = n_groups
    XtX_inv = np.linalg.pinv(X.T @ X)
    corr = G / (G - 1) * (N - 1) / (N - k)
    V = corr * XtX_inv @ meat @ XtX_inv
    return np.sqrt(np.diag(V))


def main():
    log = open(os.path.join(ROB, "robustezza_summary.txt"), "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log)
    rng = np.random.default_rng(2026)

    df = load_df()
    risk_str = " + ".join(RISKS)
    f_a1 = f"log_price ~ {risk_str} + {C_A0} + {C_LOC} + C(provincia)"
    f_a2 = f"log_price ~ {risk_str} + {C_A0} + {C_LOC} + C(comune)"
    res_a1 = fit(f_a1, df)
    res_a2 = fit(f_a2, df)

    all_rows = []
    all_rows += extract(res_a1, df, "A1 (headline)", "-")
    all_rows += extract(res_a2, df, "A2 (headline)", "-")
    res_a0 = fit(f"log_price ~ {risk_str} + {C_A0}", df)
    all_rows += extract(res_a0, df, "A0 (headline)", "-")
    all_rows += extract(fit(f_a2, df, w=df["n_particelle"]), df, "A2w", "pesi=n_particelle")

    # ------------------------------------------------------------------ 1
    sec("1. VARIANTI DI ROBUSTEZZA (S17) — delta rispetto alla headline")

    def variant(name, formula, d=None, w=None, base="A1"):
        d = df if d is None else d
        r = fit(formula, d, w)
        rows = extract(r, d, name, name.split("— ", 1)[-1])
        all_rows.extend(rows)
        return r

    # rr20mm al posto di r99pday (S4.2, S17)
    f = f_a1.replace("r99pday", "rr20mm")
    variant("A1 — precip come rr20mm", f)
    f = f_a2.replace("r99pday", "rr20mm")
    variant("A2 — precip come rr20mm", f)
    # dummy/quote al posto delle ordinali
    f = f_a1.replace("flood_ord", "flood_any").replace("landslide_ord", "landslide_any")
    variant("A1 — flood/frana come quote any", f)
    f = f_a2.replace("flood_ord", "flood_any").replace("landslide_ord", "landslide_any")
    variant("A2 — flood/frana come quote any", f)
    # seismic_zone al posto di pga (nota: zona 1=piu' pericolosa)
    f = f_a1.replace("pga", "seismic_zone")
    variant("A1 — sisma come seismic_zone", f, d=df.dropna(subset=["seismic_zone"]))
    # subsidence_risk_class al posto di sub_sink
    f = f_a2.replace("sub_sink", "subsidence_risk_class")
    variant("A2 — subsidenza come risk_class", f,
            d=df.dropna(subset=["subsidence_risk_class"]))
    # winsorizzazione p1/p99 del log prezzo
    lo, hi = df["log_price"].quantile([.01, .99])
    dw = df.copy(); dw["log_price"] = dw["log_price"].clip(lo, hi)
    variant("A1 — log prezzo winsorizzato p1/p99", f_a1, d=dw)
    variant("A2 — log prezzo winsorizzato p1/p99", f_a2, d=dw)
    # trim outlier p1/p99
    dt = df[(df["log_price"] >= lo) & (df["log_price"] <= hi)].copy()
    variant("A1 — esclusi outlier prezzo p1/p99", f_a1, d=dt)
    variant("A2 — esclusi outlier prezzo p1/p99", f_a2, d=dt)
    # A1 pesata
    variant("A1w — pesi=n_particelle", f_a1, w=df["n_particelle"])
    # aggregazione alternativa comune x tipologia
    agg = (df.groupby(["provincia", "comune", "property_type"], as_index=False)
             .agg(log_price=("log_price", "mean"), dist_cbd_km=("dist_cbd_km", "mean"),
                  is_urban=("is_urban", "mean"),
                  **{v: (v, "mean") for v in RISKS}))
    f_g = (f"log_price ~ {risk_str} + C(property_type) + dist_cbd_km + is_urban "
           f"+ C(provincia)")
    r_g = fit(f_g, agg)
    all_rows += extract(r_g, agg, "AGG — comune x tipologia (come A1 senza fascia/condizione)",
                        "aggregazione comune x tipologia")
    print(f"Varianti stimate: {len(set(r['spec'] for r in all_rows))} specifiche totali")

    # ------------------------------------------------------------------ 2
    sec("2. LEAVE-ONE-PROVINCIA-OUT (S10)")
    lopo_rows = []
    for prov in ["Milano", "Napoli", "Roma", "Torino"]:
        d = df[df["provincia"] != prov]
        for base, fml in [("A1", f_a1), ("A2", f_a2)]:
            r = fit(fml, d)
            rows = extract(r, d, f"{base} — senza {prov}", f"esclusa {prov}")
            all_rows += rows
            lopo_rows += rows
    lopo = pd.DataFrame(lopo_rows)
    piv = lopo[lopo["spec"].str.startswith("A1")].pivot_table(
        index="variabile", columns="spec", values="pct_per_1sd")
    head_a1 = {r["variabile"]: r["pct_per_1sd"] for r in all_rows if r["spec"] == "A1 (headline)"}
    piv.insert(0, "A1 headline", pd.Series(head_a1))
    print("Effetto % per +1sd, A1 headline vs leave-one-provincia-out:")
    print(piv.round(2))
    piv.round(3).to_csv(os.path.join(ROB, "r2_lopo_a1.csv"))
    piv2 = lopo[lopo["spec"].str.startswith("A2")].pivot_table(
        index="variabile", columns="spec", values="pct_per_1sd")
    head_a2 = {r["variabile"]: r["pct_per_1sd"] for r in all_rows if r["spec"] == "A2 (headline)"}
    piv2.insert(0, "A2 headline", pd.Series(head_a2))
    print("\nEffetto % per +1sd, A2 headline vs leave-one-provincia-out:")
    print(piv2.round(2))
    piv2.round(3).to_csv(os.path.join(ROB, "r2_lopo_a2.csv"))

    # ------------------------------------------------------------------ 3
    sec("3. WILD CLUSTER BOOTSTRAP A LIVELLO PROVINCIA (S9, pesi Webb, B=4999)")
    y1, X1 = patsy.dmatrices(f_a1, df, return_type="dataframe")
    Xm = X1.values
    yv = y1.values.ravel()
    cols = list(X1.columns)
    prov_codes, prov_uniq = pd.factorize(df["provincia"])
    G = len(prov_uniq)
    webb = np.array([-np.sqrt(1.5), -1, -np.sqrt(0.5), np.sqrt(0.5), 1, np.sqrt(1.5)])
    B = 4999
    wcb_rows = []
    for v in RISKS:
        j = cols.index(v)
        # t osservato con SE cluster-provincia
        Pfull = np.linalg.pinv(Xm)
        beta = Pfull @ yv
        e = yv - Xm @ beta
        se_p = cluster_meat_se(Xm, e, prov_codes, G)
        t_obs = beta[j] / se_p[j]
        # modello ristretto (null b_j = 0)
        Xr = np.delete(Xm, j, axis=1)
        Pr = np.linalg.pinv(Xr)
        br = Pr @ yv
        fr = Xr @ br
        er = yv - fr
        cnt = 0
        for b in range(B):
            wgt = webb[rng.integers(0, 6, G)][prov_codes]
            ys = fr + er * wgt
            bs = Pfull @ ys
            es = ys - Xm @ bs
            ses = cluster_meat_se(Xm, es, prov_codes, G)
            ts = bs[j] / ses[j] if ses[j] > 0 else 0.0
            if abs(ts) >= abs(t_obs):
                cnt += 1
        p_wcb = (cnt + 1) / (B + 1)
        wcb_rows.append({"variabile": v, "beta": beta[j], "t_cluster_prov": t_obs,
                         "p_wcb_webb": p_wcb})
    wcb = pd.DataFrame(wcb_rows).set_index("variabile")
    print("A1, inferenza a livello di provincia (4 cluster):")
    print(wcb.round(4))
    wcb.round(5).to_csv(os.path.join(ROB, "r3_wcb_provincia.csv"))
    print("\nNota S9: il clustering primario resta il comune (345 cluster); il")
    print("bootstrap qui sopra verifica la sensibilita' dell'inferenza al livello")
    print("di aggregazione piu' alto, dove i cluster sono solo 4.")

    # ------------------------------------------------------------------ 4
    sec("4. CORREZIONE DI HOLM GLOBALE (S17.1) — famiglia: 7 rischi x {A1, A2}")
    fam = []
    for spec, res in [("A1", res_a1), ("A2", res_a2)]:
        for v in RISKS:
            fam.append({"spec": spec, "variabile": v, "p_grezzo": res.pvalues[v]})
    famdf = pd.DataFrame(fam)
    famdf["p_holm_globale"] = multipletests(famdf["p_grezzo"], method="holm")[1]
    famdf["sopravvive_5pct"] = famdf["p_holm_globale"] < 0.05
    famdf = famdf.sort_values("p_grezzo")
    print(famdf.round(4).to_string(index=False))
    famdf.round(5).to_csv(os.path.join(ROB, "r4_holm_globale.csv"), index=False)

    # ------------------------------------------------------------------ 5
    sec("5. POTENZA / MINIMUM DETECTABLE EFFECT (S17.2, potenza 80%, alfa 5%)")
    mde_rows = []
    for spec, res in [("A1", res_a1), ("A2", res_a2)]:
        for v in RISKS:
            sd = df[v].std()
            mde_b = 2.8 * res.bse[v]
            mde_pct = 100 * (np.exp(mde_b * sd) - 1)
            sig = res.pvalues[v] < 0.05
            if sig:
                verdetto = "significativo"
            elif mde_pct <= 5.0:
                verdetto = "nullo informativo (MDE <= 5%/sd)"
            else:
                verdetto = "bassa potenza (MDE > 5%/sd)"
            mde_rows.append({"spec": spec, "variabile": v, "se": res.bse[v],
                             "MDE_beta": mde_b, "MDE_pct_per_1sd": mde_pct,
                             "p_grezzo": res.pvalues[v], "verdetto": verdetto})
    mde = pd.DataFrame(mde_rows)
    print(mde.round(3).to_string(index=False))
    mde.round(4).to_csv(os.path.join(ROB, "r5_mde.csv"), index=False)

    # ------------------------------------------------------------------ 6
    sec("6. TEST DI FALSIFICAZIONE / PLACEBO (S17.3, 999 permutazioni, FWL)")

    def fwl_placebo(formula, var, d, perm_fn, n_perm=999):
        yv_, X_ = patsy.dmatrices(formula, d, return_type="dataframe")
        yv_ = yv_.values.ravel()
        cols_ = list(X_.columns)
        j = cols_.index(var)
        Xo = np.delete(X_.values, j, axis=1)
        Po = np.linalg.pinv(Xo)
        y_perp = yv_ - Xo @ (Po @ yv_)
        gid, uq = pd.factorize(d["comune"])
        Gn = len(uq)
        x = X_.values[:, j]

        def tstat(xv):
            x_perp = xv - Xo @ (Po @ xv)
            denom = x_perp @ x_perp
            bj = (x_perp @ y_perp) / denom
            e_ = y_perp - bj * x_perp
            Sg = np.zeros(Gn)
            np.add.at(Sg, gid, x_perp * e_)
            N_, k_ = X_.shape
            corr = Gn / (Gn - 1) * (N_ - 1) / (N_ - k_)
            se_ = np.sqrt(corr * (Sg @ Sg)) / denom
            return bj / se_

        t_obs_ = tstat(x)
        t_perm = np.array([tstat(perm_fn(d)) for _ in range(n_perm)])
        rej = np.mean(np.abs(t_perm) > 1.96)
        p_emp = (np.sum(np.abs(t_perm) >= abs(t_obs_)) + 1) / (n_perm + 1)
        return t_obs_, rej, p_emp

    # (a) r99pday permutato tra comuni within provincia (A1)
    com_mean = df.groupby("comune")["r99pday"].mean()
    com_prov = df.drop_duplicates("comune").set_index("comune")["provincia"]

    def perm_r99(d):
        new_map = {}
        for prov in com_prov.unique():
            cs = com_prov[com_prov == prov].index.to_numpy()
            vals = com_mean.loc[cs].to_numpy()
            new_map.update(dict(zip(cs, rng.permutation(vals))))
        return d["comune"].map(new_map).to_numpy()

    t_o, rej, p_emp = fwl_placebo(f_a1, "r99pday", df, perm_r99)
    print(f"(a) A1, r99pday permutato tra comuni (within provincia):")
    print(f"    t osservato={t_o:.2f}; tasso di rigetto placebo al 5% "
          f"(atteso ~5%): {100*rej:.1f}%;  p empirico del coeff. osservato: {p_emp:.4f}")

    # (b) flood_ord permutato tra zone within comune (A2)
    zmean = df.groupby("zone_id")["flood_ord"].mean()
    zcom = df.drop_duplicates("zone_id").set_index("zone_id")["comune"]

    def perm_flood(d):
        new_map = {}
        for com in zcom.unique():
            zs = zcom[zcom == com].index.to_numpy()
            vals = zmean.loc[zs].to_numpy()
            new_map.update(dict(zip(zs, rng.permutation(vals))))
        return d["zone_id"].map(new_map).to_numpy()

    t_o2, rej2, p_emp2 = fwl_placebo(f_a2, "flood_ord", df, perm_flood)
    print(f"(b) A2, flood_ord permutato tra zone (within comune):")
    print(f"    t osservato={t_o2:.2f}; tasso di rigetto placebo al 5%: "
          f"{100*rej2:.1f}%;  p empirico: {p_emp2:.4f}")
    with open(os.path.join(ROB, "r6_placebo.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"r99pday A1: t_obs={t_o:.3f} rejrate={rej:.4f} p_emp={p_emp:.4f}\n"
                 f"flood_ord A2: t_obs={t_o2:.3f} rejrate={rej2:.4f} p_emp={p_emp2:.4f}\n")

    # ------------------------------------------------------------------ 7
    sec("7. CURVA DI SPECIFICAZIONE (S17.4)")
    curve = pd.DataFrame(all_rows)
    curve.round(5).to_csv(os.path.join(ROB, "r7_curva_specificazione.csv"), index=False)
    print(f"Specifiche raccolte: {curve['spec'].nunique()}  "
          f"(headline + varianti S17 + LOPO; le estensioni attivate di Tranche 3b")
    print("si aggiungono in seguito). Riepilogo per famiglia (pct per +1sd):")
    summ = curve.groupby("famiglia")["pct_per_1sd"].agg(["count", "min", "median", "max"])
    print(summ.round(2))

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=False)
    for ax, famz, ttl in [(axes[0], "flood", "Alluvione"), (axes[1], "precip", "Precipitazioni estreme")]:
        sub = curve[curve["famiglia"] == famz].sort_values("pct_per_1sd").reset_index(drop=True)
        colors = ["crimson" if "headline" in s else "steelblue" for s in sub["spec"]]
        ax.hlines(range(len(sub)), sub["pct_ci_low"], sub["pct_ci_high"],
                  color=colors, alpha=0.6)
        ax.scatter(sub["pct_per_1sd"], range(len(sub)), c=colors, s=25, zorder=3)
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub["spec"], fontsize=6)
        ax.set_xlabel("% per +1 sd (IC95)")
        ax.set_title(f"{ttl} — curva di specificazione")
    fig.tight_layout()
    fig.savefig(os.path.join(ROB, "f6_curva_specificazione.png"), dpi=150)
    plt.close(fig)
    print("Figura: f6_curva_specificazione.png (headline in rosso)")

    print("\n[OK] Tranche 3a completata. Output in output/robustezza/")
    sys.stdout = sys.__stdout__
    log.close()


if __name__ == "__main__":
    main()
