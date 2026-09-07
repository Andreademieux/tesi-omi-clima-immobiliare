#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Versione PRUNED del report finale, orientata al metodo e alla discussione orale.

Scopo: rendere immediatamente chiaro a chi legge (e a chi ascolta) IN CHE MODO
l'analisi e' stata condotta. Rispetto a output/report_finale.html sono rimossi la
rassegna di letteratura, la discussione interpretativa estesa e le sezioni
programmatiche; ogni passaggio operativo e' invece accompagnato da una
spiegazione in linguaggio accessibile ("In parole semplici"), pensata per essere
esposta a voce, piu' un glossario e un'appendice di domande prevedibili.

Legge gli stessi output di output/eda, output/modelli, output/robustezza,
output/estensioni (piu' output/celle.csv per due verifiche calcolate qui) e
genera: output/report_metodo.html

Uso:  py report_metodo.py
"""
import base64
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")
EDA = os.path.join(OUT, "eda")
MOD = os.path.join(OUT, "modelli")
ROB = os.path.join(OUT, "robustezza")
EST = os.path.join(OUT, "estensioni")
DEST = os.path.join(OUT, "report_metodo.html")

RISKS = ["flood_ord", "landslide_ord", "pga", "sub_sink", "fire_score",
         "r99pday", "is_coastal_zone"]
C_A0 = "C(property_type) + C(condition)"
C_LOC = "C(fascia) + dist_cbd_km + is_urban"

IT = dict(
    flood_ord="Alluvione (classe PGRA, ordinale)",
    landslide_ord="Frana (classe PAI, ordinale)",
    pga="Sisma (PGA)", sub_sink="Subsidenza (mm/anno, sprofond.)",
    fire_score="Incendio (EFFIS)", r99pday="Precip. estreme (r99pday)",
    is_coastal_zone="Erosione costiera (quota zona costiera)")

SCALA = {
    "flood_ord": "A2 — within informativa",
    "sub_sink": "A2 — within informativa",
    "landslide_ord": "A1 — within limitata",
    "r99pday": "A1 — coeff. A2 non interpretabile",
    "is_coastal_zone": "A1 — coeff. A2 non interpretabile",
    "fire_score": "A1 — coeff. A2 non interpretabile",
    "pga": "A1 — coeff. A2 non interpretabile",
}


def b64img(path, alt, cap):
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return (f'<figure><img src="data:image/png;base64,{data}" alt="{alt}">'
            f'<figcaption>{cap}</figcaption></figure>')


def tbl(df, nd=3, index=False, small=False):
    cls = "tab small" if small else "tab"
    return df.to_html(index=index, classes=cls, border=0, escape=False,
                      float_format=lambda x: f"{x:,.{nd}f}", na_rep="—")


def sem(testo):
    """Box di spiegazione in linguaggio accessibile."""
    return (f'<div class="semplice"><span class="lbl2">In parole semplici</span>'
            f'{testo}</div>')


# ====================================================================
# Carica risultati
# ====================================================================
m1 = pd.read_csv(os.path.join(MOD, "m1_coefficienti.csv"))
m2 = pd.read_csv(os.path.join(MOD, "m2_fit.csv"))
m3 = pd.read_csv(os.path.join(MOD, "m3_within_variance_share.csv"),
                 names=["variabile", "quota_within"], header=0)
m4 = pd.read_csv(os.path.join(MOD, "m4_vif.csv"), index_col=0)
m6 = pd.read_csv(os.path.join(MOD, "m6_decisioni_livello2.csv"))
r2a = pd.read_csv(os.path.join(ROB, "r2_lopo_a1.csv"))
r3 = pd.read_csv(os.path.join(ROB, "r3_wcb_provincia.csv"))
r4 = pd.read_csv(os.path.join(ROB, "r4_holm_globale.csv"))
r5 = pd.read_csv(os.path.join(ROB, "r5_mde.csv"))
r7 = pd.read_csv(os.path.join(ROB, "r7_curva_specificazione.csv"))
e2 = pd.read_csv(os.path.join(EST, "e2_classi_pericolosita.csv"))
e3 = pd.read_csv(os.path.join(EST, "e3_interazioni_tipologia.csv"))
e4r = pd.read_csv(os.path.join(EST, "e4_ml_r2.csv"))
e6 = pd.read_csv(os.path.join(EST, "e6_canoni.csv"))
t11 = pd.read_csv(os.path.join(EDA, "t11_fonti_rischio.csv"))
t3 = pd.read_csv(os.path.join(EDA, "t3_copertura_omi_provincia.csv"))

e6h = []
for (dip, spec), g in e6.groupby(["dip", "spec"]):
    g = g.copy()
    g["p_holm"] = multipletests(g["p"], method="holm")[1]
    e6h.append(g)
e6 = pd.concat(e6h)

# ====================================================================
# Verifiche calcolate qui: (a) effetto del clustering sugli SE,
# (b) granularita' realmente disponibile sotto il comune.
# ====================================================================
cell = pd.read_csv(os.path.join(OUT, "celle.csv"))
cell["sub_sink"] = -cell["subsidence_velocity"]
cell["log_price"] = np.log(cell["price_m2"])
_mv = (["log_price", "comune", "provincia", "zone", "property_type", "condition",
        "fascia", "dist_cbd_km", "is_urban", "n_particelle"] + RISKS)
dfc = cell.dropna(subset=_mv).copy().reset_index(drop=True)
dfc["zone_id"] = dfc["provincia"] + "|" + dfc["comune"] + "|" + dfc["zone"].astype(str)

_f_a1 = f"log_price ~ {' + '.join(RISKS)} + {C_A0} + {C_LOC} + C(provincia)"
_f_a2 = f"log_price ~ {' + '.join(RISKS)} + {C_A0} + {C_LOC} + C(comune)"
_m1o = smf.ols(_f_a1, data=dfc)
_a1_ols = _m1o.fit()
_a1_hc1 = _m1o.fit(cov_type="HC1")
_a1_cl = _m1o.fit(cov_type="cluster", cov_kwds={"groups": dfc["comune"]})
se_tab = pd.DataFrame([{
    "Rischio": IT[v],
    "SE classico": _a1_ols.bse[v], "SE HC1": _a1_hc1.bse[v],
    "SE cluster": _a1_cl.bse[v],
    "cluster / classico": _a1_cl.bse[v] / _a1_ols.bse[v],
    "p classico": _a1_ols.pvalues[v], "p cluster": _a1_cl.pvalues[v],
} for v in RISKS])

_multi = dfc.groupby("comune")["zone_id"].nunique()
_com_multi = _multi[_multi > 1].index
_dm = dfc[dfc["comune"].isin(_com_multi)]
_m2o = smf.ols(_f_a2, data=dfc)
_a2_hc1 = _m2o.fit(cov_type="HC1")
_a2_cl = _m2o.fit(cov_type="cluster", cov_kwds={"groups": dfc["comune"]})
gran = pd.DataFrame([{
    "Rischio": IT[v],
    "sd entro comune": _dm.groupby("comune")[v].transform(lambda s: s - s.mean()).std(),
    "sd complessiva": dfc[v].std(),
    "quota conservata": (_dm.groupby("comune")[v]
                         .transform(lambda s: s - s.mean()).std() / dfc[v].std()),
    "costo del clustering (A2)": _a2_cl.bse[v] / _a2_hc1.bse[v],
} for v in RISKS]).sort_values("quota conservata", ascending=False)
N_MULTI, N_CELLE_MULTI = len(_com_multi), len(_dm)

S = {}

# ====================================================================
CSS = """
:root { --ink:#1a1a1a; --acc:#0b5563; --line:#d8d8d8; --bg:#ffffff; --soft:#f4f6f7;
        --amb:#8a6d00; }
* { box-sizing:border-box; }
body { font-family: Georgia, 'Times New Roman', serif; color:var(--ink);
       background:var(--bg); margin:0; line-height:1.58; }
main { max-width: 980px; margin: 0 auto; padding: 2rem 1.4rem 5rem; }
h1 { font-size:1.7rem; color:var(--acc); line-height:1.25; margin:.2rem 0 .4rem;}
h2 { font-size:1.24rem; color:var(--acc); border-bottom:2px solid var(--acc);
     padding-bottom:.25rem; margin-top:2.8rem; }
h3 { font-size:1.03rem; margin-top:1.6rem; color:#333; }
h4 { font-size:.95rem; margin:1.1rem 0 .3rem; color:#444; }
p, li { font-size:.95rem; text-align:justify; }
.meta { color:#555; font-style:italic; margin-bottom:1.2rem; }
.abstract { background:var(--soft); border-left:4px solid var(--acc);
            padding:.9rem 1.1rem; }
.semplice { background:#fbf8f0; border-left:4px solid var(--amb);
            padding:.7rem 1rem; margin:.9rem 0; border-radius:0 4px 4px 0; }
.semplice p { margin:.4rem 0; }
.semplice p:first-of-type { margin-top:0; }
.semplice p:last-child { margin-bottom:0; }
.lbl2 { display:block; font-family:'Segoe UI', Arial, sans-serif; font-size:.68rem;
        letter-spacing:.09em; text-transform:uppercase; color:var(--amb);
        font-weight:700; margin-bottom:.35rem; }
.eq { font-family:'Cambria Math', Georgia, serif; background:var(--soft);
      padding:.65rem .9rem; margin:.7rem 0; border-radius:4px;
      overflow-x:auto; font-size:.92rem; }
.tab { border-collapse:collapse; margin:.8rem auto; font-size:.8rem;
       font-family: 'Segoe UI', Arial, sans-serif; }
.tab th { background:var(--acc); color:#fff; padding:.3rem .5rem;
          text-align:center; font-weight:600;}
.tab td { border-bottom:1px solid var(--line); padding:.28rem .5rem;
          text-align:right; vertical-align:top; }
.tab td:first-child, .tab th:first-child { text-align:left; }
.tab.small { font-size:.72rem; }
.tabwrap { overflow-x:auto; }
figure { margin:1.1rem 0; text-align:center; }
figure img { max-width:100%; border:1px solid var(--line); }
figcaption { font-size:.8rem; color:#555; margin-top:.35rem; }
.box { border:1px solid var(--acc); border-radius:6px; padding:.8rem 1.1rem;
       margin:1rem 0; background:#f7fbfc; }
.warn { border-left:4px solid #a33; background:#fdf6f6; padding:.7rem 1rem;
        margin:.8rem 0; font-size:.92rem;}
.card { border:1px solid var(--line); border-left:4px solid var(--acc);
        border-radius:4px; padding:.6rem .9rem; margin:.8rem 0;
        background:#fbfcfc; font-size:.92rem; }
.card b.lbl { color:var(--acc); }
.qa { border-bottom:1px solid var(--line); padding:.6rem 0; }
.qa .q { font-weight:700; color:var(--acc); display:block; margin-bottom:.2rem; }
.qa p { margin:.2rem 0; }
dl.gloss dt { font-weight:700; color:var(--acc); margin-top:.6rem; font-size:.93rem;
              font-family:'Segoe UI', Arial, sans-serif; }
dl.gloss dd { margin:.1rem 0 0 0; font-size:.93rem; }
code { background:var(--soft); font-size:.85em; padding:.05rem .3rem; }
.flow { display:flex; flex-wrap:wrap; align-items:center; gap:.35rem;
        margin:1rem 0; }
.flow .step { flex:1 1 130px; border:1px solid var(--acc); border-radius:6px;
        padding:.45rem .4rem; text-align:center; font-size:.76rem;
        background:#f7fbfc; font-family:'Segoe UI', Arial, sans-serif; }
.flow .step b { display:block; font-size:.98rem; color:var(--acc); }
.flow .arr { color:var(--acc); font-size:1.1rem; }
.toc { background:var(--soft); padding:.8rem 1.2rem; border-radius:6px;
       font-size:.88rem; column-count:2; }
.toc a { color:var(--acc); text-decoration:none; }
ol.spine > li { margin-bottom:.5rem; }
.ok { color:#0a6b31; font-weight:700; }
.no { color:#8a8a8a; font-weight:700; }
@media (max-width:700px){ .toc{column-count:1;} .flow .step{flex:1 1 100%;} }
@media print { h2 { page-break-after:avoid; } figure { page-break-inside:avoid; }
               .semplice { page-break-inside:avoid; } }
"""

# ====================================================================
# 0. NOTA DI LETTURA + FILO DEL DISCORSO
# ====================================================================
S["s0"] = """
<h2 id="s0">0. Che cosa è questo documento e come si legge</h2>
<div class="abstract">
<p>Questa è la <strong>versione ridotta e orientata al metodo</strong> del report finale.
Il suo scopo è rendere verificabile — e raccontabile a voce — <em>come</em> l'analisi è stata
condotta: quale dato entra, con quale trasformazione, in quale equazione, con quale regola di
decisione, con quale controllo di robustezza e con quale criterio di lettura del risultato.</p>
<p><strong>Cosa è stato tolto</strong> rispetto a <code>output/report_finale.html</code>: la
rassegna di letteratura, la discussione economica estesa, le implicazioni per l'indice
sintetico, le conclusioni discorsive. <strong>Cosa è stato aggiunto</strong>: per ogni
passaggio tecnico, una spiegazione in linguaggio accessibile, riconoscibile dal riquadro
color ambra. I due livelli sono complementari: il testo principale è quello che va scritto in
tesi, il riquadro è quello che va detto a voce se qualcuno chiede «e questo che cosa
significa?».</p>
</div>
<h3>Il filo del discorso in dieci passaggi</h3>
<p>Questa è la spina dorsale dell'intera analisi. Ogni passaggio corrisponde a una sezione
del documento, e i dieci punti insieme costituiscono il racconto completo dello studio.</p>
<ol class="spine">
<li><strong>La domanda.</strong> Le quotazioni immobiliari <em>ufficiali</em> di Milano,
Napoli, Roma e Torino riflettono il rischio climatico e ambientale del territorio? (§1)</li>
<li><strong>Il dato di prezzo non è un prezzo di mercato</strong>, ma una stima
amministrativa riferita a un'intera zona: questo decide che cosa è possibile misurare e che
cosa no. (§1)</li>
<li><strong>Poiché il prezzo è unico per zona, le singole particelle non sono osservazioni
indipendenti</strong>: si costruisce un'unità di analisi aggregata, la «cella». (§3)</li>
<li><strong>Sette famiglie di rischio</strong>, prese nella forma più grezza disponibile,
ciascuna con fonte, risoluzione e regola di attribuzione dichiarate. (§2)</li>
<li><strong>Si stima una regressione del logaritmo del prezzo</strong> aggiungendo controlli
geografici via via più stringenti: nessuno, poi la provincia, poi il comune. (§4)</li>
<li><strong>Prima di leggere i coefficienti si misura quanta informazione resta</strong> a
ciascuna scala: alcuni rischi, dentro un comune, non variano affatto. (§5)</li>
<li><strong>L'incertezza si calcola tenendo conto</strong> che celle dello stesso comune non
sono informazioni indipendenti. (§4.3)</li>
<li><strong>La diagnostica decide</strong> quali tecniche aggiuntive attivare: due sì, le
altre no, ciascuna con il test che lo stabilisce. (§6, §8)</li>
<li><strong>Il risultato viene attaccato da ogni lato</strong>: 24 specifiche, esclusione di
una provincia per volta, bootstrap, test placebo, verifica con machine learning. (§9)</li>
<li><strong>La conclusione</strong>: nessuno sconto di prezzo robusto per il rischio; e per i
risultati nulli si dimostra che un effetto, se ci fosse stato, sarebbe stato visibile. (§10)</li>
</ol>
"""

# ====================================================================
# 1. OGGETTO MISURATO
# ====================================================================
t3f = t3.rename(columns={t3.columns[0]: "provincia"})
t3f["quota_con_omi"] = (100 * t3f["quota_con_omi"]).round(1)
t3f = t3f.rename(columns={"quota_con_omi": "% con quotazione"})
S["s1"] = f"""
<h2 id="s1">1. L'oggetto misurato: perché la domanda è vincolata dal dato</h2>
<p>La variabile dipendente è la <strong>quotazione OMI</strong> (Osservatorio del Mercato
Immobiliare, Agenzia delle Entrate): un intervallo min–max €/m² pubblicato semestralmente
per <em>zona OMI × tipologia edilizia × stato conservativo</em>, costruito con criteri
amministrativi a partire da fonti miste (atti, perizie, expert opinion). Nel modello si usa
il <strong>punto medio</strong> dell'intervallo di acquisto.</p>
{sem('''<p>Non stiamo osservando il prezzo a cui una singola casa è stata venduta. Stiamo
osservando il <em>valore di zona</em> che l'Agenzia delle Entrate pubblica due volte l'anno:
una tabella che dice, per esempio, «nella zona semicentrale di questo comune, un'abitazione
civile in stato normale vale tra 1.200 e 1.500 €/m²».</p>
<p>È la differenza tra il prezzo scritto sul rogito del tuo appartamento e il valore medio
scritto in un listino ufficiale. Il secondo è quello che usano banche, periti e uffici
tecnici — quindi è un oggetto importante da studiare — ma è una <em>stima</em>, non una
transazione. Tutto ciò che segue discende da questa distinzione.</p>''')}
<div class="warn"><p>Tre proprietà del dato vincolano il metodo a valle e vanno tenute
presenti leggendo qualunque coefficiente:</p>
<ul>
<li><strong>Granularità di zona</strong>: il prezzo è attribuito alla zona, non
all'immobile. Qualunque variazione di rischio <em>interna</em> alla zona è, per
costruzione, non prezzabile nel dato. È ciò che rende necessaria l'unità statistica
aggregata di §3.</li>
<li><strong>Lisciamento temporale</strong>: tra i due semestri disponibili il prezzo varia
solo nel <strong>26,2%</strong> delle celle (Δlog medio +0,6%). Da qui l'esclusione di
qualunque disegno panel o difference-in-differences (§6.2) e il rango di semplice indizio
dato all'analisi di repricing (§9.9).</li>
<li><strong>Natura amministrativa</strong>: la stima risponde alla domanda «<em>lo strumento
ufficiale riflette il rischio?</em>» e solo indirettamente a «<em>il mercato prezza il
rischio?</em>». La verifica costruita per distinguerle è in §9.5.</li>
</ul></div>
{sem('''<p>Un esempio concreto della prima proprietà: se dentro una stessa zona OMI ci sono
due palazzine, una a ridosso del torrente e una in collina, il dato OMI attribuisce a
entrambe lo stesso valore al metro quadro. Anche se il mercato le prezzasse in modo diverso,
noi non potremmo vederlo: nel dato quella differenza semplicemente non esiste.</p>
<p>Questo non invalida lo studio, ma ne definisce l'oggetto: stiamo verificando se lo
<em>strumento di valutazione ufficiale</em> incorpora il rischio, che è esattamente la
domanda rilevante per chi usa l'OMI per valutare una garanzia immobiliare o un
patrimonio.</p>''')}
<p>Il perimetro è costituito dalle quattro province di Milano, Napoli, Roma e Torino;
la copertura OMI del campione di particelle è la seguente:</p>
{tbl(t3f, nd=1)}
<p>La copertura è guidata dall'urbanizzazione (92,7% di particelle quotate in land cover
artificiale contro 25,2% altrove). Ne discende un vincolo di validità esterna dichiarato,
non un bias per la domanda posta: <strong>il campione rappresenta il perimetro urbano
quotato</strong>, non l'intero territorio provinciale.</p>
{sem('''<p>L'OMI non copre tutto il territorio: pubblica quotazioni dove c'è un mercato
immobiliare osservabile, cioè in prevalenza dove si costruisce. Su 100 particelle estratte a
caso, circa 39 hanno una quotazione; ma la quota sale a 93 su 100 se la particella è in area
urbanizzata.</p>
<p>Conseguenza da dichiarare, non da nascondere: i risultati valgono per il territorio
quotato — cioè città e centri abitati — e non possono essere estesi alle campagne e alle
aree extraurbane, che sono anche leggermente più esposte alle precipitazioni estreme.</p>''')}
"""

# ====================================================================
# 2. VARIABILI DI RISCHIO
# ====================================================================
sd_map = m1[m1["spec"] == "A1"].set_index("rischio")["sd_var"].to_dict()
def_rows = [
    ("Alluvione", "flood_ord",
     "Classe PGRA del punto particella (punto-in-poligono): LPH=1, MPH=2, HPH=3; "
     "0 se fuori da ogni perimetro.",
     "Media delle classi sulle particelle della cella (0–3)."),
    ("Frana", "landslide_ord",
     "Classe PAI del punto particella: P1=1 … P4=4; 0 se fuori perimetro. Le aree di "
     "attenzione «AA» sono <em>escluse</em> dall'ordinale (natura amministrativa diversa).",
     "Media delle classi sulle particelle della cella (0–4)."),
    ("Sisma", "pga",
     "Peak Ground Acceleration (g) da griglia INGV MPS04, interpolata sul punto.",
     "Media di cella (g). Variante di robustezza: zona sismica comunale 1–4."),
    ("Subsidenza", "sub_sink",
     "Velocità verticale interferometrica (mm/anno) con <strong>segno invertito</strong>, "
     "così che valori alti = sprofondamento.",
     "Media di cella (mm/anno). Variante: classe di rischio subsidenza."),
    ("Incendio", "fire_score",
     "Score EFFIS Wildfire Risk Assessment (rank per terzili, 0–1) della cella ~12 km che "
     "contiene la particella; se la cella non è bruciabile, si usa la cella bruciabile più "
     "vicina (5,5% delle particelle, distanza media 13,3 km).",
     "Media di cella (0–1)."),
    ("Precipitazioni estreme", "r99pday",
     "Giorni/anno oltre il 99° percentile <em>locale</em> dei giorni piovosi, climatologia "
     "E-OBS 1995–2019 su griglia 0,1° (~11 km), attribuiti per nearest-neighbor valido "
     "(distanza media 3,8 km).",
     "Media di cella (giorni/anno). Variante a soglia assoluta: rr20mm (giorni ≥20 mm)."),
    ("Erosione costiera", "is_coastal_zone",
     "Indicatore 1/0 di appartenenza a zona costiera monitorata (esistenza di almeno un "
     "transetto di costa associato alla particella).",
     "Quota di particelle della cella in zona costiera (0–1)."),
]
defs = pd.DataFrame([{
    "Rischio": a, "Variabile": f"<code>{b}</code>",
    "Costruzione alla particella": c, "Valore di cella": d,
    "sd nel campione": round(sd_map[b], 3)} for a, b, c, d in def_rows])

S["s2"] = f"""
<h2 id="s2">2. Le sette variabili di rischio: come un pericolo diventa un numero</h2>
<p>Il vettore di rischio è deliberatamente tenuto nelle <strong>forme più grezze e
disaggregate disponibili</strong> (classi ordinali, valori fisici), senza indici sintetici
costruiti a monte: l'obiettivo dello studio è verificare se un indice sia costruibile, e
partire da un indice sarebbe circolare.</p>
{sem('''<p>Questo è il punto da spiegare per primo, perché è quello meno intuitivo: come si
trasforma «questa zona è a rischio alluvione» in un numero da mettere in un'equazione.</p>
<p>Ci sono due modi, a seconda della fonte. <strong>Primo: i perimetri.</strong> I piani di
gestione del rischio disegnano sulla mappa delle aree — «qui l'alluvione è probabile», «qui
lo è meno». Si prende il punto che identifica la particella catastale e si guarda dentro
quale area cade: fuori da tutte = 0, area a bassa probabilità = 1, media = 2, alta = 3. È il
metodo usato per alluvione e frana.</p>
<p><strong>Secondo: le griglie.</strong> Per clima e incendi il dato non è un perimetro ma una
scacchiera che copre l'Europa, con caselle di circa 11–12 km di lato, ciascuna con il suo
valore. La particella eredita il valore della casella in cui si trova. È il metodo usato per
precipitazioni estreme e incendio, ed è anche il motivo per cui quei due dati sono più
grossolani degli altri (§11, punto 3).</p>''')}
<div class="tabwrap">{tbl(defs, nd=3, small=True)}</div>
{sem('''<p>L'ultima colonna serve per rendere confrontabili rischi misurati in unità
diversissime — grammi di accelerazione sismica, millimetri all'anno di sprofondamento,
giorni di pioggia. La deviazione standard è, in pratica, «quanto varia tipicamente questa
grandezza da una zona all'altra». Esprimere tutti i risultati «per +1 deviazione standard»
equivale a chiedersi: <em>passando da una zona normale a una zona sensibilmente più esposta,
di quanto cambia il prezzo?</em> È l'unità di misura comune che rende paragonabile
l'alluvione con il terremoto.</p>''')}
<p>Non sono usate come regressori: le aree di attenzione frana AA, la distanza e la variazione
della linea di costa (missing strutturale per il 95,7% delle celle, §3.4), e qualunque indice
climatico precalcolato.</p>
<h3>2.1 Fonti, vintage e regola di attribuzione geografica</h3>
<div class="tabwrap">{tbl(t11, index=False, small=True)}</div>
<p>Ogni variabile è quindi <em>tracciabile</em> a una fonte, a una data di scarico e a una
regola di attribuzione esplicita. Due attribuzioni introducono errore di misura noto e
dichiarato — incendio (~12 km, e 13,3 km medi per il 5,5% attribuito alla cella bruciabile
più vicina) e precipitazioni (~11 km, 3,8 km medi) — con la conseguenza analitica discussa
in §11: l'errore di misura attenua le stime verso zero, quindi qualifica i nulli.</p>
"""

# ====================================================================
# 3. DAL DATO GREZZO AL CAMPIONE
# ====================================================================
FLOW = """
<div class="flow">
  <div class="step"><b>65.800</b>particelle scaricate<br>(≈100 per comune, 658 comuni)</div>
  <div class="arr">→</div>
  <div class="step"><b>65.618</b>dopo pulizia<br>(−182 fuori provincia;<br>2.290 comuni riattribuiti)</div>
  <div class="arr">→</div>
  <div class="step"><b>25.468</b>con ≥1 quotazione OMI<br>(38,8%)</div>
  <div class="arr">→</div>
  <div class="step"><b>212.417</b>righe particella ×<br>voce OMI (pseudo-repliche)</div>
  <div class="arr">→</div>
  <div class="step"><b>2.793</b>celle<br>(unità statistica)</div>
  <div class="arr">→</div>
  <div class="step"><b>2.728</b>campione di stima<br>(97,7%; 345 comuni)</div>
</div>
"""
S["s3"] = f"""
<h2 id="s3">3. Dal dato grezzo al campione di stima: la catena completa</h2>
{FLOW}
{sem('''<p>Il diagramma si legge da sinistra a destra come un imbuto. Partiamo da 65.800
particelle catastali estratte dal database. Ne togliamo 182 che, controllando il comune
catastale vero, risultano fuori dalle quattro province. Di queste, poco meno di 4 su 10 hanno
una quotazione OMI.</p>
<p>Il salto importante è il quarto: quelle 25.468 particelle, incrociate con le diverse
tipologie edilizie presenti in zona, generano 212.417 righe — ma sono in gran parte
<em>ripetizioni della stessa informazione</em>, perché il prezzo è unico per zona. Comprimendo
le ripetizioni restano 2.793 unità realmente distinte, che diventano 2.728 escludendo quelle
con un dato mancante. È su queste 2.728 che si stimano tutti i modelli.</p>''')}
<h3>3.1 Pulizia e riattribuzione territoriale</h3>
<p>Il comune e la provincia non sono presi dal campo usato per interrogare l'API, ma
<strong>ricostruiti dal comune catastale</strong> (<code>municipality</code> del dettaglio
particella), che è la verità di base. La verifica ha rivelato un <em>boundary leakage</em>:
182 particelle raccolte come appartenenti a una delle quattro province risultano
catastalmente fuori (escluse) e 2.290 risultano in un comune diverso all'interno della
stessa provincia (riattribuite).</p>
{sem('''<p>Le particelle erano state scaricate interrogando il database comune per comune. Ma
il confine amministrativo usato per la ricerca e il confine catastale reale non coincidono
sempre: alcune particelle «assegnate» a un comune appartengono in realtà a quello
confinante.</p>
<p>Perché è importante e non è un dettaglio: tutto il cuore dell'analisi consiste nel
confrontare zone <em>dello stesso comune</em>. Se una zona è attribuita al comune sbagliato,
la confrontiamo con le zone sbagliate. È come correggere i compiti di una classe dopo aver
messo per errore nella pila alcuni compiti della classe accanto.</p>''')}
<h3>3.2 Perché l'unità statistica è la «cella» e non la particella</h3>
<p>Il prezzo OMI è unico per zona × tipologia × stato: usare le 25.468 particelle quotate
(o le 212.417 righe particella × voce) come osservazioni indipendenti replicherebbe lo
stesso prezzo decine di volte, gonfiando artificialmente la precisione. La verifica
empirica del problema è diretta: <strong>la deviazione standard del prezzo entro il
raggruppamento è esattamente 0 nel 99,1% dei casi</strong> (il residuo 0,9% deriva dalla
compresenza di più microzone nella stessa zona OMI; in quei casi il prezzo di cella è la
media). L'analisi primaria è quindi condotta sull'unità aggregata:</p>
<p class="eq">cella&nbsp;c&nbsp;=&nbsp;provincia × comune × zona OMI × tipologia ×
condizione × fascia&nbsp;&nbsp;(N = 2.793; campione di stima N = 2.728)</p>
{sem('''<p>Questo è il passaggio metodologico più importante del disegno, e va raccontato
così: se in una zona ci sono 70 particelle e l'OMI attribuisce a tutte lo stesso valore di
1.300 €/m², quelle 70 righe non sono 70 osservazioni — sono <strong>una sola informazione
scritta 70 volte</strong>.</p>
<p>Trattarle come 70 osservazioni indipendenti farebbe credere al modello di avere una massa
di dati che non ha, e ogni risultato apparirebbe molto più preciso e più significativo di
quanto sia. È il fenomeno che in letteratura si chiama <em>pseudo-replicazione</em>.</p>
<p>La soluzione è comprimere ogni gruppo di righe con lo stesso prezzo in una sola riga — la
«cella» — e assegnarle come valore di rischio la media delle particelle che la compongono.
Da 212.417 righe si passa così a 2.793 unità genuinamente distinte.</p>''')}
<p>Le variabili di rischio della cella sono <strong>medie sulle particelle sottostanti</strong>
(1–142 particelle per cella, mediana 70). L'aggregazione ha un effetto sostanziale, non solo
tecnico: trasforma indicatori binari o ordinali di particella in <em>quote di esposizione</em>
continue della zona, ed è ciò che rende leggibile il coefficiente come effetto di un aumento
di esposizione dell'area.</p>
<h3>3.3 Pesi: due stime, non una scelta arbitraria</h3>
<p>Le celle hanno numerosità molto diversa (5° percentile 2 particelle, mediana 70, massimo
142). Poiché l'osservazione rilevante è <em>la quotazione amministrativa</em> e non la
particella, la specifica non pesata è la headline; la stima pesata per numero di particelle
(A2w) è riportata in parallelo come confronto. I risultati sono quasi identici (§4.4), quindi
la scelta dei pesi non è un grado di libertà che orienta le conclusioni.</p>
{sem('''<p>Domanda naturale: una cella costruita su 140 particelle deve «pesare» più di una
costruita su 2? Ci sono due risposte difendibili, e invece di sceglierne una a tavolino
abbiamo stimato il modello in entrambi i modi. Poiché i risultati coincidono, la questione
non incide sulle conclusioni — ed è questo, non la scelta in sé, il fatto da riportare.</p>''')}
<h3>3.4 Dati mancanti: prima il meccanismo, poi il trattamento</h3>
<ul>
<li><strong>Missing strutturale</strong> — distanza e variazione della linea di costa
mancano per il 95,7% delle celle perché <em>non costiere</em>: non imputati; l'esposizione
costiera entra con la quota <code>is_coastal_zone</code>, che è definita ovunque.</li>
<li><strong>Missing tecnico raro</strong> — subsidenza (1,9%, 52 celle) e incendio (0,5%,
13 celle) → <em>listwise deletion</em>, che porta da 2.793 a 2.728 celle (97,7%). Sotto il 5%
di perdita il protocollo non richiede analisi di sensibilità all'imputazione.</li>
</ul>
{sem('''<p>Un dato mancante non è sempre lo stesso problema. La distanza dalla linea di costa
manca per il 96% delle celle non perché il dato sia andato perso, ma perché quelle celle
<em>non sono sul mare</em>: la domanda non ha senso. Inventare un valore (imputare) sarebbe
un errore. Si usa invece una variabile che è definita per tutti: «quanta parte di questa zona
è in area costiera monitorata», che vale zero nell'entroterra.</p>
<p>Diverso è il caso di subsidenza e incendio, dove il dato manca per motivi tecnici in poche
decine di celle: lì si eliminano semplicemente quelle righe, perdendo il 2,3% del campione.</p>''')}
<h3>3.5 Perché il logaritmo del prezzo</h3>
<p>La distribuzione in livelli è fortemente asimmetrica (skewness +2,60; mediana 1.085 €/m²,
p99 3.500, max 9.450 — Capri), mentre il logaritmo è quasi simmetrico (skewness −0,07,
curtosi in eccesso −0,32). La forma log-lineare è quindi giustificata <em>ex ante</em> dalla
distribuzione e, separatamente, sottoposta a test formale (RESET, §8) e a forme flessibili
alternative (§9.6).</p>
{sem('''<p>Due ragioni, entrambe semplici.</p>
<p><strong>La prima è la forma dei dati.</strong> I prezzi immobiliari hanno una coda lunga
verso l'alto: moltissime zone tra 500 e 1.500 €/m², poche zone a 5.000, e Capri a 9.450. Una
distribuzione così sbilanciata viola le condizioni in cui la regressione lineare funziona
bene. Prendendo il logaritmo la distribuzione diventa quasi simmetrica.</p>
<p><strong>La seconda è l'interpretazione.</strong> Con il prezzo in euro, un coefficiente
direbbe «questa zona costa 80 €/m² in meno»: ma 80 € su 500 sono tantissimo, su 5.000 sono
niente. Con il logaritmo il coefficiente si legge direttamente <em>in percentuale</em>: «questa
zona costa il 5% in meno», che è un'affermazione confrontabile tra mercati diversi.</p>''')}
"""

# ====================================================================
# 4. EQUAZIONI
# ====================================================================
fit_show = m2.rename(columns={m2.columns[0]: "spec"})
S["s4"] = f"""
<h2 id="s4">4. Le equazioni stimate e la logica di identificazione</h2>
<p>Vengono stimate tre specifiche annidate che differiscono <strong>solo</strong> per il
livello di controllo geografico. Il confronto tra loro <em>è</em> la strategia di
identificazione: non si sceglie il modello migliore, si legge come il coefficiente si muove
quando si aggiunge controllo.</p>
{sem('''<p>Il problema da risolvere è questo: le zone più esposte al rischio idrogeologico
sono spesso anche zone periferiche, di pianura industriale, lontane dal centro. Se
osserviamo che costano meno, non sappiamo se costano meno <em>perché</em> sono a rischio o
<em>perché</em> sono periferiche.</p>
<p>La strategia consiste nel porre la stessa domanda tre volte, con vincoli sempre più
stretti:</p>
<p><strong>A0</strong> — le zone più esposte costano meno? (senza chiedersi dove siano)<br>
<strong>A1</strong> — ...a parità di provincia, di fascia di centralità, di distanza dal
centro e di urbanizzazione?<br>
<strong>A2</strong> — ...confrontando soltanto zone che stanno <em>nello stesso comune</em>?</p>
<p>Un'analogia: per capire se un metodo di studio migliora i voti, prima confronti tutti gli
studenti d'Italia, poi solo quelli della stessa scuola, poi solo quelli della stessa classe.
Se il vantaggio sparisce quando confronti compagni di classe, quel che vedevi all'inizio era
la differenza tra scuole, non l'effetto del metodo.</p>''')}
<p class="eq"><strong>A0 (naive)</strong>:&nbsp; log P<sub>c</sub> = α +
Σ<sub>k</sub> β<sub>k</sub>·RISK<sub>k,c</sub> + δ<sub>tip(c)</sub> + δ<sub>cond(c)</sub>
+ ε<sub>c</sub></p>
<p class="eq"><strong>A1 (provincia + amenità)</strong>:&nbsp; log P<sub>c</sub> = α +
Σ<sub>k</sub> β<sub>k</sub>·RISK<sub>k,c</sub> + δ<sub>tip(c)</sub> + δ<sub>cond(c)</sub>
+ δ<sub>fascia(c)</sub> + γ<sub>1</sub>·dist_cbd<sub>c</sub> + γ<sub>2</sub>·is_urban<sub>c</sub>
+ α<sub>prov(c)</sub> + ε<sub>c</sub></p>
<p class="eq"><strong>A2 (within-comune, modello principale)</strong>:&nbsp;
log P<sub>c</sub> = α + Σ<sub>k</sub> β<sub>k</sub>·RISK<sub>k,c</sub> + δ<sub>tip(c)</sub>
+ δ<sub>cond(c)</sub> + δ<sub>fascia(c)</sub> + γ<sub>1</sub>·dist_cbd<sub>c</sub>
+ γ<sub>2</sub>·is_urban<sub>c</sub> + α<sub>com(c)</sub> + ε<sub>c</sub></p>
<p><strong>A2w</strong>: identica ad A2, stimata con WLS e pesi pari al numero di particelle
della cella.</p>
{sem('''<p>Come si legge un'equazione di questo tipo, termine per termine: a sinistra c'è
quello che vogliamo spiegare (il logaritmo del prezzo). A destra, il primo blocco
<em>Σ β·RISK</em> sono i sette rischi, e i β sono i numeri che ci interessano: quanto il
prezzo si muove al variare di ciascun rischio.</p>
<p>Tutti i termini con δ e α sono <strong>controlli</strong>: servono a neutralizzare
differenze che non c'entrano con il rischio ma che influenzano il prezzo — il tipo di
immobile, lo stato di conservazione, la centralità, il comune. Il termine ε è ciò che il
modello non riesce a spiegare.</p>
<p>L'unica differenza tra le tre equazioni è l'ultimo controllo geografico: assente in A0,
la provincia in A1, il comune in A2.</p>''')}
<h3>4.1 Che cosa contiene ciascun termine</h3>
<ul>
<li><strong>δ<sub>tip</sub></strong> — 12 tipologie edilizie: assorbe il fatto che un box e
un ufficio non hanno lo stesso €/m².</li>
<li><strong>δ<sub>cond</sub></strong> — 3 stati conservativi (normale / ottimo / scadente).</li>
<li><strong>δ<sub>fascia</sub></strong> — 5 fasce OMI (B centrale, C semicentrale,
D periferica, E suburbana, R extraurbana): è la gerarchia di centralità <em>interna</em> al
comune, e quindi il controllo di amenità più importante in A2.</li>
<li><strong>dist_cbd</strong> — distanza in linea d'aria dal capoluogo di provincia (km).</li>
<li><strong>is_urban</strong> — quota di particelle della cella in land cover artificiale.</li>
<li><strong>α<sub>prov</sub></strong> (4) e <strong>α<sub>com</sub></strong> (345) — effetti
fissi geografici, il termine che cambia tra le tre specifiche.</li>
</ul>
{sem('''<p>«Effetto fisso di comune» suona astratto, ma significa una cosa molto concreta:
il modello stima un livello di prezzo proprio di ciascun comune, e poi guarda le differenze
<em>rispetto a quel livello</em>. Tutto ciò che è comune all'intero comune — il reddito
medio, il mercato del lavoro locale, la reputazione della città, la pressione fiscale — viene
neutralizzato in blocco, anche se non l'abbiamo misurato.</p>
<p>È il vantaggio di questa tecnica: non serve conoscere <em>tutte</em> le caratteristiche di
un comune per tenerle sotto controllo. Il costo, che vedremo in §5, è che restano
confrontabili solo zone dello stesso comune.</p>''')}
<h3>4.2 Che cosa isola ogni passaggio</h3>
<div class="card"><b class="lbl">A0 → A1.</b> Aggiunge provincia, fascia, centralità e
urbanizzazione. Un coefficiente che si dimezza qui stava misurando
<em>macro-localizzazione</em>: i territori più rischiosi sono anche mercati mediamente più
economici, per ragioni che non hanno a che vedere con il rischio.</div>
<div class="card"><b class="lbl">A1 → A2.</b> Sostituisce la provincia con il comune: il
confronto avviene <em>tra zone dello stesso comune</em>. Sopravvive solo ciò che è
identificato da differenze di esposizione interne al singolo comune — il caso in cui una
capitalizzazione del rischio sarebbe più credibile, perché il mercato locale è comune.</div>
<div class="card"><b class="lbl">Il limite di A2.</b> La saturazione ha un prezzo: se un
rischio non varia entro comune, il suo coefficiente A2 è stimato su quasi nulla. Per questo
la scala di identificazione è misurata esplicitamente in §5, <em>prima</em> di leggere i
coefficienti.</div>
<h3>4.3 Inferenza: perché gli standard error sono clusterizzati per comune</h3>
<p>Il termine di errore è eteroschedastico (verificato, §8) e correlato entro comune (ICC del
log-prezzo a livello comunale = 0,36; sui residui di A1 la correlazione entro comune sale a
0,89). Tutte le specifiche usano <strong>standard error cluster-robust per comune</strong>:
345 cluster, ampiamente sopra la soglia convenzionale (~30) sotto la quale l'approssimazione
asintotica diventa inaffidabile. Il livello provincia (4 cluster) non è usato per l'inferenza
primaria ed è verificato a parte con bootstrap (§9.2).</p>
{sem('''<p>Lo standard error misura <em>quanta informazione indipendente</em> c'è davvero nei
dati. Immagina di voler stimare l'altezza media degli italiani misurando 2.728 persone: se
però queste appartengono a 345 famiglie, non hai 2.728 informazioni indipendenti, perché i
membri della stessa famiglia si somigliano. Il decimo fratello aggiunge molto meno del primo.</p>
<p>Qui le «famiglie» sono i comuni, e la somiglianza interna nasce da due fatti che si
sommano: (1) la variabile di rischio è spesso identica per tutte le celle dello stesso comune
— la pericolosità sismica di un comune è una sola; (2) i prezzi dello stesso comune sono
stimati dallo stesso ufficio, con gli stessi criteri, sullo stesso mercato locale, quindi gli
errori del modello si muovono insieme invece di compensarsi.</p>
<p>Clusterizzare significa dire al calcolo dell'incertezza: <strong>conta i comuni, non le
celle</strong>. Non cambia la stima, cambia solo quanto siamo sicuri di essa.</p>''')}
<p>Quanto conti concretamente, sui dati di questo studio, si vede confrontando lo stesso
modello A1 con tre diversi modi di calcolare l'incertezza — stessi coefficienti, solo
l'errore standard cambia:</p>
<div class="tabwrap">{tbl(se_tab, nd=4, small=True)}</div>
{sem('''<p>La tabella è il modo più diretto per far vedere perché il clustering non è un
tecnicismo. Con il calcolo classico gli standard error sono 2,5–4,5 volte più piccoli, e sei
rischi su sette risultano significativi con p inferiore a 0,001.</p>
<p>In particolare l'alluvione risulterebbe associata a prezzi più bassi con altissima
significatività: avremmo concluso che il rischio alluvionale <em>è</em> scontato nelle
quotazioni. Con il calcolo corretto lo stesso coefficiente ha p = 0,059 e diventa il
risultato nullo su cui si regge la conclusione opposta.</p>
<p>La colonna «SE HC1» mostra che non basta correggere per l'eteroschedasticità: quella
corregge la <em>variabilità</em> degli errori, non il fatto che siano <em>correlati tra loro</em>
dentro lo stesso comune.</p>''')}
<h3>4.4 Bontà di adattamento delle quattro specifiche</h3>
<div class="tabwrap">{tbl(fit_show, nd=3, small=True)}</div>
<p>Il salto di R² da A1 (0,861) ad A2 (0,958) non è un miglioramento del modello di rischio:
è l'effetto meccanico di 344 dummy comunali aggiuntive (371 parametri su 2.728 osservazioni,
2.357 gradi di libertà residui). Va letto come costo dell'identificazione, non come guadagno
esplicativo.</p>
{sem('''<p>L'R² dice quale quota della variabilità dei prezzi il modello riesce a riprodurre.
Passa da 86% a 96% quando si aggiungono gli effetti fissi comunali — ma è un aumento
scontato: stiamo aggiungendo 344 variabili che descrivono il livello di prezzo di ciascun
comune, quindi il modello «impara» i livelli locali quasi per definizione.</p>
<p>Se in sede d'esame qualcuno leggesse quel 96% come segno di un modello eccellente,
la risposta corretta è: non è una misura di qualità della relazione rischio-prezzo, è il
prezzo pagato per poter fare confronti dentro lo stesso comune. Per questo lo si riporta
sempre insieme al numero di parametri e ai gradi di libertà residui.</p>''')}
"""

# ====================================================================
# 5. SCALA DI IDENTIFICAZIONE
# ====================================================================
m3f = m3.copy()
m3f["quota_within_%"] = (100 * m3f["quota_within"]).round(1)
m3f["Rischio"] = m3f["variabile"].map(IT)
m3f["Scala informativa"] = m3f["variabile"].map(SCALA)
m3show = m3f[["Rischio", "quota_within_%", "Scala informativa"]].rename(
    columns={"quota_within_%": "% varianza within-comune"})
gran_show = gran.copy()
gran_show["quota conservata"] = (100 * gran_show["quota conservata"]).round(0).astype(int).astype(str) + "%"
gran_show["costo del clustering (A2)"] = gran_show["costo del clustering (A2)"].round(2).astype(str) + "×"

S["s5"] = f"""
<h2 id="s5">5. Passaggio decisivo: a quale scala ciascun rischio è identificato</h2>
<p>Prima di leggere qualunque coefficiente A2, si misura <strong>quanta varianza di ciascun
rischio sopravvive alla rimozione delle medie comunali</strong>. È la quantità di
informazione realmente disponibile per stimare quel coefficiente within-comune:</p>
{tbl(m3show, nd=1)}
{sem('''<p>Il ragionamento è questo. In A2 confrontiamo zone dello stesso comune. Ma per
confrontarle serve che il rischio <em>cambi</em> da una zona all'altra dentro quel comune. Se
il rischio è lo stesso ovunque nel comune, non c'è nulla da confrontare, e il numero che il
modello produce è aria fritta.</p>
<p>La tabella misura esattamente questo. Il rischio sismico ha 0,0%: dentro un comune la
pericolosità sismica è per costruzione la stessa ovunque — non esiste un comune con una zona
sismica e una no. Incendio, costa e precipitazioni sono poco sopra lo zero, perché derivano
da griglie più grandi del comune stesso.</p>
<p>Alluvione (10,4%) e subsidenza (7,7%) sono invece gli unici due rischi che variano
davvero da una zona all'altra dello stesso comune: solo per loro la domanda «a parità di
comune, le zone più esposte costano meno?» ha una risposta sensata. Ed è un bene che sia
proprio l'alluvione, che è il rischio centrale della tesi.</p>''')}
<p>Convenzione di lettura dichiarata (non è un test): ≥7% di varianza within → A2 è la
specifica di riferimento; 1,5–7% → A1, con within limitata; &lt;1,5% → il coefficiente A2
non è interpretabile e la scala informativa è A1 (tra comuni, entro provincia).</p>
<div class="box"><p><strong>Conseguenza pratica.</strong> Per PGA, incendio, costa e r99pday
il coefficiente A2 è stimato su una frazione trascurabile di variazione — infatti l'intervallo
di confidenza della PGA in A2 è di ±20 punti percentuali, cioè privo di contenuto
informativo. <strong>Ogni conclusione dello studio dichiara la scala a cui si riferisce</strong>,
e le analisi di eterogeneità (§9.7) sono condotte nella scala informativa di ciascun rischio,
non in una scala unica scelta per comodità.</p></div>
<h3>5.1 Quanta granularità esiste davvero sotto il comune</h3>
<p>Un confronto tra zone dello stesso comune è possibile solo dove un comune ha più di una
zona OMI quotata: sono <strong>{N_MULTI} comuni su 345</strong>, per
<strong>{N_CELLE_MULTI} celle</strong>. Dentro quei comuni, la variazione di ciascun rischio
effettivamente disponibile e il costo che il clustering comporta in A2 sono i seguenti:</p>
<div class="tabwrap">{tbl(gran_show[["Rischio", "sd entro comune", "sd complessiva",
                                      "quota conservata", "costo del clustering (A2)"]],
                          nd=4, small=True)}</div>
{sem('''<p>Questa tabella risponde a un'obiezione che è ragionevole aspettarsi: «aggregando e
clusterizzando non state perdendo il dettaglio fine, quello in cui l'effetto del clima
potrebbe manifestarsi?».</p>
<p>La risposta è in due parti. <strong>Primo</strong>: dove il dettaglio esiste davvero, il
metodo lo usa e lo paga poco. Per l'alluvione, dentro i comuni con più zone, si conserva il
60% della variabilità complessiva, e clusterizzare allarga l'intervallo di appena 1,24 volte.
<strong>Secondo</strong>: dove il dettaglio non esiste, il metodo giustamente non se lo
inventa. La pericolosità sismica conserva il 3%: quella variazione è il residuo aritmetico
delle medie sulle particelle, non un gradiente reale di rischio dentro il comune.</p>
<p>Da notare il caso dell'incendio, dove il clustering <em>restringe</em> l'intervallo
(0,75×): la formula segue la struttura dei dati, non applica una penalizzazione fissa a
scopo prudenziale.</p>''')}
"""

# ====================================================================
# 6. PROTOCOLLO
# ====================================================================
CRIT = {
    "Modelli spaziali SAR/SEM/SDM": (
        "Modelli spaziali SAR / SEM / SDM",
        "<em>Condizione</em>: dipendenza spaziale residua non assorbita dagli effetti fissi. "
        "Moran's I sui residui A2 con W ristretta ai vicini di <em>altri</em> comuni: "
        "I = −0,008 (p = 0,42). Con W piena I = −0,103 (p = 0,0001), ma è artefatto di "
        "demeaning (§8.1). Condizione non soddisfatta."),
    "Forme non lineari (spline/GAM/dummy classi)": (
        "Forme non lineari (spline / GAM / dummy di classe)",
        "<em>Condizione</em>: rigetto della forma funzionale. RESET su A2: stat = 137,4, "
        "p ≈ 1·10⁻³⁰. Condizione soddisfatta → esito in §9.6."),
    "Trattamento multicollinearita' (PCA/Ridge/blocchi)": (
        "Trattamento della multicollinearità (PCA / Ridge / blocchi)",
        "<em>Condizione</em>: almeno 2 variabili con VIF within-comune &gt; 10. "
        "Osservato: 0 variabili sopra soglia (VIF massimo 1,53)."),
    "Mixed/multilevel model": (
        "Modello multilivello (mixed model)",
        "<em>Condizione</em>: varianza di cluster non già catturata da un effetto fisso. "
        "ICC di comune = 0,36, interamente assorbita dagli FE comunali di A2."),
    "Machine learning (RF/GBM) sintetico": (
        "Machine learning sintetico (RF / gradient boosting)",
        "Nessuna condizione diagnostica: controllo indipendente previsto dal protocollo a "
        "prescindere dall'esito del Livello 1 → esito in §9.8."),
}
m6f = pd.DataFrame([{
    "Tecnica di Livello 2": CRIT.get(t, (t, m))[0],
    "Esito": ('<span class="ok">ATTIVATA</span>' if bool(a)
              else '<span class="no">non attivata</span>'),
    "Criterio di attivazione e statistica osservata": CRIT.get(t, (t, m))[1],
} for t, a, m in zip(m6["tecnica"], m6["attivata"], m6["motivazione"])])
liv3 = pd.DataFrame([
    ("Panel / effetti fissi temporali", "Almeno 2 periodi con variazione utile",
     "Solo 2 semestri e prezzo variato nel 26% delle celle → usato solo come indizio di "
     "repricing (§9.9), non come panel"),
    ("Difference-in-differences", "Shock databile + gruppo di controllo credibile",
     "Nessun evento con data e controllo difendibili nella finestra osservata"),
    ("Variabili strumentali", "Strumento con esogeneità argomentabile",
     "Nessuno strumento difendibile: non forzato (uno strumento debole avrebbe prodotto "
     "stime peggiori del problema che risolve)"),
    ("Regression discontinuity", "Soglia netta, non manipolabile, osservabile",
     "Le classi PGRA/PAI non presentano una soglia continua non manipolabile a livello di "
     "cella"),
], columns=["Tecnica di Livello 3", "Condizione richiesta", "Perché è esclusa"])

S["s6"] = f"""
<h2 id="s6">6. La regola di decisione: come si è scelto cosa stimare</h2>
<p>Il rischio metodologico principale in uno studio con molte tecniche disponibili è
scegliere <em>a posteriori</em> la specifica che produce il risultato desiderato. Il
protocollo lo previene con una gerarchia dichiarata prima della stima:</p>
<ul>
<li><strong>Livello 1 — sempre eseguito</strong>: OLS log-lineare con effetti fissi
geografici crescenti, diagnostica completa, robustezza di base.</li>
<li><strong>Livello 2 — solo se una diagnostica lo attiva</strong>: modelli spaziali, forme
non lineari, trattamento della multicollinearità, multilivello, machine learning.</li>
<li><strong>Livello 3 — solo se la struttura del dato lo consente</strong>: panel,
difference-in-differences, variabili strumentali, regression discontinuity.</li>
</ul>
{sem('''<p>Questo è il punto che protegge lo studio dall'accusa più insidiosa: «avete provato
molte tecniche e riportato quella che vi conveniva».</p>
<p>La difesa consiste nell'aver deciso <em>prima</em>, e per iscritto, la regola:
«useremo il modello spaziale <strong>se e solo se</strong> questo test darà questo esito».
Così l'esito dei test — non il risultato che ci piace — determina quali analisi vengono
eseguite. Ed è per questo che le tecniche <em>non</em> usate vengono comunque elencate con il
test che le ha escluse: un'esclusione motivata è un risultato dello studio, non una
mancanza.</p>''')}
<p>Due vincoli aggiuntivi: (i) la <strong>significatività statistica non è mai usata come
criterio di selezione</strong> del modello; (ii) tutte le stime sono descritte come
associazioni condizionate, mai come effetti causali.</p>
<h3>6.1 Esito delle attivazioni di Livello 2</h3>
<div class="tabwrap">{tbl(m6f, small=True)}</div>
<h3>6.2 Esclusioni di Livello 3, con il criterio esplicito</h3>
<div class="tabwrap">{tbl(liv3, small=True)}</div>
{sem('''<p>Le tecniche di Livello 3 sono quelle che permetterebbero di parlare di
<em>causa</em> e non solo di associazione. Richiedono però situazioni particolari: due
momenti nel tempo con un cambiamento vero (panel), un evento con una data precisa e un gruppo
di confronto non colpito (difference-in-differences), oppure una soglia amministrativa netta
che separi due gruppi altrimenti simili (regression discontinuity).</p>
<p>Nessuna di queste condizioni è presente in questi dati. Da qui la scelta — dichiarata in
apertura e mantenuta in ogni sezione — di non usare mai un linguaggio causale: si dice
«associato a», non «causato da».</p>''')}
"""

# ====================================================================
# 7. LETTURA DEI NUMERI
# ====================================================================
S["s7"] = """
<h2 id="s7">7. Come si legge ogni numero prodotto</h2>
<h3>7.1 Metrica economica: % per +1 deviazione standard</h3>
<p>Un β su una variabile in unità fisiche eterogenee (g di PGA, mm/anno, giorni/anno) non è
confrontabile tra rischi. Ogni coefficiente è quindi riportato anche come variazione
percentuale del prezzo associata a un aumento di una deviazione standard campionaria del
rischio:</p>
<p class="eq">effetto % per +1 sd = 100 · ( e<sup>β · sd(RISK)</sup> − 1 ),&nbsp;&nbsp;
con l'IC95 trasformato nella stessa metrica</p>
""" + sem("""<p>Serve a rendere confrontabili grandezze che non lo sono. Chiedersi «cosa
succede al prezzo se la PGA aumenta di 1» è privo di senso, perché la PGA varia tra 0 e 0,19:
un aumento di 1 non esiste in natura. Chiedersi «cosa succede se passo a una zona
sensibilmente più esposta» ha invece senso per tutti i rischi, e la deviazione standard è la
misura di quel «sensibilmente più esposta».</p>
<p>Perciò tutti i risultati si leggono così: <em>+1 deviazione standard di rischio è
associata a una variazione del prezzo del X%</em>.</p>""") + """
<h3>7.2 Test multipli: due famiglie, dichiarate</h3>
<p>Con 7 rischi × più specifiche, testare tutto al 5% grezzo produce falsi positivi. Si
applica la correzione di <strong>Holm</strong> (che controlla la probabilità di commettere
almeno un falso positivo nella famiglia) su due famiglie esplicite:</p>
<ul>
<li><strong>famiglia interna</strong>: i 7 rischi entro ciascuna specifica — riportata
accanto ai coefficienti;</li>
<li><strong>famiglia primaria globale</strong>: 7 rischi × 2 scale informative
{A1, A2} = 14 test — è il filtro con cui si decide che cosa «sopravvive» (§9).</li>
</ul>
""" + sem("""<p>Il p-value al 5% significa: «se non ci fosse alcun effetto, un risultato così
uscirebbe per puro caso una volta su venti». Il problema è che noi non facciamo una domanda:
ne facciamo quattordici. Su quattordici domande, trovarne una «significativa» per puro caso è
quasi normale — è la stessa ragione per cui, lanciando venti volte una moneta, prima o poi
escono cinque teste di fila.</p>
<p>La correzione di Holm alza l'asticella in proporzione al numero di domande poste. Dopo la
correzione, delle quattordici, ne restano due — ed è questo che permette di dire che quelle
due non sono un artefatto del numero di test.</p>""") + """
<h3>7.3 Assenza di evidenza ≠ evidenza di assenza: la potenza statistica</h3>
<p>Un p-value alto può significare due cose opposte: l'effetto non c'è, oppure il disegno
non era in grado di vederlo. Per separarle, ogni nullo è accompagnato dal <strong>minimum
detectable effect</strong> all'80% di potenza e al 5%:</p>
<p class="eq">MDE<sub>β</sub> = (z<sub>0,975</sub> + z<sub>0,80</sub>) · SE(β) ≈ 2,8 · SE(β)
&nbsp;&nbsp;→&nbsp;&nbsp; MDE<sub>%</sub> = 100 · ( e<sup>MDE<sub>β</sub> · sd</sup> − 1 )</p>
<p>Regola di classificazione applicata a ogni coefficiente non significativo: se
MDE ≤ 5% per sd il nullo è <strong>informativo</strong> (il disegno avrebbe visto uno sconto
della magnitudine tipicamente documentata in letteratura, 5–10%, e non lo trova); se
MDE &gt; 5% per sd il risultato è <strong>non conclusivo per bassa potenza</strong>.</p>
""" + sem("""<p>Questo è forse il concetto più importante da spiegare bene, perché è ciò che
trasforma un «non abbiamo trovato niente» in un risultato scientifico.</p>
<p>L'analogia: se peso due sacchi con una bilancia da cucina e segna lo stesso valore, posso
dire che pesano uguale. Ma se la bilancia ha una precisione di 100 grammi, non posso
escludere che uno pesi 80 grammi più dell'altro. Prima di dire «sono uguali» devo dichiarare
quanto è precisa la mia bilancia.</p>
<p>Il minimum detectable effect è la precisione della nostra bilancia. Per l'alluvione vale
3,7%: significa che se ci fosse stato uno sconto di prezzo del 4% o più, l'avremmo visto. La
letteratura internazionale documenta sconti tipici tra il 2% e l'8%: gran parte di quel
campo di valori sarebbe stato rilevabile, e non lo osserviamo. È in questo senso preciso che
il nostro «zero» è un risultato e non una resa.</p>""")

# ====================================================================
# 8. DIAGNOSTICA
# ====================================================================
diag = pd.DataFrame([
    ("Multicollinearità", "VIF sui livelli e within-comune (soglie 5 / 10)",
     "massimo 1,53 (nessuna variabile &gt; 5)",
     "Nessun trattamento: PCA/Ridge/blocchi non attivati"),
    ("Eteroschedasticità", "Breusch-Pagan su A2",
     "LM = 738,5; p ≈ 9·10⁻²⁷ → presente",
     "Già gestita: SE cluster-robust adottati in tutte le specifiche"),
    ("Normalità dei residui", "Jarque-Bera su A2",
     "JB = 144,6; p ≈ 4·10⁻³² (skew −0,15; curtosi ecc. +1,09)",
     "Rigetto formale non critico con N = 2.728 e SE robusti; rilevante solo sui "
     "sottoinsiemi piccoli"),
    ("Forma funzionale", "RESET di Ramsey (ŷ², ŷ³; Wald con SE cluster)",
     "stat = 137,4; p ≈ 1·10⁻³⁰ → rigetto della log-lineare",
     "<strong>Attiva</strong> le forme non lineari di Livello 2 (esito in §9.6)"),
    ("Osservazioni influenti", "Cook's D, soglia 4/n = 0,0015",
     "11 celle con leva h &gt; 0,99 (assorbite da FE quasi-singleton, Cook indefinito); "
     "sulle restanti 193 celle (7,1%) sopra soglia, massimo 0,015",
     "Nessuna osservazione dominante; verificato con winsorizzazione e trim p1/p99 (§9.1)"),
    ("Dipendenza spaziale", "Moran's I sui residui di zona (KNN k=8, 9.999 permutazioni) "
     "+ test LM lag/error",
     "A1: I = +0,302 (p = 0,0001); A2 con W piena: I = −0,102 (p = 0,0001); A2 con W "
     "ristretta ai vicini di <em>altri</em> comuni: I = −0,008 (p = 0,42)",
     "Dipendenza genuina interamente assorbita dagli FE comunali → SAR/SEM/SDM "
     "<strong>non</strong> attivati (§8.1)"),
    ("Struttura gerarchica", "ICC da modello a intercetta casuale",
     "ICC comune 0,359; ICC zona OMI 0,352",
     "Varianza di cluster già catturata dagli FE di comune → multilivello non attivato"),
], columns=["Ipotesi verificata", "Test", "Statistica osservata", "Conseguenza operativa"])
vif_show = m4.reset_index().rename(columns={"index": "variabile"})

S["s8"] = f"""
<h2 id="s8">8. Diagnostica: che cosa è stato verificato e che cosa ne è seguito</h2>
<p>La diagnostica non è un allegato di controllo: è il meccanismo che <em>attiva</em> o
<em>esclude</em> le tecniche successive. Per questo è riportata integralmente, anche quando
il suo esito è «non serve fare altro».</p>
{sem('''<p>Ogni modello statistico funziona bene solo se valgono certe condizioni sui dati.
La diagnostica è la serie di controlli che verifica se quelle condizioni reggono e, quando
non reggono, indica che cosa bisogna cambiare.</p>
<p>Il modo più chiaro di leggere la tabella è per colonne: la prima dice <em>quale condizione
sto controllando</em>, la seconda <em>con quale test</em>, la terza <em>cosa ho trovato</em>,
la quarta — la più importante — <em>che cosa ho di conseguenza deciso di fare</em>. Nessun
test è riportato senza una conseguenza operativa.</p>''')}
<div class="tabwrap">{tbl(diag, small=True)}</div>
<div class="tabwrap">{tbl(vif_show, nd=2, small=True)}</div>
{sem('''<p>Traduzione dei quattro controlli principali:</p>
<p><strong>VIF</strong> — verifica che i sette rischi non siano tra loro talmente sovrapposti
da rendere impossibile distinguerne gli effetti. Il valore massimo è 1,53 contro una soglia
di attenzione di 5: i rischi sono sufficientemente distinti tra loro.</p>
<p><strong>Breusch-Pagan</strong> — verifica se l'errore del modello sia più grande in certe
zone che in altre. Lo è (i mercati costosi sono più variabili), ed è esattamente ciò da cui
proteggono gli standard error robusti già adottati.</p>
<p><strong>Jarque-Bera</strong> — verifica se gli errori seguano la curva a campana. Non la
seguono perfettamente, ma con 2.728 osservazioni e standard error robusti questo non
compromette l'inferenza: è un rigetto formale senza conseguenze pratiche, e va detto così.</p>
<p><strong>RESET</strong> — verifica se la forma dell'equazione sia adeguata. Qui il test
rigetta, ed è l'unico caso in cui la diagnostica ha imposto un'analisi aggiuntiva: le forme
flessibili di §9.6, che servono a capire <em>dove</em> stia il problema.</p>''')}
<h3>8.1 Il caso spaziale: perché un Moran significativo non ha attivato i modelli spaziali</h3>
<p>È il punto in cui la diagnostica poteva essere letta male. Gli effetti fissi comunali
impongono che i residui sommino a circa zero <em>entro ciascun comune</em>: due zone vicine
dello stesso comune risultano perciò meccanicamente anticorrelate, e il Moran's I diventa
negativo e «significativo» per un artefatto di costruzione, non per dipendenza spaziale
reale. La verifica decisiva consiste nel ricostruire la matrice dei pesi escludendo i vicini
dello stesso comune: la dipendenza in A1 resta forte (I = +0,278, p = 0,0001), mentre in A2
scompare (I = −0,008, p = 0,42).</p>
{sem('''<p>Il test di Moran misura se zone vicine tendono ad avere errori simili: se sì,
significa che il modello sta trascurando qualcosa che ha una struttura geografica, e servirebbe
un modello spaziale.</p>
<p>Il tranello è questo: quando si inseriscono gli effetti fissi di comune, il modello
costringe gli errori di ciascun comune a compensarsi tra loro, cioè a sommare zero. Se una
zona ha errore positivo, un'altra zona dello stesso comune deve averlo negativo — non perché
esista un fenomeno geografico, ma per come è costruito il calcolo. Il test lo segnala come
«dipendenza spaziale» e ci indurrebbe a stimare un modello complicato per correggere un
problema inesistente.</p>
<p>La verifica giusta è quindi guardare solo le coppie di zone vicine che appartengono a
<em>comuni diversi</em>, dove l'artefatto non può prodursi. Fatto questo, la dipendenza
sparisce (p = 0,42). Conclusione: gli effetti fissi comunali hanno già assorbito tutta la
struttura geografica, e il modello spaziale non serve.</p>''')}
"""

# ====================================================================
# 9. ROBUSTEZZA
# ====================================================================
f5 = b64img(os.path.join(MOD, "f5_traiettoria_coefficienti.png"), "traiettoria",
            "Fig. 1 — Traiettoria dei coefficienti di rischio A0 → A1 → A2 (IC95, SE cluster "
            "comune). Ogni pannello è un rischio; i tre punti sono le tre specifiche. Un "
            "coefficiente che si avvicina alla linea dello zero passando da A0 ad A2 stava "
            "misurando localizzazione, non rischio.")
f6 = b64img(os.path.join(ROB, "f6_curva_specificazione.png"), "curva specificazione",
            "Fig. 2 — Curva di specificazione per alluvione e precipitazioni estreme: ogni "
            "riga è una specifica stimata, il punto è la stima e la barra l'intervallo di "
            "confidenza; le headline sono in rosso.")
r3f = r3.copy()
r3f["variabile"] = r3f["variabile"].map(IT)
r3f = r3f.rename(columns={"variabile": "Rischio", "beta": "β (A1)",
                          "t_cluster_prov": "t (cluster provincia)",
                          "p_wcb_webb": "p wild bootstrap"})
lopo1 = r2a.copy()
lopo1["variabile"] = lopo1["variabile"].map(IT)
curve_summ = (r7.groupby("famiglia")["pct_per_1sd"]
              .agg(["count", "min", "median", "max"]).round(2).reset_index()
              .rename(columns={"famiglia": "Famiglia di rischio", "count": "n. specifiche",
                               "min": "min %/sd", "median": "mediana %/sd",
                               "max": "max %/sd"}))
e6show = e6[e6["dip"] == "log(canone/prezzo)"][
    ["spec", "variabile", "pct_per_1sd", "p", "p_holm"]].copy()
e6show["variabile"] = e6show["variabile"].map(IT)
e6show = e6show.rename(columns={"spec": "Spec.", "variabile": "Rischio",
                                "pct_per_1sd": "% per +1 sd", "p": "p grezzo",
                                "p_holm": "p Holm"})
e3f = e3.copy()
e3f["rischio"] = e3f["rischio"].map(IT)
e3p = e3f.pivot_table(index=["spec", "rischio"], columns="gruppo",
                      values="pct_per_1sd").round(2).reset_index()
e4rs = e4r.rename(columns={"modello": "Modello", "r2_cv": "R² out-of-comune",
                           "sd": "sd tra fold"})

S["s9"] = f"""
<h2 id="s9">9. La batteria di robustezza: procedura per procedura</h2>
<p>Ogni verifica qui elencata risponde a una domanda diversa su una possibile fragilità del
risultato. Sono riportate tutte, anche quelle il cui esito non cambia nulla, perché
l'informazione rilevante è <em>quali attacchi il risultato ha retto</em>.</p>
{sem('''<p>Il senso di questa sezione, detto in una frase: <em>abbiamo cercato in ogni modo di
far crollare il nostro risultato, e vi mostriamo tutti i tentativi</em>.</p>
<p>Ogni sottosezione è un tentativo diverso: e se avessimo misurato la pioggia in un altro
modo? e se togliessimo una provincia? e se il calcolo dell'incertezza fosse sbagliato? e se
usassimo un modello completamente diverso? Un risultato che sopravvive a tutti questi
attacchi è affidabile in un senso molto più forte di un risultato che ha solo un p-value
basso.</p>''')}
{f5}
<h3>9.1 Varianti di specifica (24 specifiche stimate in tutto, headline incluse)</h3>
<p>Tutte replicano A1/A2 con <strong>una sola modifica per volta</strong>: precipitazioni
misurate con rr20mm (soglia assoluta 20 mm) invece di r99pday; alluvione e frana come quote
di particelle esposte invece che come ordinali; sisma come zona sismica invece che PGA;
subsidenza come classe di rischio invece che velocità; log-prezzo winsorizzato a p1/p99;
esclusione degli outlier di prezzo; stime pesate A1w e A2w; aggregazione territoriale
alternativa comune × tipologia.</p>
<div class="tabwrap">{tbl(curve_summ, nd=2, small=True)}</div>
{f6}
{sem('''<p>Molte scelte fatte durante l'analisi erano legittime ma non obbligate: misurare
la pioggia come «giorni oltre il 99° percentile» o come «giorni sopra 20 mm»; trattare
l'alluvione come classe di gravità o come semplice presenza/assenza; tenere o togliere le
zone con prezzi estremi.</p>
<p>Il rischio è di aver ottenuto il risultato per via di una di queste scelte. La verifica
consiste nel rifarle tutte, una alla volta, e guardare dove finisce il coefficiente. La
tabella e la figura mostrano l'esito: per l'alluvione, sulle 24 specifiche, la stima resta
sempre tra −5,2% e +0,5%, con mediana −1,4%, e non è mai significativamente negativa nelle
specifiche controllate. Il risultato non dipende dalle scelte discrezionali.</p>
<p>Nota importante: la curva raccoglie solo le specifiche <em>giustificate dal protocollo</em>,
non tutte le combinazioni immaginabili. Una curva costruita provando ogni combinazione
possibile sarebbe essa stessa una forma di p-hacking.</p>''')}
<h3>9.2 Inferenza con pochi cluster: wild cluster bootstrap di provincia</h3>
<p>Il clustering primario è al comune (345 cluster). Resta però la domanda: e se gli shock
fossero comuni a un'intera provincia? A quel livello i cluster sono 4, troppo pochi perché
gli SE asintotici siano affidabili. La procedura è il <strong>wild cluster bootstrap con
null imposto</strong>: (i) si stima il modello imponendo β<sub>k</sub> = 0; (ii) si rigenerano
B = 4.999 campioni moltiplicando i residui per pesi di Webb estratti <em>a livello di
provincia</em>; (iii) si ricalcola la t su ogni campione; (iv) p = (#{{|t*| ≥ |t<sub>oss</sub>|}} + 1)/(B + 1).</p>
<div class="tabwrap">{tbl(r3f, nd=4, small=True)}</div>
{sem('''<p>Fin qui abbiamo trattato i comuni come unità indipendenti. Ma se esistesse un
fattore che agisce su un'<em>intera provincia</em> — una dinamica del mercato milanese, una
politica regionale — allora le unità davvero indipendenti sarebbero quattro, non 345.</p>
<p>Con quattro sole unità le formule statistiche standard non funzionano più, perché sono
approssimazioni valide quando i gruppi sono molti. Il bootstrap aggira il problema con una
simulazione: si costruiscono al computer quasi cinquemila mondi possibili in cui l'effetto
per ipotesi <em>non esiste</em>, e si guarda quanto spesso in quei mondi si ottiene per caso
un risultato forte come il nostro.</p>
<p>L'esito è che a questo livello tutte le associazioni diventano borderline. È un limite
reale del disegno a quattro province, e lo dichiariamo invece di nasconderlo.</p>''')}
<h3>9.3 Leave-one-provincia-out</h3>
<p>Il modello è ristimato quattro volte, escludendo ogni volta una provincia (effetto % per
+1 sd, specifica A1):</p>
<div class="tabwrap">{tbl(lopo1, nd=2, small=True)}</div>
{sem('''<p>Verifica semplice da spiegare: rifacciamo tutto senza Milano, poi senza Napoli,
poi senza Roma, poi senza Torino. Se un risultato regge solo grazie a una provincia, qui si
vede subito.</p>
<p>Ed è successo: il premio per la vicinanza alla costa passa da +14,4% con tutte le province
a +3,0% senza Napoli. Non lo cancelliamo, lo <em>riqualifichiamo</em>: non è un premio
costiero generale, è un fenomeno delle isole e delle località costiere campane. Anche questa
è una scoperta, ottenuta proprio perché la verifica è stata fatta.</p>''')}
<h3>9.4 Test di falsificazione (placebo permutazionale)</h3>
<p>Si permuta la variabile di rischio mantenendo la struttura di raggruppamento e si ristima
999 volte, con calcolo esatto per Frisch-Waugh-Lovell.</p>
<ul>
<li><strong>(a) r99pday permutato tra comuni entro provincia, in A1</strong> — tasso di
rigetto placebo al 5%: <strong>7,0%</strong> (≈ nominale); p empirico del coefficiente
osservato: 0,013 → il segnale positivo non è un artefatto.</li>
<li><strong>(b) flood_ord permutato tra zone entro comune, in A2</strong> — tasso di rigetto
placebo: <strong>17,7%</strong>, oltre tre volte il nominale: <em>l'inferenza analitica
within-comune su flood è anticonservativa</em>. Conseguenza operativa: per quel coefficiente
si usa il p-value di permutazione, che vale 0,60 e conferma il nullo.</li>
</ul>
{sem('''<p>L'idea è quella del gruppo di controllo in medicina: si somministra al modello una
variabile finta e si verifica che non «reagisca». Concretamente si prende il rischio
alluvionale e lo si rimescola a caso tra le zone, in modo che il valore attribuito a ciascuna
zona sia sbagliato per costruzione. Se il modello continua a trovare effetti significativi su
un dato falso, c'è qualcosa che non va nella procedura.</p>
<p>Si ripete il rimescolamento 999 volte e si conta quante volte esce un risultato
«significativo». Dovrebbe capitare in circa il 5% dei casi. Nel test (a) è successo il 7%
delle volte: normale. Nel test (b) il 17,7%: troppo.</p>
<p>Che cosa significa il 17,7% e perché lo riportiamo comunque: significa che, nel confronto
tra zone dello stesso comune, la formula standard produce intervalli un po' troppo stretti, e
quindi qualche falso positivo. Nel nostro caso questo <em>rafforza</em> la conclusione — se il
metodo tende a trovare effetti che non ci sono e nonostante ciò non ne trova, il nullo è
ancora più solido — ma la correttezza impone di sostituire quel p-value con quello ottenuto
per permutazione, che non soffre del problema.</p>''')}
<h3>9.5 Canoni e rapporto canone/prezzo: la verifica che discrimina le due ipotesi</h3>
<p>Se il mercato prezzasse il rischio ma l'OMI-prezzi non lo riflettesse, il rischio dovrebbe
comparire nel <strong>rapporto canone/prezzo</strong> (rendimento lordo richiesto più alto
dove il rischio è più alto). Lo stesso modello è quindi ristimato su log(canone) e su
log(canone/prezzo), con Holm entro famiglia:</p>
<div class="tabwrap">{tbl(e6show, nd=3, small=True)}</div>
{sem('''<p>Questo è il test più elegante dello studio e vale la pena raccontarlo per esteso.</p>
<p>Abbiamo trovato che i prezzi OMI non scontano il rischio. Ma restano due spiegazioni
possibili: (a) il mercato non prezza il rischio; (b) il mercato lo prezza, ma lo strumento
amministrativo non lo registra.</p>
<p>Il rapporto tra canone d'affitto e prezzo di vendita è, in finanza, il rendimento
richiesto per possedere quell'immobile. Se un investitore percepisse un rischio maggiore,
pretenderebbe un rendimento maggiore: a parità di affitto, pagherebbe meno l'immobile. Quindi
nelle zone rischiose ci aspetteremmo un rapporto canone/prezzo più alto — anche se i prezzi
in sé non mostrano nulla.</p>
<p>Non lo troviamo: l'unico coefficiente che sopravvive alla correzione ha addirittura segno
opposto a quello atteso. Il che significa che l'OMI «liscia» in modo simmetrico sia i prezzi
sia i canoni, e che con soli dati OMI le due ipotesi non si possono separare del tutto. Lo
diciamo esplicitamente: è una conclusione dichiarata inconclusiva, non un risultato
mascherato.</p>''')}
<h3>9.6 Forme non lineari (attivate dal RESET)</h3>
<p>Il rigetto della forma log-lineare va localizzato: riguarda i controlli edonici o la
relazione rischio-prezzo? Quattro verifiche: spline cubica su dist_cbd e log-distanza;
interazione tipologia × fascia; alluvione e frana sostituite dalle quote di cella per classe
di pericolosità; r99pday in dummy di quartile.</p>
<div class="tabwrap">{tbl(e2.round(4), nd=4, small=True)}</div>
{sem('''<p>Il test RESET ha detto che l'equazione, così com'è, non descrive perfettamente i
dati. La domanda successiva — quella utile — è: <em>dove</em> sta l'imperfezione?</p>
<p>Se stesse nella relazione tra rischio e prezzo, sarebbe un problema grave: vorrebbe dire
che l'effetto del rischio esiste ma ha una forma che una retta non cattura (per esempio: nessun
effetto fino a una certa soglia di pericolosità, poi un crollo). Per verificarlo abbiamo
sostituito la misura continua con le tre classi di pericolosità separate, che permettono
qualsiasi andamento, anche a gradini.</p>
<p>Risultato: nessun gradiente coerente (classe bassa +6,5%, media −22,4%, alta −2,9%; test
congiunto p = 0,22). Se ci fosse davvero uno sconto per il rischio alluvionale, ci
aspetteremmo che la classe alta mostri lo sconto maggiore: non accade.</p>
<p>L'imperfezione sta invece nei controlli — nel modo in cui il prezzo varia con la distanza
dal centro e con il tipo di immobile — e restando lì non tocca le conclusioni sui rischi. Ed
è infatti quello che si osserva: cambiando la forma dei controlli, i coefficienti di rischio
non si muovono.</p>''')}
<h3>9.7 Eterogeneità per tipologia edilizia</h3>
<p>Le 12 tipologie sono raggruppate in Residenziale (1.288 celle), Commerciale (465),
Produttivo (662) e Box (313); le interazioni rischio × gruppo sono stimate
<strong>nella scala informativa di ciascun rischio</strong> (§5): A2 per alluvione e
subsidenza, A1 per PGA, incendio, precipitazioni e costa.</p>
<div class="tabwrap">{tbl(e3p, nd=2, small=True)}</div>
{sem('''<p>Qui rispondiamo a una domanda specifica della tesi: esistono tipologie di immobile
più penalizzate dal rischio? Un capannone in zona alluvionale dovrebbe soffrire più di un
box, per esempio.</p>
<p>Tecnicamente si lascia che il coefficiente del rischio sia diverso per ciascun gruppo,
invece di imporre che sia unico. La risposta è che l'alluvione non mostra sconto in
<em>nessun</em> gruppo. L'unica differenza forte riguarda la pericolosità sismica, che è
positiva per commerciale e produttivo: non è un premio al rischio, è il riflesso del fatto
che i territori a maggiore pericolosità sismica (campani e laziali) hanno una struttura del
mercato non residenziale diversa. Lo diciamo così, senza forzarne una lettura
economica.</p>''')}
<h3>9.8 Verifica indipendente con machine learning</h3>
<p>Si stimano gradient boosting e random forest, liberi di catturare non linearità e
interazioni, con <strong>GroupKFold a 5 fold per comune</strong>: nessun comune compare sia in
addestramento sia in validazione, il che impedisce leakage spaziale.</p>
<div class="tabwrap">{tbl(e4rs, nd=4, small=True)}</div>
{sem('''<p>Obiezione possibile: «il vostro risultato dipende dall'aver imposto una relazione
lineare; un modello più flessibile troverebbe l'effetto».</p>
<p>Per rispondere abbiamo usato algoritmi di machine learning, che non impongono alcuna
forma alla relazione e possono scoprire soglie, curve e interazioni. Il test consiste nel
misurare quanto migliora la capacità di previsione aggiungendo le sette variabili di rischio:
si passa da 0,81 a 0,89, ma il guadagno viene quasi tutto da costa, sismicità e
precipitazioni — cioè dagli stessi canali «positivi» già trovati. Alluvione, frana e incendio
contribuiscono praticamente zero.</p>
<p>Un dettaglio metodologico da citare, perché è ciò che rende la verifica credibile: la
validazione è organizzata per comune. Se lo stesso comune comparisse sia nei dati di
addestramento sia in quelli di verifica, l'algoritmo potrebbe semplicemente «ricordare» il
livello di prezzo di quel comune e sembrare bravissimo senza aver capito nulla.</p>''')}
<h3>9.9 Repricing tra i due semestri (indizio, non test)</h3>
<p>Il modello è ristimato con dipendente Δlog(prezzo) tra semestre corrente e precedente:
nessun rischio risulta associato alla variazione (tutti p &gt; 0,18 in A1; un isolato
p = 0,0499 su PGA in A2, positivo, compatibile con il caso su 14 test). Data l'immobilità del
dato (26% di celle riprezzate) il risultato è dichiarato come indizio preliminare e
<strong>non</strong> come analisi panel.</p>
"""

# ====================================================================
# 10. ESITO
# ====================================================================
res = m1[m1["spec"].isin(["A1", "A2"])].copy()
res["IC95 (% per sd)"] = res.apply(
    lambda r: f"[{100*(np.exp(r['ci95_low']*r['sd_var'])-1):+.1f}; "
              f"{100*(np.exp(r['ci95_high']*r['sd_var'])-1):+.1f}]", axis=1)
res = res.merge(r4.rename(columns={"variabile": "rischio"})[
    ["spec", "rischio", "p_holm_globale", "sopravvive_5pct"]],
    on=["spec", "rischio"], how="left")
res = res.merge(r5.rename(columns={"variabile": "rischio"})[
    ["spec", "rischio", "MDE_pct_per_1sd"]], on=["spec", "rischio"], how="left")


def lettura(r):
    if bool(r["sopravvive_5pct"]):
        return "<strong>associazione robusta</strong> (sopravvive a Holm)"
    if r["p_grezzo"] < 0.05:
        return "significativa al grezzo, cade con Holm"
    if r["MDE_pct_per_1sd"] <= 5.0:
        return "nullo <strong>informativo</strong> (MDE ≤ 5%/sd)"
    return "non conclusivo (bassa potenza)"


res["Lettura"] = res.apply(lettura, axis=1)
res["Rischio"] = res["rischio"].map(IT)
res_show = res[["spec", "Rischio", "pct_per_1sd", "IC95 (% per sd)", "p_grezzo",
                "p_holm_globale", "MDE_pct_per_1sd", "Lettura"]].rename(columns={
    "spec": "Spec.", "pct_per_1sd": "% per +1 sd", "p_grezzo": "p grezzo",
    "p_holm_globale": "p Holm (14 test)", "MDE_pct_per_1sd": "MDE %/sd"})

S["s10"] = f"""
<h2 id="s10">10. Che cosa produce questa macchina: esito in forma minima</h2>
<p>Tabella unica di sintesi: per ciascun rischio e per ciascuna delle due scale informative,
la magnitudine economica, l'incertezza, il p-value grezzo, il p corretto sulla famiglia
primaria di 14 test e la potenza disponibile. La colonna «Lettura» applica meccanicamente i
criteri dichiarati in §7.2 e §7.3, senza aggiungere interpretazione.</p>
<div class="tabwrap">{tbl(res_show, nd=3, small=True)}</div>
{sem('''<p>Come si legge una riga, per esempio la prima: nella specifica A1, passando a una
zona con un livello di rischio alluvionale superiore di una deviazione standard, il prezzo
risulta inferiore del 2,5%; ma l'intervallo di confidenza va da −5,0% a +0,1%, cioè include
lo zero, e il p-value è 0,059. Il minimum detectable effect dice che uno sconto del 3,8% o
più sarebbe stato rilevato. Conclusione: non c'è sconto, e non è per mancanza di
precisione.</p>''')}
<div class="box"><p><strong>Lettura complessiva, in quattro punti fattuali.</strong></p>
<ol>
<li>Nessuna associazione <em>negativa</em> tra rischio e quotazioni sopravvive alla
correzione per test multipli, a nessuna delle due scale.</li>
<li>Il caso dell'alluvione mostra la firma tipica del confondimento da localizzazione, non
dello sconto di prezzo: −5,1% per sd in A0 (p = 0,011) → −2,5% in A1 (p = 0,059) → −0,9% in
A2 (p = 0,50), con potenza sufficiente a rilevare effetti ≥ 3,7%.</li>
<li>Le uniche due associazioni che sopravvivono a Holm sono <em>positive</em> e a scala
tra-comuni: prossimità costiera (+14,4% per sd, trainata da Napoli, §9.3) e r99pday
(+6,0% per sd, p corretto 0,046); entrambe borderline al bootstrap di provincia (§9.2).</li>
<li>La verifica non parametrica concorda: il contributo predittivo di alluvione, frana e
incendio è ≈ 0 anche in un modello libero da vincoli di forma (§9.8).</li>
</ol></div>
{sem('''<p>La sintesi in una frase, per l'esposizione orale: <strong>nelle quotazioni
immobiliari ufficiali delle quattro province il rischio climatico e ambientale non risulta
scontato; e per il rischio più importante, quello alluvionale, siamo in grado di dimostrare
che uno sconto della dimensione documentata dalla letteratura internazionale sarebbe stato
visibile nei nostri dati.</strong> Le due associazioni che restano hanno segno positivo e si
spiegano con canali di amenità — il mare — o con gradienti territoriali, non con una
prezzatura del rischio.</p>''')}
"""

# ====================================================================
# 11. LIMITI
# ====================================================================
S["s11"] = """
<h2 id="s11">11. Che cosa questo metodo non può dire (e perché)</h2>
<p>Ogni limite qui elencato è la conseguenza diretta di una scelta o di un vincolo descritti
sopra, non una cautela generica.</p>
<ol>
<li><strong>Nessuna capitalizzazione intra-zona è osservabile</strong> — conseguenza della
granularità del dato OMI (§1): il prezzo è unico per zona, quindi qualunque differenza di
prezzo tra immobili più e meno esposti <em>dentro</em> la stessa zona è invisibile.</li>
<li><strong>I risultati non si estendono alle aree non quotate</strong> — conseguenza della
selezione campionaria (§1): il campione è il perimetro urbano quotato (38,8% delle
particelle, dal 18% di Torino all'87% di Napoli).</li>
<li><strong>I nulli su incendio e precipitazioni sono attenuati</strong> — conseguenza
dell'errore di misura da attribuzione su griglia (§2.1): l'errore spinge i coefficienti verso
zero, quindi il MDE effettivo sul rischio «vero» è più alto di quello nominale.</li>
<li><strong>I coefficienti A2 di PGA, incendio, costa e r99pday non sono
interpretabili</strong> — conseguenza della degenerazione della varianza within-comune (§5).</li>
<li><strong>L'inferenza è debole rispetto a shock di provincia</strong> — conseguenza del
disegno a 4 province (§9.2).</li>
<li><strong>Nessuna affermazione causale è sostenibile</strong> — conseguenza dell'esclusione
motivata di tutto il Livello 3 (§6.2).</li>
<li><strong>La distinzione «mercato che non prezza» vs «strumento che non riflette» resta
aperta</strong> — conseguenza dell'avere una sola fonte di prezzo (§9.5).</li>
<li><strong>Assenza di controlli socioeconomici fini</strong> (reddito di zona, qualità dei
servizi): fascia OMI, centralità e urbanizzazione li proxano solo in parte.</li>
</ol>
""" + sem("""<p>Un consiglio di impostazione per l'esposizione: questa sezione è un punto di
forza, non una debolezza. Elencare i limiti <em>e mostrare da quale scelta ciascuno
discende</em> dimostra il controllo del disegno. La formula da usare è sempre la stessa:
«questo limite deriva da quella scelta, che abbiamo fatto per questa ragione, e la sua
conseguenza sui risultati è questa».</p>""")

# ====================================================================
# 12. RIPRODUCIBILITA'
# ====================================================================
pipe = pd.DataFrame([
    ("<code>integra_rischio_incendio.py</code><br><code>integra_rischio_precipitazioni.py</code>"
     "<br><code>integra_rischio_p99.py</code>",
     "Integrazione delle griglie EFFIS ed E-OBS nel DB SQLite, con distanza di attribuzione "
     "salvata per ogni particella", "tabelle <code>parcels_rischio_*</code>", "§2.1"),
    ("<code>prepara_dataset_analisi.py</code>",
     "Pulizia, riattribuzione comune/provincia catastale, costruzione indicatori di rischio, "
     "esplosione delle voci OMI, aggregazione all'unità statistica",
     "<code>output/particelle.csv</code>, <code>output/celle.csv</code>", "§3.1–3.2"),
    ("<code>analisi_esplorativa.py</code>",
     "EDA, copertura e selezione campionaria, distribuzioni, missing, tabella fonti",
     "<code>output/eda/</code>", "§1, §2.1, §3.4–3.5"),
    ("<code>modelli_livello1.py</code>",
     "Specifiche headline A0/A1/A2/A2w, quota di varianza within, diagnostica completa, "
     "decisioni di attivazione di Livello 2",
     "<code>output/modelli/</code>", "§4, §5, §6.1, §8"),
    ("<code>robustezza.py</code>",
     "Varianti di specifica, leave-one-provincia-out, wild cluster bootstrap, Holm globale, "
     "MDE, placebo permutazionali, curva di specificazione",
     "<code>output/robustezza/</code>", "§7.2–7.3, §9.1–9.4"),
    ("<code>estensioni_livello2.py</code>",
     "Forme non lineari, eterogeneità per tipologia, machine learning, repricing, canoni",
     "<code>output/estensioni/</code>", "§9.5–9.9"),
    ("<code>report_finale.py</code> / <code>report_metodo.py</code>",
     "Report completo e questa versione ridotta orientata al metodo",
     "<code>output/report_finale.html</code>, <code>output/report_metodo.html</code>", "—"),
], columns=["Script", "Cosa fa", "Output", "Sezioni di questo documento"])

S["s12"] = f"""
<h2 id="s12">12. Riproducibilità: mappa esatta script → output → sezione</h2>
<p>L'intera analisi è riproducibile dai dati grezzi eseguendo la pipeline nell'ordine
indicato (Python 3.13; pandas, statsmodels, patsy, libpysal/esda/spreg, scikit-learn, shap,
xarray). I semi casuali sono fissati per bootstrap, permutazioni e machine learning.</p>
<div class="tabwrap">{tbl(pipe, small=True)}</div>
<p>Ogni numero citato in questo documento è tracciabile a un file CSV della cartella
<code>output/</code> e alla specifica che lo ha generato. Le statistiche diagnostiche non
salvate in CSV (Breusch-Pagan, Jarque-Bera, RESET, Moran, ICC) sono nel log completo
<code>output/modelli/modelli_summary.txt</code>; i test placebo in
<code>output/robustezza/r6_placebo.txt</code>. Dati di prezzo: download API 09–11/07/2026.</p>
"""

# ====================================================================
# APPENDICE A — GLOSSARIO
# ====================================================================
S["appA"] = """
<h2 id="appA">Appendice A — Glossario minimo</h2>
<p>I termini tecnici usati nel documento, ciascuno in una riga di linguaggio comune.</p>
<dl class="gloss">
<dt>Cella (unità statistica)</dt>
<dd>Il gruppo di immobili che nel dato OMI condividono per forza lo stesso prezzo: stessa
zona, stesso comune, stessa tipologia, stesso stato conservativo. È la riga del nostro
dataset.</dd>
<dt>Coefficiente (β)</dt>
<dd>Il numero che dice di quanto si muove il prezzo quando il rischio aumenta, tenendo fermo
tutto il resto.</dd>
<dt>Deviazione standard (sd)</dt>
<dd>Di quanto una grandezza varia tipicamente da un caso all'altro. Usata come unità di
misura comune per confrontare rischi diversi.</dd>
<dt>Intervallo di confidenza al 95%</dt>
<dd>L'intervallo di valori compatibili con i dati. Se contiene lo zero, non si può escludere
che l'effetto sia nullo.</dd>
<dt>p-value</dt>
<dd>La probabilità di osservare un risultato come il nostro se in realtà l'effetto non
esistesse. Piccolo = difficile da spiegare col caso.</dd>
<dt>Standard error</dt>
<dd>La misura di quanto la stima è incerta. Più piccolo, più la stima è precisa.</dd>
<dt>Cluster (standard error clusterizzati)</dt>
<dd>Modo di calcolare l'incertezza che tiene conto del fatto che osservazioni dello stesso
gruppo (qui: dello stesso comune) non portano informazione del tutto nuova.</dd>
<dt>Effetto fisso</dt>
<dd>Una variabile che assegna a ciascun comune (o tipologia, o fascia) il suo livello
proprio, così che il confronto avvenga solo <em>dentro</em> quel gruppo.</dd>
<dt>Within / between</dt>
<dd>«Within-comune» = confronto tra zone dello stesso comune. «Between» o «tra comuni» =
confronto tra comuni diversi.</dd>
<dt>R²</dt>
<dd>La quota di variabilità dei prezzi che il modello riesce a riprodurre. Non è una misura
di correttezza: sale automaticamente aggiungendo variabili.</dd>
<dt>Eteroschedasticità</dt>
<dd>Quando l'errore del modello è più grande in alcune zone che in altre.</dd>
<dt>VIF</dt>
<dd>Indice che segnala se due o più variabili esplicative sono così simili tra loro da non
poter essere distinte.</dd>
<dt>Test RESET</dt>
<dd>Controlla se la forma dell'equazione (in questo caso una retta sui logaritmi) è adeguata
ai dati.</dd>
<dt>Moran's I</dt>
<dd>Misura se zone vicine geograficamente tendono ad avere errori simili, cioè se il modello
sta trascurando qualcosa di geografico.</dd>
<dt>Correzione di Holm</dt>
<dd>Alza la soglia di significatività in proporzione al numero di test svolti, per evitare
falsi positivi da tentativi ripetuti.</dd>
<dt>Potenza statistica e MDE</dt>
<dd>La capacità del disegno di vedere un effetto se esiste; il MDE è l'effetto più piccolo
che saremmo stati in grado di rilevare.</dd>
<dt>Test placebo</dt>
<dd>Si dà al modello una variabile volutamente falsa: se «trova» un effetto, la procedura ha
un problema.</dd>
<dt>Bootstrap</dt>
<dd>Simulazione al computer di migliaia di campioni possibili, usata quando le formule
statistiche standard non sono affidabili.</dd>
<dt>GroupKFold</dt>
<dd>Modo di validare un modello di machine learning in cui gli stessi comuni non compaiono
mai sia in addestramento sia in verifica.</dd>
<dt>Pseudo-replicazione</dt>
<dd>Contare più volte la stessa informazione come se fossero osservazioni indipendenti.</dd>
</dl>
"""

# ====================================================================
# APPENDICE B — DOMANDE PREVEDIBILI
# ====================================================================
QA = [
    ("Perché una regressione OLS e non un modello spaziale, trattandosi di dati geografici?",
     "Perché la diagnostica dice che non serve. Il test di Moran sui residui, calcolato solo "
     "tra zone di comuni diversi per evitare l'artefatto di costruzione, dà p = 0,42: la "
     "dipendenza spaziale è interamente assorbita dagli effetti fissi comunali. Stimare un "
     "SAR o un SEM avrebbe aggiunto complessità senza correggere nulla, contro il principio "
     "di parsimonia dichiarato all'inizio."),
    ("Perché il logaritmo del prezzo?",
     "Per due ragioni: la distribuzione dei prezzi in euro è fortemente asimmetrica "
     "(skewness +2,6) mentre in logaritmi è simmetrica (−0,07); e il coefficiente diventa "
     "leggibile in percentuale, confrontabile tra mercati con livelli di prezzo diversi."),
    ("Perché gli standard error sono clusterizzati al comune e non alla zona o alla provincia?",
     "Perché il comune è il livello a cui i dati sono stati campionati (100 particelle per "
     "comune) ed è più grosso della cella, cioè del livello a cui variano i rischi. La zona "
     "sarebbe troppo fine, la provincia ha solo 4 cluster: quest'ultima è comunque verificata "
     "a parte con il wild cluster bootstrap, e l'esito è riportato."),
    ("Il clustering non vi fa perdere granularità?",
     "No: il clustering non aggrega, non media e non elimina nessuna osservazione. Il "
     "campione resta di 2.728 celle e i coefficienti sono numericamente identici. Cambia solo "
     "il calcolo dell'incertezza. La granularità si perde altrove — nel fatto che il prezzo "
     "OMI è unico per zona — ed è un limite del dato, non dell'estimatore."),
    ("Un R² di 0,96 non è un segno di sovradattamento?",
     "Il salto da 0,86 a 0,96 è meccanico: deriva dall'aggiunta di 344 variabili che "
     "identificano ciascun comune. Non misura la qualità della relazione rischio-prezzo. "
     "Proprio per questo riportiamo accanto il numero di parametri (371 su 2.728 osservazioni) "
     "e, soprattutto, la quota di varianza di ciascun rischio che sopravvive within-comune."),
    ("Come potete affermare che l'effetto non c'è, invece di dire che non l'avete trovato?",
     "Grazie al calcolo della potenza. Per l'alluvione il minimum detectable effect è 3,7–3,8%: "
     "significa che uno sconto di quella dimensione o superiore sarebbe stato rilevato con "
     "l'80% di probabilità. Poiché la letteratura documenta sconti tipicamente tra il 2% e "
     "l'8%, gran parte di quell'intervallo era alla nostra portata. È un nullo informativo, "
     "non un'assenza di evidenza."),
    ("Non è sospetto usare A1 per alcuni rischi e A2 per altri? Sembra una scelta a posteriori.",
     "La regola è dichiarata prima e non guarda i p-value: si sceglie la specifica in base "
     "alla quota di varianza del rischio che sopravvive agli effetti fissi comunali, cioè in "
     "base a quanta informazione esiste. La PGA ha 0,0% di varianza within: il suo coefficiente "
     "A2 sarebbe rumore, e infatti ha un intervallo di ±20 punti percentuali."),
    ("L'associazione positiva con le precipitazioni estreme non è controintuitiva?",
     "Sì, ed è il motivo per cui non la interpretiamo come premio al rischio. L'indicatore "
     "r99pday è relativo per costruzione (giorni oltre il 99° percentile locale), e la sua "
     "variazione territoriale segue gradienti collinari e pedemontani che sono anche di "
     "pregio. La variante a soglia assoluta rr20mm conferma il segno, quindi non è un "
     "artefatto della normalizzazione; ma resta un'associazione, non un meccanismo."),
    ("Perché solo quattro province?",
     "È il perimetro dei dati disponibili, e le quattro coprono mercati molto diversi. Il "
     "limite non è nella stima ma nell'inferenza rispetto a shock che colpiscano un'intera "
     "provincia: con 4 cluster il wild cluster bootstrap rende tutte le associazioni "
     "borderline, e lo dichiariamo come limite del disegno."),
    ("Perché non avete usato i prezzi delle transazioni reali?",
     "Non erano disponibili in forma georeferenziata. È il limite principale dello studio, ed "
     "è anche ciò che definisce l'oggetto: stiamo misurando se lo strumento di valutazione "
     "ufficiale riflette il rischio — domanda rilevante per banche e periti — non se lo "
     "riflette il mercato."),
    ("Perché non avete costruito l'indice sintetico previsto come obiettivo finale?",
     "Perché mancava la premessa. Un indice di capitalizzazione richiede pesi «rivelati dai "
     "prezzi»: se i prezzi non scontano i rischi, quei pesi non esistono, e costruirlo sui "
     "due coefficienti positivi significherebbe sintetizzare amenità, non rischio. Resta "
     "invece percorribile un indice di esposizione fisica, con pesi normativi espliciti, "
     "a scala di zona o comune."),
]
S["appB"] = ("""
<h2 id="appB">Appendice B — Domande prevedibili in sede di discussione</h2>
<p>Undici obiezioni ricorrenti su questo disegno di ricerca, con la risposta ancorata al
passaggio del documento che la sostiene. Appendice di preparazione: può essere rimossa dalla
versione consegnata.</p>
""" + "".join(f'<div class="qa"><span class="q">{q}</span><p>{a}</p></div>'
               for q, a in QA))

# ====================================================================
TOC = """
<nav class="toc"><strong>Indice</strong><br>
<a href="#s0">0. Che cosa è questo documento</a><br>
<a href="#s1">1. L'oggetto misurato</a><br>
<a href="#s2">2. Le variabili di rischio</a><br>
<a href="#s3">3. Dal dato grezzo al campione</a><br>
<a href="#s4">4. Equazioni e identificazione</a><br>
<a href="#s5">5. Scala di identificazione</a><br>
<a href="#s6">6. Regola di decisione</a><br>
<a href="#s7">7. Come si legge ogni numero</a><br>
<a href="#s8">8. Diagnostica</a><br>
<a href="#s9">9. Batteria di robustezza</a><br>
<a href="#s10">10. Esito in forma minima</a><br>
<a href="#s11">11. Limiti del metodo</a><br>
<a href="#s12">12. Riproducibilità</a><br>
<a href="#appA">App. A — Glossario</a><br>
<a href="#appB">App. B — Domande prevedibili</a>
</nav>
"""

html = f"""<!DOCTYPE html>
<html lang="it"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Come è stata condotta l'analisi — versione metodologica del report</title>
<style>{CSS}</style></head><body><main>
<h1>Come è stata condotta l'analisi<br>
<span style="font-size:1.05rem;color:#444;">Versione ridotta e orientata al metodo del
report su rischio climatico-ambientale e quotazioni OMI (Milano, Napoli, Roma, Torino)</span></h1>
<p class="meta">Documento di accompagnamento a <code>output/report_finale.html</code> —
stessi dati, stessi output, nessun risultato nuovo: cambia solo ciò che viene messo in primo
piano. Ogni passaggio tecnico è affiancato da una spiegazione in linguaggio accessibile
(riquadri color ambra), pensata per l'esposizione orale. {pd.Timestamp.today():%d/%m/%Y}</p>
{TOC}
{S["s0"]}{S["s1"]}{S["s2"]}{S["s3"]}{S["s4"]}{S["s5"]}{S["s6"]}
{S["s7"]}{S["s8"]}{S["s9"]}{S["s10"]}{S["s11"]}{S["s12"]}
{S["appA"]}{S["appB"]}
</main></body></html>
"""

with open(DEST, "w", encoding="utf-8") as f:
    f.write(html)
print(f"[OK] Report metodologico scritto: {DEST}  "
      f"({os.path.getsize(DEST)/1e6:.2f} MB)")
