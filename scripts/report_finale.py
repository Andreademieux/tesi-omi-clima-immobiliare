#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tranche 4 - Report finale HTML (protocollo S21, conclusioni S22).
Legge gli output di output/eda, output/modelli, output/robustezza,
output/estensioni e genera un unico documento: output/report_finale.html.

Uso:  py report_finale.py
"""
import base64
import os

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")
EDA = os.path.join(OUT, "eda")
MOD = os.path.join(OUT, "modelli")
ROB = os.path.join(OUT, "robustezza")
EST = os.path.join(OUT, "estensioni")
DEST = os.path.join(OUT, "report_finale.html")

IT = dict(  # etichette leggibili
    flood_ord="Alluvione (classe PGRA, ordinale)",
    landslide_ord="Frana (classe PAI, ordinale)",
    pga="Sisma (PGA)", sub_sink="Subsidenza (mm/anno, sprofond.)",
    fire_score="Incendio (EFFIS)", r99pday="Precip. estreme (r99pday)",
    is_coastal_zone="Erosione costiera (quota zona costiera)")


def b64img(path, alt, cap):
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return (f'<figure><img src="data:image/png;base64,{data}" alt="{alt}">'
            f'<figcaption>{cap}</figcaption></figure>')


def tbl(df, nd=3, index=False, small=False):
    cls = "tab small" if small else "tab"
    return df.to_html(index=index, classes=cls, border=0,
                      float_format=lambda x: f"{x:,.{nd}f}", na_rep="—")


S = {}  # sezioni html

# ====================================================================
# Carica risultati
# ====================================================================
m1 = pd.read_csv(os.path.join(MOD, "m1_coefficienti.csv"))
m2 = pd.read_csv(os.path.join(MOD, "m2_fit.csv"))
m3 = pd.read_csv(os.path.join(MOD, "m3_within_variance_share.csv"),
                 names=["variabile", "quota_within"], header=0)
m4 = pd.read_csv(os.path.join(MOD, "m4_vif.csv"), index_col=0)
r2a = pd.read_csv(os.path.join(ROB, "r2_lopo_a1.csv"))
r2b = pd.read_csv(os.path.join(ROB, "r2_lopo_a2.csv"))
r3 = pd.read_csv(os.path.join(ROB, "r3_wcb_provincia.csv"))
r4 = pd.read_csv(os.path.join(ROB, "r4_holm_globale.csv"))
r5 = pd.read_csv(os.path.join(ROB, "r5_mde.csv"))
r7 = pd.read_csv(os.path.join(ROB, "r7_curva_specificazione.csv"))
e2 = pd.read_csv(os.path.join(EST, "e2_classi_pericolosita.csv"))
e3 = pd.read_csv(os.path.join(EST, "e3_interazioni_tipologia.csv"))
e4r = pd.read_csv(os.path.join(EST, "e4_ml_r2.csv"))
e5 = pd.read_csv(os.path.join(EST, "e5_repricing.csv"))
e6 = pd.read_csv(os.path.join(EST, "e6_canoni.csv"))
t11 = pd.read_csv(os.path.join(EDA, "t11_fonti_rischio.csv"))
t1 = pd.read_csv(os.path.join(EDA, "t1_struttura_province.csv"))
t3 = pd.read_csv(os.path.join(EDA, "t3_copertura_omi_provincia.csv"))
t4 = pd.read_csv(os.path.join(EDA, "t4_selection_smd.csv"))
t5 = pd.read_csv(os.path.join(EDA, "t5_selection_logit.csv"))
t7 = pd.read_csv(os.path.join(EDA, "t7_rischi_descrittive.csv"), index_col=0)

# Holm sul rapporto canone/prezzo (famiglia per dip x spec)
e6h = []
for (dip, spec), g in e6.groupby(["dip", "spec"]):
    g = g.copy()
    g["p_holm"] = multipletests(g["p"], method="holm")[1]
    e6h.append(g)
e6 = pd.concat(e6h)

# ====================================================================
CSS = """
:root { --ink:#1a1a1a; --acc:#0b5563; --line:#d8d8d8; --bg:#ffffff; --soft:#f4f6f7; }
* { box-sizing:border-box; }
body { font-family: Georgia, 'Times New Roman', serif; color:var(--ink);
       background:var(--bg); margin:0; line-height:1.55; }
main { max-width: 980px; margin: 0 auto; padding: 2rem 1.4rem 5rem; }
h1 { font-size:1.75rem; color:var(--acc); line-height:1.25; margin:.2rem 0 .4rem;}
h2 { font-size:1.28rem; color:var(--acc); border-bottom:2px solid var(--acc);
     padding-bottom:.25rem; margin-top:2.6rem; }
h3 { font-size:1.05rem; margin-top:1.6rem; color:#333; }
p, li { font-size:.95rem; text-align:justify; }
.meta { color:#555; font-style:italic; margin-bottom:1.4rem; }
.abstract { background:var(--soft); border-left:4px solid var(--acc);
            padding:.9rem 1.1rem; }
.eq { font-family:'Cambria Math', Georgia, serif; background:var(--soft);
      padding:.65rem .9rem; margin:.7rem 0; border-radius:4px;
      overflow-x:auto; font-size:.92rem; }
.tab { border-collapse:collapse; margin:.8rem auto; font-size:.8rem;
       font-family: 'Segoe UI', Arial, sans-serif; }
.tab th { background:var(--acc); color:#fff; padding:.3rem .5rem;
          text-align:center; font-weight:600;}
.tab td { border-bottom:1px solid var(--line); padding:.28rem .5rem;
          text-align:right; }
.tab td:first-child, .tab th:first-child { text-align:left; }
.tab.small { font-size:.72rem; }
.tabwrap { overflow-x:auto; }
figure { margin:1.1rem 0; text-align:center; }
figure img { max-width:100%; border:1px solid var(--line); }
figcaption { font-size:.8rem; color:#555; margin-top:.35rem; }
.box { border:1px solid var(--acc); border-radius:6px; padding: .8rem 1.1rem;
       margin: 1rem 0; background: #f7fbfc; }
.warn { border-left:4px solid #a33; background:#fdf6f6; padding:.7rem 1rem;
        margin:.8rem 0; font-size:.92rem;}
.ev-forte { color:#0a6b31; font-weight:700; }
.ev-moderata { color:#8a6d00; font-weight:700; }
.ev-debole { color:#a35b00; font-weight:700; }
.ev-inconclusiva { color:#8a8a8a; font-weight:700; }
code { background:var(--soft); font-size:.85em; padding:.05rem .3rem; }
ol.concl > li { margin-bottom: .9rem; }
.toc { background:var(--soft); padding:.8rem 1.2rem; border-radius:6px;
       font-size:.88rem; column-count:2; }
.toc a { color:var(--acc); text-decoration:none; }
@media (max-width:700px){ .toc{column-count:1;} }
@media print { h2 { page-break-after:avoid; } figure { page-break-inside:avoid; } }
"""

# ====================================================================
# 1. ABSTRACT
# ====================================================================
S["s1"] = """
<h2 id="s1">1. Abstract e sintesi dei risultati principali</h2>
<div class="abstract">
<p><strong>Obiettivo.</strong> Lo studio verifica quanto il rischio climatico e ambientale
risulti già scontato (capitalizzato) nelle quotazioni immobiliari ufficiali OMI delle
province di Milano, Napoli, Roma e Torino, su sette famiglie di rischio (alluvione PGRA,
frana PAI, sisma, subsidenza, incendio EFFIS, precipitazioni estreme E-OBS, erosione
costiera), con un protocollo a livelli guidato dalla diagnostica e improntato alla parsimonia.</p>
<p><strong>Dati e disegno.</strong> 65.618 particelle catastali (658 comuni) con quotazioni
OMI per zona × tipologia × stato, aggregate all'unità statistica corretta
(provincia × comune × zona OMI × tipologia × condizione × fascia: 2.793 celle, campione di
stima 2.728) per neutralizzare la pseudo-replicazione del prezzo di zona. Modello log-lineare
con saturazione crescente degli effetti fissi geografici (A0 → A1 provincia+amenità →
A2 within-comune) e standard error clusterizzati per comune (345 cluster).</p>
<p><strong>Risultato principale.</strong> Non emerge alcuna capitalizzazione
<em>negativa</em> robusta del rischio nelle quotazioni OMI. L'associazione negativa
"naive" dell'alluvione (−5,1% per +1 sd in A0, p=0,011) si riduce a −2,5% (p=0,059) con
controlli di localizzazione e si annulla within-comune (−0,9%, p=0,50), con potenza
adeguata a rilevare effetti ≥3,7% (nullo informativo). Dopo correzione di Holm sull'intera
famiglia dei test sopravvivono soltanto due associazioni <em>positive</em> a scala
tra-comuni: la prossimità costiera (+14,4% per sd, amenità che domina il rischio erosivo,
guidata dalla provincia di Napoli) e le precipitazioni estreme r99pday (+6,0% per sd,
p corretto = 0,046). Il machine learning con validazione GroupKFold per comune conferma in
modo indipendente: il contributo predittivo di alluvione, frana e incendio è ≈0.</p>
<p><strong>Lettura.</strong> Poiché le quotazioni OMI sono stime amministrative per zona
(non prezzi di transazione), l'assenza di sconto è compatibile sia con la mancata
prezzatura da parte del mercato sia con il lisciamento amministrativo dello strumento;
l'evidenza indiziaria (canoni con gli stessi pattern dei prezzi, repricing semestrale
nullo rispetto al rischio, letteratura sulla salienza informativa) orienta verso un
contributo rilevante della seconda componente, senza poterla isolare. La costruzione di
un indice sintetico <em>di capitalizzazione</em> risulta prematura; è invece percorribile
un indice di <em>esposizione</em> a scala comunale/di zona.</p>
</div>
"""

# ====================================================================
# 2. INTRODUZIONE
# ====================================================================
S["s2"] = """
<h2 id="s2">2. Introduzione e motivazione dello studio</h2>
<p>La letteratura internazionale documenta in modo crescente che i rischi fisici legati al
clima possono riflettersi nei valori immobiliari, ma anche che tale capitalizzazione è
condizionata da informazione, salienza degli eventi e caratteristiche istituzionali del
mercato (§3). Per l'Italia — Paese ad alta esposizione idrogeologica e sismica e a bassa
penetrazione assicurativa sui rischi catastrofali — la domanda operativa è duplice:
(i) il mercato sconta i rischi climatico-ambientali? (ii) lo strumento di valutazione
ufficiale più usato nelle prassi amministrative, creditizie ed estimative (le quotazioni
OMI dell'Agenzia delle Entrate) li riflette?</p>
<p>Questo studio affronta la seconda domanda in modo diretto e la prima in modo indiretto,
stimando quanto sette famiglie di rischio siano associate ai livelli delle quotazioni OMI
in quattro grandi mercati provinciali (Milano, Napoli, Roma, Torino), con un protocollo
metodologico esplicito: (a) parsimonia — il modello più semplice che superi la diagnostica;
(b) gerarchia a tre livelli di tecniche, attivate solo dalla diagnostica; (c) standard di
reporting completo per ogni coefficiente; (d) distinzione sistematica tra assenza di
evidenza ed evidenza di assenza tramite potenza statistica; (e) linguaggio non causale:
tutte le stime sono associazioni condizionate, non effetti causali, in assenza di una
strategia identificativa di Livello 3 (§9).</p>
<p>Un contributo specifico del disegno è la valutazione esplicita della <em>scala
geografica</em> della capitalizzazione: la saturazione progressiva degli effetti fissi
(nessuno → provincia → comune) consente di distinguere associazioni tra-comuni da
associazioni within-comune, e la decomposizione della varianza dei rischi rivela per quali
rischi ciascuna scala sia effettivamente informativa (§8.3).</p>
"""

# ====================================================================
# 3. LETTERATURA
# ====================================================================
S["s3"] = """
<h2 id="s3">3. Contesto di letteratura</h2>
<p>La rassegna si basa sul corpus di lavori disponibile nella cartella di progetto,
vagliato integralmente; i riferimenti citati sono quindi verificabili nei file allegati
alla tesi. Quattro filoni sono direttamente rilevanti.</p>
<h3>3.1 Survey su clima e real estate</h3>
<p>Contat, Hopkins, Mejia e Suandi (2024, <em>Real Estate Economics</em>) passano in
rassegna prezzi, performance dei mutui e migrazioni in relazione ad alluvioni, incendi e
innalzamento del livello del mare: gli sconti di prezzo per il rischio alluvionale sono
frequenti ma eterogenei, spesso concentrati dopo eventi salienti o in presenza di obblighi
informativi. Clayton, Devaney, Sayce e Van de Wetering (2021, <em>Journal of Portfolio
Management</em>) rilevano che per il commercial real estate l'evidenza è più scarsa e che
valutazioni e prassi estimative tendono a incorporare il rischio climatico con ritardo
rispetto ai mercati delle transazioni — punto direttamente pertinente per uno strumento
amministrativo come l'OMI. Giglio, Kelly e Stroebel (2021, <em>Annual Review of Financial
Economics</em>) inquadrano il real estate tra le asset class in cui la prezzatura del
rischio climatico è documentata ma parziale. Warren-Myers e Hurlimann (2020) sottolineano
il divario tra conoscenza scientifica dei rischi e prassi valutative del settore.</p>
<h3>3.2 Alluvione, informazione e salienza</h3>
<p>Hino e Burke (2021, <em>PNAS</em>) mostrano che negli USA lo sconto per il rischio
alluvionale dipende in modo cruciale dall'informazione: dove la mappatura regolatoria è
poco saliente, lo sconto è piccolo o nullo. Baldauf, Garlappi e Yannelis (2020, <em>Review
of Financial Studies</em>) trovano che lo sconto dipende dalle credenze locali sul
cambiamento climatico. Questi risultati forniscono la griglia interpretativa per un esito
nullo in un contesto — quello italiano — privo di obbligo sistematico di disclosure del
rischio nelle compravendite.</p>
<h3>3.3 Rischio sismico e geologico</h3>
<p>Ikefuji, Laeven, Magnus e Yue (2022, <em>JASA</em>) documentano che il rischio sismico
è incorporato nei prezzi in cinque città giapponesi, in un contesto ad altissima salienza;
Tu, Andersson, Shyr e Lin (2023, <em>Applied Spatial Analysis and Policy</em>) trovano
sconti per zone sismiche e alluvionali a Taichung; Shi e Naylor (2023, <em>Journal of
Housing and the Built Environment</em>) mostrano, sul terremoto di Canterbury, che la
percezione del rischio si aggiorna (e si capitalizza) dopo l'evento. Zubizarreta e Aquino
(2026, preprint) offrono una rassegna sistematica su esposizione immobiliare e rischi
sismici/di sottosuolo. Hernández, Luna e Madeira (2022, Banco Central de Chile) trovano
effetti di temperatura e precipitazioni sui prezzi cileni usando prezzi di transazione.</p>
<h3>3.4 Esiti nulli, canale assicurativo-creditizio e quadro istituzionale</h3>
<p>Miłuch e Kopczewska (2024, <em>Environmental Science and Pollution Research</em>) — un
esito nullo metodologicamente vicino al nostro: l'inquinamento atmosferico non risulta nei
prezzi di Varsavia, attribuito a limiti informativi e bassa variabilità intra-urbana;
Sager e Singer (2025, <em>AEJ: Economic Policy</em>) mostrano al contrario che con
variazione regolatoria ben identificata i danni da inquinamento capitalizzati risultano
maggiori di quanto stimato in precedenza; Cvijanović, Rolheiser e Van de Minne (2023)
trovano effetti su valori e redditi del commercial real estate. Sul canale
assicurativo-creditizio: Buschbom, Eastman, Wang e Zhou (2025) documentano il repricing
assicurativo del rischio climatico nel commercial real estate USA; Taylor (2020) analizza
la securitizzazione del rischio catastrofale in Florida; Feng, Lu-Andrews e Wu (2022) e il
lavoro su beta e hazard exposure dei REIT (<em>JREFE</em>, 2026) mostrano la trasmissione
ai veicoli quotati; i lavori su NPL e rischio climatico (Fan et al. 2024; Khemiri e
Nouaili 2025; Di Febo, Angelini e Le 2024) e su disclosure (TCFD 2021; Tumewang, Ntim e
Haque 2025; Matsumura, Prakash e Vera-Muñoz 2022) delineano il contesto di vigilanza; il
rapporto EIOPA-ECB (2024) sul <em>protection gap</em> assicurativo europeo è centrale per
l'Italia, dove la scarsa copertura catastrofale riduce un canale (i premi) attraverso cui
il rischio diventa saliente e prezzabile. NGFS (2020) fornisce il quadro di scenario.</p>
<div class="box"><p><strong>Posizionamento.</strong> Rispetto a questa letteratura, lo
studio: (i) usa quotazioni <em>amministrative</em> e non prezzi di transazione — ciò che
misura è la capitalizzazione nello strumento ufficiale, non nel mercato; (ii) trova
un'assenza di sconto coerente con i contesti a bassa salienza informativa (Hino-Burke;
Miłuch-Kopczewska) e con il ritardo valutativo documentato per gli strumenti estimativi
(Clayton et al.); (iii) aggiunge una decomposizione esplicita della scala geografica di
identificazione, raramente presente nei lavori esaminati.</p></div>
"""

# ====================================================================
# 4. DATABASE
# ====================================================================
fonti_html = tbl(t11, index=False, small=True)
t3f = t3.rename(columns={t3.columns[0]: "provincia"})
t3f["quota_con_omi"] = (100 * t3f["quota_con_omi"]).round(1)
S["s4"] = f"""
<h2 id="s4">4. Database, natura del dato di prezzo e copertura campionaria</h2>
<h3>4.1 Struttura del database</h3>
<p>Il database (SQLite, API Zornade) contiene 65.800 particelle catastali campionate in
658 comuni delle quattro province (100 particelle per comune), con: geometria e land
cover; rischi alla particella (perimetri PGRA a tre classi di probabilità LPH/MPH/HPH,
perimetri PAI P1–P4, zona sismica e PGA, subsidenza interferometrica, erosione costiera);
quotazioni OMI della zona di appartenenza per tipologia edilizia e stato conservativo
(intervallo min–max €/m², acquisto e locazione, semestre corrente e precedente). A queste
si aggiungono, integrate per questo studio su griglia: il rischio incendio EFFIS
(prevalenze delle classi di rischio, ~12 km) e due indicatori di precipitazioni estreme
E-OBS a 0,1° (r99pday, headline; rr20mm, robustezza), come da tabella fonti.</p>
<p>La provincia e il comune di appartenenza sono stati riattribuiti dal
<code>municipality</code> catastale (182 particelle raccolte per errore fuori dalle 4
province escluse; 2.290 riattribuzioni di comune within-provincia). Dataset pulito:
65.618 particelle.</p>
<div class="tabwrap">{fonti_html}</div>
<h3>4.2 Natura amministrativa del dato di prezzo (punto centrale, non nota a margine)</h3>
<div class="warn"><p><strong>Le quotazioni OMI non sono prezzi di transazione.</strong>
Sono intervalli di valutazione (€/m²) pubblicati semestralmente dall'Osservatorio del
Mercato Immobiliare dell'Agenzia delle Entrate per zona OMI × tipologia × stato
conservativo, costruiti da fonti miste (atti, perizie, expert opinion) con criteri
amministrativi di continuità e prudenza. Ne discendono tre proprietà che vincolano
l'interpretazione: (i) <em>granularità</em>: il prezzo è assegnato alla zona, non
all'immobile — ogni variazione intra-zona del rischio è per costruzione non prezzabile
nel dato; (ii) <em>lisciamento temporale</em>: tra i due semestri disponibili varia il
prezzo solo nel 26,2% delle celle (Δlog medio +0,6%); (iii) <em>vintage</em>: dati
scaricati il 9–11/07/2026, riferiti all'ultimo semestre OMI pubblicato con confronto sul
semestre precedente. Ogni risultato del presente studio risponde quindi alla domanda
"lo strumento ufficiale riflette il rischio?", e solo indirettamente alla domanda
"il mercato prezza il rischio?" — la distinzione è ripresa in §12 e §15 (conclusione 7).</p></div>
<p>Il prezzo analizzato è il punto medio dell'intervallo di acquisto; i canoni di
locazione (disponibili per il 100% delle celle) sono usati come confronto in §12.4.</p>
<h3>4.3 Copertura campionaria e selezione (§3 del protocollo)</h3>
<p>Ha almeno una quotazione OMI il 38,8% delle particelle (25.468 su 65.618), con forte
eterogeneità tra province e un driver dominante: l'urbanizzazione (92,7% di copertura tra
le particelle in land cover artificiale contro 25,2% altrove).</p>
{tbl(t3f, nd=1)}
<p>Le differenze standardizzate (SMD) tra particelle con e senza quotazione mostrano
squilibri non trascurabili su sisma (zona sismica −0,29), precipitazioni (r99pday −0,37;
rr20mm −0,32) e costa (+0,35). In un logit descrittivo P(quotazione) ~ rischi
standardizzati + FE provincia (SE cluster comune), l'urbanizzazione domina (OR 4,15);
a parità di provincia le aree con più precipitazioni estreme risultano lievemente meno
coperte (OR 0,78 per +1 sd, p=0,023).</p>
<div class="tabwrap">{tbl(t4, nd=3)}</div>
<p><strong>Implicazione.</strong> Il campione di stima rappresenta il perimetro
urbano/quotato, non l'intero territorio: i risultati non sono estrapolabili alle aree non
quotate (tendenzialmente extraurbane e, per r99pday, lievemente più esposte). Non è un
bias per la domanda dello studio — che riguarda le quotazioni esistenti — ma delimita la
validità esterna (§13).</p>
"""

# ====================================================================
# 5. UNITA' STATISTICA
# ====================================================================
S["s5"] = """
<h2 id="s5">5. Unità statistica e trattamento della pseudo-replicazione</h2>
<p>Poiché il prezzo OMI è assegnato per zona × tipologia × stato, trattare le 25.468
particelle quotate (o le 212.417 righe particella × voce) come osservazioni indipendenti
produrrebbe pseudo-replicazione massiva: la deviazione standard del prezzo entro il
raggruppamento comune × zona × tipologia × condizione è esattamente 0 per il 99,1% delle
celle (l'0,9% residuo deriva dalla compresenza di più microzone nella stessa zona; il
prezzo di cella è la media). L'analisi primaria è quindi condotta sull'unità aggregata
corretta:</p>
<p class="eq">cella&nbsp;c &nbsp;=&nbsp; provincia × comune × zona OMI × tipologia ×
condizione × fascia &nbsp;&nbsp;(N = 2.793; campione di stima N = 2.728)</p>
<p>Le variabili di rischio della cella sono medie sulle particelle sottostanti (1–142 per
cella, mediana 70); gli standard error sono clusterizzati al comune (345 cluster nel
campione di stima), livello amministrativo pertinente e conservativo rispetto alla zona.
Le analisi a livello di particella restano confinate a usi esplorativi (EDA, §6) e di
sensitivity. La numerosità diseguale delle celle motiva la doppia stima non pesata /
pesata per numero di particelle (§7.2): l'osservazione è la quotazione amministrativa,
per cui la versione non pesata è la headline e quella pesata è riportata come confronto
(risultati quasi identici, §8).</p>
"""

# ====================================================================
# 6. EDA
# ====================================================================
f1 = b64img(os.path.join(EDA, "f1_distribuzione_prezzi.png"), "distribuzione prezzi",
            "Fig. 1 — Distribuzione dei prezzi OMI per provincia, livelli e log.")
f2 = b64img(os.path.join(EDA, "f2_corr_rischi.png"), "correlazioni",
            "Fig. 2 — Correlazioni di Spearman tra variabili di rischio e localizzazione (celle).")
f4 = b64img(os.path.join(EDA, "f4_mappa_prezzi.png"), "mappa",
            "Fig. 3 — Posizione delle celle e log-prezzo, per provincia.")
t7f = t7.reset_index().rename(columns={"index": "variabile"})
S["s6"] = f"""
<h2 id="s6">6. Analisi esplorativa dei dati e trattamento dei dati mancanti</h2>
<p>Il prezzo di cella ha mediana 1.085 €/m² (p1 270, p99 3.500, max 9.450 — Capri); la
distribuzione in livelli è fortemente asimmetrica (skewness +2,60) mentre il logaritmo è
quasi simmetrico (skewness −0,07, curtosi in eccesso −0,32), coerentemente con la forma
log-lineare adottata (§7) e successivamente sottoposta a test formale (§8.4).</p>
{f1}
<p>L'esposizione ai rischi è eterogenea tra province: è in perimetro alluvionale una quota
positiva di particelle nel 32,4% delle celle (massima nel torinese), in perimetro frana
nel 12,0% (nulla a Milano — l'effetto frana non è quindi identificabile nel milanese),
in zona costiera monitorata il 4,3% (solo Napoli e Roma). PGA e r99pday variano
soprattutto <em>tra</em> comuni (v. §8.3). Le correlazioni tra rischi sono moderate
(|ρ| max 0,57 tra PGA e rr20mm): nessuna collinearità critica (VIF in §8.4).</p>
<div class="tabwrap">{tbl(t7f, nd=3, small=True)}</div>
{f2}
{f4}
<h3>Precipitazioni estreme: natura dell'indicatore headline</h3>
<div class="warn"><p><code>r99pday</code> (C3S/E-OBS) misura i giorni/anno oltre il 99°
percentile <em>locale</em> della distribuzione dei giorni piovosi della cella di griglia:
è un indicatore <em>relativo</em> (~1 g/anno per costruzione; range Italia 0,40–2,32; CV
0,27), la cui variazione spaziale riflette numerosità dei giorni piovosi e scostamenti
dal periodo di riferimento, non l'intensità assoluta. La variante <code>rr20mm</code>
(giorni ≥20 mm/anno) è a soglia assoluta (CV 0,64); le due correlano solo 0,37 sulla
griglia italiana. Per questo la lettura congiunta headline/robustezza (§9) è sostantiva
e le stime su r99pday vanno interpretate come "anomalia pluviometrica relativa".</p></div>
<h3>Dati mancanti</h3>
<p>Meccanismi e trattamento (§6 del protocollo): (i) variabili costiere mancanti per il
95,7% delle celle = missing <em>strutturale</em> (celle non costiere), gestito con la
quota <code>is_coastal_zone</code> e non imputato; (ii) subsidenza (1,9%) e incendio
(0,5%, celle EFFIS non bruciabili senza cella valida vicina) = missing tecnico raro,
plausibilmente MCAR rispetto al prezzo → listwise deletion (campione 2.728 su 2.793,
97,7%); data l'entità (&lt;5%) non è richiesta un'analisi di sensibilità all'imputazione.
Il rischio incendio è attribuito dalla cella EFFIS bruciabile più vicina per il 5,5%
delle particelle (dist. media 13,3 km): errore di misura discusso in §13.</p>
"""

# ====================================================================
# 7. METODOLOGIA
# ====================================================================
S["s7"] = """
<h2 id="s7">7. Metodologia econometrica e specificazione completa</h2>
<h3>7.1 Gerarchia a livelli e parsimonia (motivazione)</h3>
<p>Il protocollo adotta tre livelli: il Livello 1 (OLS log-lineare con FE geografici
crescenti, diagnostica completa, robustezza di base) è sempre eseguito; le tecniche di
Livello 2 (modelli spaziali, forme non lineari, trattamento della multicollinearità,
multilevel, ML) si attivano solo su indicazione della diagnostica; il Livello 3 (panel,
DID, IV, RDD) solo se la struttura dei dati lo consente. Le decisioni effettive, con la
diagnostica che le motiva, sono riepilogate in §9.6. La significatività statistica non è
mai stata usata come criterio di selezione del modello.</p>
<h3>7.2 Specifiche headline (equazioni complete)</h3>
<p>Variabile dipendente: log del prezzo OMI di cella (punto medio, €/m²). Vettore di
rischio RISK<sub>c</sub> = (flood_ord, landslide_ord, pga, sub_sink, fire_score, r99pday,
is_coastal_zone), dove flood_ord e landslide_ord sono le medie di cella delle classi
ordinali PGRA (0–3) e PAI (0–4), sub_sink = −velocità di subsidenza (mm/anno, positivo =
sprofondamento), is_coastal_zone = quota di particelle in zona costiera monitorata.</p>
<p class="eq"><strong>A0 (naive)</strong>:&nbsp;
log P<sub>c</sub> = α + Σ<sub>k</sub> β<sub>k</sub>·RISK<sub>k,c</sub>
+ δ<sub>tip(c)</sub> + δ<sub>cond(c)</sub> + ε<sub>c</sub></p>
<p class="eq"><strong>A1 (benchmark provincia + amenità)</strong>:&nbsp;
log P<sub>c</sub> = α + Σ<sub>k</sub> β<sub>k</sub>·RISK<sub>k,c</sub>
+ δ<sub>tip(c)</sub> + δ<sub>cond(c)</sub> + δ<sub>fascia(c)</sub>
+ γ<sub>1</sub>·dist_cbd<sub>c</sub> + γ<sub>2</sub>·is_urban<sub>c</sub>
+ α<sub>prov(c)</sub> + ε<sub>c</sub></p>
<p class="eq"><strong>A2 (within-comune, modello principale)</strong>:&nbsp;
log P<sub>c</sub> = α + Σ<sub>k</sub> β<sub>k</sub>·RISK<sub>k,c</sub>
+ δ<sub>tip(c)</sub> + δ<sub>cond(c)</sub> + δ<sub>fascia(c)</sub>
+ γ<sub>1</sub>·dist_cbd<sub>c</sub> + γ<sub>2</sub>·is_urban<sub>c</sub>
+ α<sub>com(c)</sub> + ε<sub>c</sub></p>
<p>dove δ<sub>tip</sub> (12 tipologie), δ<sub>cond</sub> (3 stati), δ<sub>fascia</sub>
(5 fasce OMI: B/C/D/E/R), α<sub>prov</sub> (4), α<sub>com</sub> (345) sono effetti fissi;
dist_cbd è la distanza haversine dal capoluogo (km) e is_urban la quota di particelle in
land cover artificiale. Termine di errore ε<sub>c</sub>: eteroschedastico e correlato
entro comune; inferenza con SE cluster-robust per comune in tutte le specifiche
(345 cluster ≫ 30: asintotica affidabile, §9 del protocollo; la sensibilità al clustering
per provincia, 4 cluster, è trattata con wild cluster bootstrap in §9.3).
<strong>A2w</strong>: come A2, WLS con pesi = n. particelle della cella (§7.2 del
protocollo).</p>
<p>Ulteriori specifiche headline (forma funzionale diversa, §7.1 del protocollo),
stimate nell'ambito dell'attivazione delle forme non lineari (§9.6):
<em>A2-spline</em> = A2 con dist_cbd in spline B cubica (df=4); <em>A2-classi</em> = A2
con flood_ord sostituito dalle quote di cella nelle classi PGRA (sh_LPH, sh_MPH, sh_HPH)
e landslide_ord dalle quote PAI (sh_P12, sh_P34); <em>A1-quartili</em> = A1 con r99pday
in dummy di quartile. Tutte le altre stime riportate sono varianti richiamate per delta
rispetto a una headline (§9).</p>
<h3>7.3 Identificazione per saturazione</h3>
<p>Il confronto A0 → A1 → A2 separa: associazioni spurie da macro-localizzazione
(assorbite da A1), confondimento da amenità locali (fascia, centralità, urbanizzazione),
e associazioni che sopravvivono a parità di comune (A2). La traiettoria completa dei
coefficienti è riportata in §8.2; la quota di varianza di ciascun rischio che sopravvive
within-comune — cruciale per giudicare quali coefficienti A2 siano identificati — in §8.3.</p>
"""

# ====================================================================
# 8. RISULTATI
# ====================================================================
mm = m1.copy()
mm["rischio"] = mm["rischio"].map(IT)
mm["IC95 (pct/1sd)"] = mm.apply(
    lambda r: f"[{100*(np.exp(r['ci95_low']*r['sd_var'])-1):+.2f}; "
              f"{100*(np.exp(r['ci95_high']*r['sd_var'])-1):+.2f}]", axis=1)
cols_show = ["rischio", "beta", "ci95_low", "ci95_high", "p_grezzo", "p_holm",
             "pct_per_1sd", "IC95 (pct/1sd)"]


def spec_table(name):
    d = mm[mm["spec"] == name][cols_show].rename(columns={
        "beta": "β", "ci95_low": "IC95 β low", "ci95_high": "IC95 β high",
        "p_grezzo": "p grezzo", "p_holm": "p Holm", "pct_per_1sd": "% per +1sd"})
    return tbl(d, nd=4, small=True)


fit_show = m2.rename(columns={m2.columns[0]: "spec"})
f5 = b64img(os.path.join(MOD, "f5_traiettoria_coefficienti.png"), "traiettoria",
            "Fig. 4 — Traiettoria dei coefficienti di rischio A0 → A1 → A2 (IC95, SE cluster comune).")
m3f = m3.copy()
m3f["quota_within_%"] = (100 * m3f["quota_within"]).round(1)
S["s8"] = f"""
<h2 id="s8">8. Risultati principali</h2>
<h3>8.1 Standard di reporting</h3>
<p>Per ogni coefficiente di rischio: stima puntuale β, intervallo di confidenza al 95%,
p-value grezzo e corretto per test multipli (Holm entro la famiglia dei 7 rischi della
specifica; la correzione <em>globale</em> sull'intera famiglia primaria 7 rischi × 2
scale è in §9.2), magnitudine economica come variazione percentuale del prezzo per +1
deviazione standard del rischio, 100·(e<sup>β·sd</sup>−1), con IC95 nella stessa metrica.
N = 2.728, cluster (comuni) = 345 in tutte le specifiche headline.</p>
<h3>8.2 Traiettoria A0 → A1 → A2</h3>
<h4>A0 — naive (solo controlli edilizi)</h4>
<div class="tabwrap">{spec_table("A0")}</div>
<h4>A1 — provincia + amenità</h4>
<div class="tabwrap">{spec_table("A1")}</div>
<h4>A2 — within-comune (modello principale)</h4>
<div class="tabwrap">{spec_table("A2")}</div>
<h4>A2w — within-comune, pesata (n. particelle)</h4>
<div class="tabwrap">{spec_table("A2w")}</div>
{f5}
<div class="tabwrap">{tbl(fit_show, nd=3, small=True)}</div>
<p><strong>Lettura.</strong> (i) L'associazione negativa dell'alluvione in A0 (−5,1% per
+1 sd, p=0,011) si riduce a −2,5% (p=0,059) in A1 e a −0,9% (p=0,50) in A2: il pattern è
quello di un confondimento di macro-localizzazione progressivamente assorbito, non di uno
sconto di prezzo. (ii) Le uniche associazioni forti sono <em>positive</em>: la prossimità
costiera (+14,4% per sd in A1) — un premio di amenità che domina qualunque sconto per
l'erosione — e r99pday (+6,0% per sd in A1). (iii) Nessun rischio mostra sconto
significativo within-comune. (iv) La stima pesata A2w è quasi identica ad A2 (le
conclusioni non dipendono dai pesi).</p>
<h3>8.3 Gradi di libertà effettivi e scala di identificazione (§16.1)</h3>
<p>Il salto di R² tra A1 (0,861) e A2 (0,958) va contestualizzato: A2 stima 371 parametri
(di cui 344 dummy comunali) su 2.728 osservazioni (df residui 2.357) e 280 comuni su 345
hanno una sola zona OMI quotata. La quota di varianza di ciascun rischio che sopravvive
al netto degli FE comunali è:</p>
{tbl(m3f[["variabile", "quota_within_%"]], nd=1)}
<p><strong>Questo è un risultato sostantivo dello studio</strong>: PGA (0,0%), incendio
(0,6%), costa (1,4%) e r99pday (1,6%) variano quasi solo <em>tra</em> comuni — il loro
coefficiente A2 è identificato su una frazione trascurabile di variazione (emblematico
l'IC95 di ±20 punti percentuali sulla PGA in A2) e non va interpretato. Per questi rischi
la scala informativa è A1 (tra comuni, entro provincia); A2 è la specifica di riferimento
per alluvione (10,4% di varianza within) e subsidenza (7,7%). Le conclusioni (§15)
distinguono sistematicamente le due scale.</p>
<h3>8.4 Diagnostica obbligatoria (§16)</h3>
<ul>
<li><strong>Multicollinearità</strong> — VIF (soglie 5/10), sia sui livelli sia within-comune:
massimo 1,53. Nessun trattamento di Livello 2 necessario.</li>
<li><strong>Eteroschedasticità</strong> — Breusch-Pagan su A2: LM=738,5, p≈9·10⁻²⁷ →
eteroschedasticità presente, già gestita dai SE cluster-robust.</li>
<li><strong>Normalità dei residui</strong> — Jarque-Bera su A2: JB=144,6, p≈4·10⁻³²
(skew −0,15, curtosi ecc. 1,09): rigetto formale, non critico per l'inferenza con
N=2.728 e SE robusti; rilevante solo per i sottoinsiemi piccoli (§11).</li>
<li><strong>Forma funzionale</strong> — RESET di Ramsey (potenze 2ª e 3ª del fitted,
Wald con SE cluster): stat=137,4, p&lt;10⁻²⁹ → rigetto della forma log-lineare →
<em>attivazione</em> delle forme flessibili di Livello 2 (esito in §9.5: il misfit
riguarda la struttura edonica dei controlli, i coefficienti di rischio sono invarianti).</li>
<li><strong>Osservazioni influenti</strong> — Cook's D con soglia 4/n=0,0015: 11 celle con
leva h&gt;0,99 (assorbite da FE quasi-singleton, Cook indefinito — segnalate, non
"influenti"); sulle restanti, 193 celle (7,1%) sopra soglia con massimo 0,015: nessuna
osservazione dominante; il trim dei prezzi estremi (§9.1) non altera i risultati.</li>
<li><strong>Autocorrelazione spaziale</strong> — Moran's I e test LM: §10.</li>
<li><strong>Bontà di adattamento</strong> — tabella in §8.2 (R², adj-R², AIC, BIC, RMSE,
MAE); capacità predittiva out-of-comune in §12.2 (ML e confronto).</li>
</ul>
<div class="tabwrap">{tbl(m4.reset_index().rename(columns={"index": "variabile"}), nd=2, small=True)}</div>
"""

# ====================================================================
# 9. ROBUSTEZZA
# ====================================================================
r4f = r4.copy()
r4f["variabile"] = r4f["variabile"].map(IT)
r5f = r5.copy()
r5f["variabile"] = r5f["variabile"].map(IT)
r5f = r5f[["spec", "variabile", "MDE_pct_per_1sd", "p_grezzo", "verdetto"]]
r3f = r3.copy()
r3f["variabile"] = r3f["variabile"].map(IT)
f6 = b64img(os.path.join(ROB, "f6_curva_specificazione.png"), "curva specificazione",
            "Fig. 5 — Curva di specificazione per alluvione e precipitazioni estreme "
            "(tutte le specifiche effettivamente stimate; headline in rosso).")
lopo1 = r2a.round(2)
lopo2 = r2b.round(2)
S["s9"] = f"""
<h2 id="s9">9. Analisi di robustezza</h2>
<h3>9.1 Varianti di specifica (delta rispetto alle headline)</h3>
<p>Tutte le seguenti varianti replicano A1/A2 con la sola modifica indicata (§7.1):
precipitazioni come rr20mm (soglia assoluta 20 mm, §4.2 del protocollo) anziché r99pday;
alluvione/frana come quote di particelle esposte (any) anziché ordinali; sisma come zona
sismica anziché PGA; subsidenza come classe di rischio anziché velocità; winsorizzazione
del log-prezzo a p1/p99; esclusione degli outlier di prezzo (trim p1/p99); stima pesata
(A1w, A2w); aggregazione territoriale alternativa comune × tipologia. Nessuna variante
modifica il quadro qualitativo: l'alluvione resta tra −5,2% e +0,5% (mediana −1,4%) e mai
significativamente negativa nelle specifiche sature; le precipitazioni estreme restano
positive nella quasi totalità delle specifiche (mediana +5,3%); con rr20mm al posto di
r99pday il coefficiente A1 resta positivo, coerente con un gradiente pluviometrico
correlato ad altre caratteristiche territoriali più che con uno sconto di rischio.
Dettaglio completo in <code>output/robustezza/r7_curva_specificazione.csv</code>.</p>
<h3>9.2 Correzione per test multipli globale (§17.1)</h3>
<p>Holm sull'intera famiglia primaria (7 rischi × {{A1, A2}} = 14 test):</p>
<div class="tabwrap">{tbl(r4f, nd=4, small=True)}</div>
<p>Sopravvivono al 5% soltanto costa A1 e r99pday A1, entrambe positive. Nessuna
associazione negativa sopravvive (né esisteva al livello grezzo nelle specifiche sature).</p>
<h3>9.3 Inferenza con pochi cluster (§9) e stabilità del campione (§10)</h3>
<p>Il clustering primario è al comune (345 cluster: asintotica affidabile, wild bootstrap
non necessario a quel livello). Al livello provincia i cluster sono 4: il wild cluster
bootstrap (null imposto, pesi di Webb, B=4.999) su A1 dà p: frana 0,050, costa 0,059,
r99pday 0,061, subsidenza 0,214, incendio 0,287, alluvione 0,438, PGA 0,725 — al livello
di aggregazione più alto tutte le associazioni diventano borderline: le stime non vanno
lette come robuste a shock comuni di provincia.</p>
<div class="tabwrap">{tbl(r3f, nd=4, small=True)}</div>
<p>Leave-one-provincia-out (effetto % per +1 sd):</p>
<div class="tabwrap">{tbl(lopo1, nd=2, small=True)}</div>
<div class="tabwrap">{tbl(lopo2, nd=2, small=True)}</div>
<p><strong>Lettura (criterio §9-§10)</strong>: r99pday in A1 è qualitativamente stabile
(sempre positivo, +3,2/+8,5%). Il premio costiero è <em>guidato da Napoli</em>: senza
Napoli scende a +3,0% in A1 e cambia segno in A2 (−12,0%) → viene derubricato a evidenza
trainata da un cluster (isole e costiere campane), non a stima robusta a sé stante.
PGA e incendio oscillano attorno allo zero (coerente con la debole identificazione);
l'alluvione resta non positiva in ogni sottocampione.</p>
<h3>9.4 Potenza statistica e MDE (§17.2)</h3>
<div class="tabwrap">{tbl(r5f, nd=2, small=True)}</div>
<p><strong>Distinzione centrale</strong>: in A1 i nulli di alluvione (MDE 3,8%), subsidenza
(4,0%) e incendio (3,2%) sono <em>nulli informativi</em> — la potenza era sufficiente a
rilevare effetti della magnitudine tipicamente documentata in letteratura (5–10%);
il nullo della PGA è invece a bassa potenza (MDE 6,9%). In A2 solo l'alluvione ha un
nullo informativo (MDE 3,7%); per gli altri rischi la scala within-comune è poco
informativa (MDE 6,8–169%), coerentemente con §8.3.</p>
<h3>9.5 Test di falsificazione (§17.3) e forme non lineari attivate</h3>
<p><strong>Placebo</strong> (999 permutazioni, stima esatta via Frisch-Waugh-Lovell):
(a) r99pday permutato tra comuni within-provincia in A1: tasso di rigetto placebo 7,0%
(≈ nominale) e p empirico del coefficiente osservato 0,013 → il segnale positivo non è un
artefatto della pipeline; (b) flood_ord permutato tra zone within-comune in A2: tasso di
rigetto 17,7% → l'inferenza analitica within-comune su flood è <em>anticonservativa</em>;
il p-value di permutazione del coefficiente osservato è 0,60, che conferma il nullo con
un metodo esente dal problema.</p>
<p><strong>Forme non lineari</strong> (Livello 2, attivate dal RESET): con spline su
dist_cbd, log-distanza o interazioni tipologia × fascia il RESET resta rigettato
(p≈10⁻²⁷–10⁻³⁰) ma i coefficienti di rischio sono invarianti (es. flood −0,9% → −0,9%;
r99pday +5,4% → +4,2/+5,2%); le quote per classe di pericolosità PGRA non mostrano alcun
gradiente coerente con uno sconto (LPH +6,5%, MPH −22,4% p=0,08, HPH −2,9% p=0,58; test
congiunto p=0,22) e i quartili di r99pday sono debolmente monotoni ma non significativi.
Conclusione: il misfit di forma riguarda la struttura edonica dei controlli, non la
relazione rischio-prezzo; la log-lineare resta la specifica primaria.</p>
<div class="tabwrap">{tbl(e2.round(4), nd=4, small=True)}</div>
<h3>9.6 Curva di specificazione (§17.4) e riepilogo attivazioni Livello 2/3</h3>
{f6}
<p>La curva raccoglie le sole specifiche effettivamente stimate e giustificate dal
protocollo (headline, varianti §17, leave-one-out, estensioni attivate) — non una griglia
combinatoria a posteriori. Le stime headline sono rappresentative della distribuzione.</p>
<div class="box"><p><strong>Attivazioni Livello 2</strong> — <em>Attivate</em>: forme non
lineari (RESET p&lt;10⁻²⁹; esito: invarianza dei coefficienti di rischio); machine
learning sintetico (sempre previsto, §15 del protocollo). <em>Non attivate</em>: modelli
spaziali SAR/SEM/SDM (Moran cross-comune sui residui A2 I=−0,008, p=0,42: dipendenza già
assorbita dagli FE, §10); trattamento della multicollinearità (VIF max 1,53 &lt; 5);
multilevel (ICC comune 0,36 già assorbita dagli FE di comune).<br>
<strong>Livello 3</strong> — tutte escluse: panel (soli 2 semestri, quotazioni liscie:
usato solo come indizio di repricing, §12.3), DID (nessuno shock con data e gruppo di
controllo credibile nel periodo), IV (nessuno strumento con esogeneità difendibile:
non forzato, §0.1), RDD (le classi PGRA/PAI non presentano soglie continue non
manipolabili osservabili nel dato di cella).</p></div>
"""

# ====================================================================
# 10. SPAZIALE
# ====================================================================
S["s10"] = """
<h2 id="s10">10. Analisi spaziale</h2>
<p>Diagnostica obbligatoria (§12 del protocollo), sempre riportata anche se i modelli
spaziali non risultano attivati:</p>
<ul>
<li><strong>Moran's I sui residui medi di zona OMI</strong> (n=421, W = KNN k=8,
9.999 permutazioni): residui A1 (FE provincia): I=+0,302, p=0,0001 — forte dipendenza
spaziale positiva quando la localizzazione è controllata solo a livello di provincia;
residui A2 (FE comune): I=−0,102, p=0,0001 <em>sulla W piena</em>.</li>
<li><strong>Decomposizione dell'artefatto</strong>: gli FE comunali impongono residui a
somma ≈0 entro comune, inducendo meccanicamente correlazione negativa tra vicini dello
stesso comune. Con una W ristretta ai vicini di <em>altri</em> comuni: A1 I=+0,278
(p=0,0001); A2 I=−0,008 (p=0,42). La dipendenza spaziale genuina è interamente assorbita
dagli effetti fissi comunali.</li>
<li><strong>Test LM su A2</strong> (W KNN k=8 di cella): LM-lag 53,2 (p&lt;10⁻⁴),
LM-error 77,5 (p&lt;10⁻⁴), robust LM-lag 0,79 (p=0,37), robust LM-error 25,1
(p&lt;10⁻⁴) — segnale residuo di tipo error within-comune, riconducibile all'artefatto
di demeaning e alla correlazione intra-comune già gestita dai SE clusterizzati.</li>
</ul>
<p><strong>Decisione (criterio §12)</strong>: la condizione di attivazione — dipendenza
non assorbita dagli FE geografici — non è soddisfatta (Moran cross-comune p=0,42):
SAR/SEM/SDM non stimati; il modello con FE comunali resta la specifica primaria. La
sensibilità alla definizione di W è stata comunque esercitata nella forma della doppia
matrice (KNN piena vs cross-comune) e del doppio livello (cella vs zona), con esito
coerente.</p>
"""

# ====================================================================
# 11. ETEROGENEITA'
# ====================================================================
e3f = e3.copy()
e3f["rischio"] = e3f["rischio"].map(IT)
e3p = e3f.pivot_table(index=["spec", "rischio"], columns="gruppo",
                      values="pct_per_1sd").round(2).reset_index()
S["s11"] = f"""
<h2 id="s11">11. Interazioni ed effetti eterogenei per tipologia edilizia</h2>
<p>Le 12 tipologie sono raggruppate in Residenziale (1.288 celle), Commerciale (465),
Produttivo (662), Box (313); le interazioni rischio × gruppo sono stimate nella scala
informativa di ciascun rischio (§8.3): A2 per alluvione e subsidenza, A1 per PGA,
incendio, precipitazioni e costa. Effetti marginali per gruppo (% per +1 sd), con Holm
entro famiglia:</p>
<div class="tabwrap">{tbl(e3p, nd=2, small=True)}</div>
<p><strong>Risultati</strong>: (i) l'alluvione non mostra sconto significativo in
<em>nessun</em> gruppo (test congiunto di eterogeneità p=0,067; tutti i p di gruppo
&gt;0,21) — nessuna tipologia risulta vulnerabile nel dato OMI; (ii) r99pday è positivo e
uniforme (+5,4/+8,4%, tutti Holm&lt;0,05; uguaglianza tra gruppi non rigettata, p=0,12);
(iii) la PGA mostra l'unica eterogeneità forte (congiunto p&lt;0,0001): positiva per
Commerciale (+9,7%, Holm 0,0007) e Produttivo (+6,7%) — un'associazione di segno
opposto allo sconto, interpretabile come correlazione tra pericolosità sismica dei
territori campani/laziali e struttura dei mercati non residenziali, non come premio al
rischio; (iv) la costa è positiva per tutti i gruppi (+11,4/+16,9%). Le voci per gruppo
non alterano quindi il quadro: nessuna evidenza di sconto differenziale per tipologia.</p>
"""

# ====================================================================
# 12. INTERPRETAZIONE + ML + repricing + canoni
# ====================================================================
f8 = b64img(os.path.join(EST, "f8_ml_importance_rischi.png"), "importance",
            "Fig. 6 — Permutation importance out-of-comune (HGB): variabili di rischio.")
f9 = b64img(os.path.join(EST, "f9_ml_pdp.png"), "pdp",
            "Fig. 7 — Partial dependence (HGB, full sample, descrittiva).")
e6show = e6[e6["dip"] == "log(canone/prezzo)"][
    ["spec", "variabile", "pct_per_1sd", "p", "p_holm"]].copy()
e6show["variabile"] = e6show["variabile"].map(IT)
S["s12"] = f"""
<h2 id="s12">12. Interpretazione economica dei risultati</h2>
<p>Tutte le formulazioni che seguono sono deliberatamente associative (§1 del
protocollo): i coefficienti misurano associazioni condizionate nei livelli delle
quotazioni amministrative, non effetti causali del rischio sui prezzi.</p>
<h3>12.1 Il quadro d'insieme</h3>
<p>Nei valori OMI delle quattro province il rischio climatico-ambientale non risulta
scontato: nessuna famiglia di rischio è associata negativamente alle quotazioni in modo
robusto, a nessuna delle due scale (tra comuni; within-comune). Le due associazioni che
sopravvivono all'intera batteria di robustezza sono positive e coerenti con canali di
amenità o con gradienti territoriali: la prossimità costiera (+14,4% per sd, trainata
dalla provincia di Napoli) e l'anomalia pluviometrica relativa r99pday (+6,0% per sd,
stabile al leave-one-out ma borderline al bootstrap di provincia). Per r99pday la lettura
amenitiva è la più plausibile: l'indicatore, relativo per costruzione (§6), correla con
zone collinari/pedemontane e assi territoriali di pregio entro provincia; la variante
assoluta rr20mm conferma il segno positivo, escludendo che il risultato dipenda dalla
normalizzazione locale dell'indicatore.</p>
<h3>12.2 Conferma indipendente via machine learning (§15 del protocollo)</h3>
<p>Gradient boosting e random forest con validazione GroupKFold per comune (nessun
leakage spaziale): R² out-of-comune 0,89 (HGB) contro 0,81 senza le 7 variabili di
rischio (ΔR² +0,086; RF: +0,047). Il contributo predittivo dei rischi esiste ma è
concentrato — per importance e mean|SHAP| — su costa, PGA e r99pday, cioè sugli stessi
canali "positivi" del modello parametrico; alluvione, frana e incendio hanno contributo
≈0 anche in un modello non parametrico libero di catturare non linearità e interazioni.
Le partial dependence confermano l'assenza di gradiente negativo per flood_ord.</p>
{f8}
{f9}
<h3>12.3 Repricing dinamico (indizio, non test)</h3>
<p>Sui due soli semestri disponibili (26% di celle con prezzo variato), la variazione
Δlog(P) non è associata ad alcun rischio (tutti i p&gt;0,18 in A1; l'unico p=0,0499
isolato su PGA in A2, positivo, è compatibile con il caso su 14 test): nessun segnale di
repricing in corso legato al rischio. Indizio preliminare, esplicitamente non un panel
(Livello 3 non attivato).</p>
<h3>12.4 Canoni e rapporto canone/prezzo (§2): discriminare le ipotesi</h3>
<p>Se il mercato prezzasse il rischio ma l'OMI-prezzi non lo riflettesse, il rischio
dovrebbe comparire nel rapporto canone/prezzo (rendimento lordo richiesto più alto nelle
zone rischiose). I canoni OMI mostrano gli stessi pattern dei prezzi (costa +13,0%
p&lt;0,001; r99pday +5,0% p=0,006 in A1). Nel rapporto canone/prezzo:</p>
<div class="tabwrap">{tbl(e6show, nd=3, small=True)}</div>
<p>Solo la subsidenza sopravvive alla correzione di Holm (−0,9%, segno opposto a quello
atteso da un premio al rischio: plausibile rumore); l'alluvione mostra un +0,9% (p=0,009)
che non sopravvive (Holm 0,054) — al più un indizio debole di rendimento richiesto
lievemente più alto in zona alluvionale. Nel complesso, lo strumento OMI appare
"lisciare" simmetricamente prezzi e canoni: la discriminazione piena tra "mercato non
prezza" e "strumento non riflette" non è possibile con soli dati OMI (v. §13 e
conclusione 7).</p>
"""

# ====================================================================
# 13. LIMITI
# ====================================================================
S["s13"] = """
<h2 id="s13">13. Limiti dello studio</h2>
<ol>
<li><strong>Natura amministrativa del dato di prezzo</strong> (§4.2): quotazioni di zona,
lisciate e prudenziali; qualunque capitalizzazione intra-zona o più rapida del ciclo di
aggiornamento OMI è invisibile per costruzione. È il limite principale e definisce
l'oggetto dello studio (lo strumento, non il mercato).</li>
<li><strong>Selezione campionaria</strong>: copre il perimetro urbano quotato (38,8% delle
particelle; 18–87% a seconda della provincia); le aree non quotate sono più extraurbane e
lievemente più esposte a precipitazioni estreme: i risultati non vi si estendono.</li>
<li><strong>Errore di misura nei rischi interpolati</strong> (§11 del protocollo):
precipitazioni attribuite dal punto griglia più vicino (3,8 km in media, p95 6,0 km);
incendio da cella ~12 km, e dalla cella bruciabile più vicina (13,3 km in media) per il
5,5% delle particelle urbane dense. L'errore di misura spinge le stime verso zero
(attenuation): i nulli "informativi" su incendio vanno quindi qualificati — il MDE
effettivo sul rischio vero è più alto di quello nominale. Per i perimetri PGRA/PAI e la
zona sismica (attribuzione puntuale/comunale) il problema è minore.</li>
<li><strong>Gradi di libertà effettivi in A2</strong> (§8.3): per PGA, incendio, costa e
r99pday la scala within-comune è quasi degenere; le conclusioni su questi rischi valgono
alla scala tra-comuni (A1) e con la cautela del bootstrap di provincia (§9.3).</li>
<li><strong>Quattro province</strong>: al livello di shock comuni di provincia l'inferenza
è debole per costruzione (4 cluster); il WCB rende tutte le associazioni borderline.</li>
<li><strong>Dimensione temporale minima</strong>: due semestri di quotazioni liscie; il
controllo di repricing è solo un indizio.</li>
<li><strong>r99pday come indicatore relativo</strong>: misura anomalia pluviometrica
locale, non intensità assoluta; la triangolazione con rr20mm mitiga ma non elimina il
limite.</li>
<li><strong>Assenza di variabili socioeconomiche fini</strong> (reddito di zona,
qualità dei servizi): i confronti within-comune con fascia e centralità li proxano solo
in parte; le associazioni positive residue possono riflettere amenità non osservate.</li>
</ol>
"""

# ====================================================================
# 14. INDICE
# ====================================================================
S["s14"] = """
<h2 id="s14">14. Implicazioni per la costruzione futura di un indice sintetico</h2>
<p>La condizione posta dal protocollo (§0) — procedere all'indice solo in presenza di una
relazione empiricamente solida e a una scala di aggregazione adeguata — <strong>non è
soddisfatta per un indice di capitalizzazione</strong>: non esiste un vettore di pesi
"rivelato dai prezzi OMI" da cui derivare un indice del rischio scontato, perché i prezzi
ufficiali non scontano i rischi fisici misurati. Costruirlo forzando i coefficienti
positivi (costa, r99pday) significherebbe sintetizzare amenità, non rischio.</p>
<p>Restano percorribili, con basi diverse:</p>
<ul>
<li>un <strong>indice di esposizione fisica</strong> multi-hazard a scala di zona
OMI/comune (le fonti integrate — PGRA, PAI, PGA, EFFIS, E-OBS, subsidenza, costa — hanno
risoluzione e copertura adeguate a questa scala; la correlazione tra rischi è modesta,
VIF max 1,5, quindi l'aggregazione richiede pesi normativi espliciti, non statistici);</li>
<li>un <strong>indice di divario di capitalizzazione</strong> (quanto il valore ufficiale
<em>non</em> riflette il rischio), utile in ottica di collateral/risk management: la sua
costruzione richiede però prezzi di transazione (es. quotazioni notarili georeferenziate)
o canoni di mercato per stimare il benchmark "prezzato".</li>
</ul>
<p>La scala geografica valida, in entrambi i casi, è quella tra-comuni/di zona (A1):
within-comune la variazione dei rischi non perimetrali è quasi nulla (§8.3).</p>
"""

# ====================================================================
# 15. CONCLUSIONI
# ====================================================================
S["s15"] = """
<h2 id="s15">15. Conclusioni</h2>
<p>Risposte alle domande obbligatorie del protocollo (§22), con livello di evidenza
(<span class="ev-forte">forte</span> / <span class="ev-moderata">moderata</span> /
<span class="ev-debole">debole</span> / <span class="ev-inconclusiva">inconclusiva</span>).</p>
<ol class="concl">
<li><strong>Esiste una relazione significativa tra rischio e quotazioni?</strong>
Nel senso rilevante per la capitalizzazione — uno sconto — <em>no</em>: nessuna
associazione negativa robusta a nessuna scala (evidenza
<span class="ev-forte">forte</span> per l'alluvione, nullo informativo a entrambe le
scale con MDE ≤3,8%; <span class="ev-moderata">moderata</span> per subsidenza e incendio,
nulli informativi in A1 ma con attenuation bias plausibile per l'incendio;
<span class="ev-debole">debole</span> per PGA e frana, bassa potenza). Esistono invece
due associazioni positive robuste a Holm a scala tra-comuni: costa +14,4% e r99pday
+6,0% per sd (evidenza <span class="ev-moderata">moderata</span>: sopravvivono a Holm e
ai placebo, ma la prima è trainata da Napoli e entrambe sono borderline al wild cluster
bootstrap di provincia).</li>
<li><strong>Direzione dell'effetto.</strong> Dove significativa, positiva — opposta allo
sconto atteso da capitalizzazione del rischio; coerente con premi di amenità e gradienti
territoriali correlati ai rischi (evidenza <span class="ev-moderata">moderata</span>).</li>
<li><strong>Dimensione economica.</strong> Alluvione within-comune: −0,9% per +1 sd
(IC95 [−3,4%; +1,7%]) — economicamente trascurabile; costa +14,4% (IC95 [+10,5%; +18,4%])
e r99pday +6,0% (IC95 [+1,9%; +10,3%]) per +1 sd in A1. In prospettiva: gli sconti
documentati in letteratura per il rischio alluvionale (tipicamente −2/−8%) sarebbero
stati rilevabili in A1 (MDE 3,8%).</li>
<li><strong>Rischi più rilevanti / più chiaramente irrilevanti.</strong> Chiaramente non
riflessi nelle quotazioni, con potenza adeguata: alluvione (entrambe le scale),
subsidenza e incendio (A1; per l'incendio con la qualifica dell'errore di misura).
Non riflessi ma con potenza insufficiente: PGA, frana (within), erosione come rischio
(indistinguibile dall'amenità costiera). Associazioni positive dominanti: costa,
r99pday (evidenza <span class="ev-forte">forte</span> sulla classificazione dei nulli,
dalla tabella MDE).</li>
<li><strong>Tipologie più vulnerabili.</strong> Nessuna: l'alluvione è nulla in tutti i
gruppi (Residenziale/Commerciale/Produttivo/Box); l'unica eterogeneità forte è
l'associazione positiva PGA × Commerciale (+9,7%), non interpretabile come vulnerabilità
(evidenza <span class="ev-moderata">moderata</span>).</li>
<li><strong>Robustezza complessiva.</strong> L'assenza di sconto è robusta a: 16 varianti
di specifica, pesi, winsorizzazione/trim, aggregazioni alternative, leave-one-provincia-out,
correzione di Holm globale, placebo permutazionali, forme non lineari e verifica ML
indipendente; la diagnostica spaziale mostra dipendenza interamente assorbita dagli FE
comunali (evidenza <span class="ev-forte">forte</span>). Le associazioni positive sono
robuste a Holm ma sensibili al livello di clustering provinciale e (costa) alla
composizione del campione (evidenza <span class="ev-moderata">moderata</span>).</li>
<li><strong>Mercato che non prezza o strumento che non riflette?</strong> I dati OMI da
soli non discriminano pienamente. Indizi a favore di un contributo rilevante dello
<em>strumento</em>: granularità di zona e lisciamento anche dei canoni (§12.4), quasi
immobilità semestrale (26% di celle riprezzate), letteratura sul ritardo degli strumenti
valutativi (Clayton et al. 2021). Indizi che anche il <em>mercato</em> possa non prezzare
in questo contesto: assenza di obbligo di disclosure e bassa penetrazione assicurativa
catastrofale in Italia (EIOPA-ECB 2024), meccanismi di salienza (Hino-Burke 2021;
Baldauf et al. 2020). L'unico segnale di mercato indipendente (rapporto canone/prezzo)
dà al più un indizio debole di premio alluvionale (+0,9%, non sopravvive a Holm).
Verdetto: <span class="ev-inconclusiva">inconclusiva</span> sulla discriminazione piena;
<span class="ev-moderata">moderata</span> sul fatto che lo strumento ufficiale, comunque
sia, non riflette il rischio — che è la risposta operativa per usi di collateral e risk
management.</li>
<li><strong>Indice sintetico: procedere?</strong> Non per un indice di capitalizzazione
(nessuna base empirica); sì, con basi normative, per un indice di esposizione a scala
comune/zona OMI, ed eventualmente per un indice di divario valore-rischio subordinato
all'acquisizione di prezzi di transazione (evidenza
<span class="ev-forte">forte</span> sulla prematurità; §14).</li>
<li><strong>Tecniche di Livello 2/3: attivate ed escluse.</strong>
<em>Attivate</em>: forme non lineari (RESET p&lt;10⁻²⁹; esito: invarianza dei rischi);
ML sintetico (previsto dal protocollo; esito: conferma). <em>Escluse</em>: SAR/SEM/SDM
(Moran cross-comune p=0,42: dipendenza assorbita dagli FE); PCA/Ridge/blocchi (VIF max
1,53 &lt; 5); multilevel (ICC 0,36 già assorbita dagli FE comunali); panel (2 soli
semestri amministrativi); DID (nessun evento con data e controllo credibile); IV (nessuno
strumento difendibile — non forzato); RDD (nessuna soglia netta non manipolabile
osservabile). Ciascuna esclusione discende dal criterio di attivazione §0.1.</li>
</ol>
"""

# ====================================================================
# APPENDICE
# ====================================================================
S["app"] = """
<h2 id="app">Appendice — Riproducibilità (§20)</h2>
<p>L'intera analisi è riproducibile dai dati grezzi con la pipeline (Python 3.13;
pandas, statsmodels, libpysal/esda/spreg, scikit-learn, shap, xarray):</p>
<ol>
<li><code>integra_rischio_incendio.py</code>, <code>integra_rischio_precipitazioni.py</code>,
<code>integra_rischio_p99.py</code> — integrazione griglie EFFIS/E-OBS nel DB SQLite
(vintage in tabella fonti, §4).</li>
<li><code>prepara_dataset_analisi.py</code> — pulizia (riattribuzione comune catastale,
esclusione boundary leakage), esplosione voci OMI, aggregazione all'unità statistica;
output <code>output/particelle.csv</code>, <code>output/celle.csv</code>.</li>
<li><code>analisi_esplorativa.py</code> — EDA, selezione campionaria, tabella fonti
(<code>output/eda/</code>).</li>
<li><code>modelli_livello1.py</code> — specifiche headline, diagnostica §16, decisioni di
attivazione (<code>output/modelli/</code>).</li>
<li><code>robustezza.py</code> — varianti, LOPO, WCB, Holm, MDE, placebo, curva di
specificazione (<code>output/robustezza/</code>).</li>
<li><code>estensioni_livello2.py</code> — forme non lineari, eterogeneità, ML, repricing,
canoni (<code>output/estensioni/</code>).</li>
<li><code>report_finale.py</code> — questo documento.</li>
</ol>
<p>Ogni numero del report è tracciabile a un file CSV di output e alla specifica
(headline o delta) che lo ha generato. Semi casuali fissati (bootstrap, permutazioni,
ML). Dati di prezzo: download API 09–11/07/2026.</p>
"""

# ====================================================================
TOC = """
<nav class="toc"><strong>Indice</strong><br>
<a href="#s1">1. Abstract</a><br><a href="#s2">2. Introduzione</a><br>
<a href="#s3">3. Letteratura</a><br><a href="#s4">4. Database e natura del dato</a><br>
<a href="#s5">5. Unità statistica</a><br><a href="#s6">6. Analisi esplorativa</a><br>
<a href="#s7">7. Metodologia</a><br><a href="#s8">8. Risultati principali</a><br>
<a href="#s9">9. Robustezza</a><br><a href="#s10">10. Analisi spaziale</a><br>
<a href="#s11">11. Effetti eterogenei</a><br><a href="#s12">12. Interpretazione</a><br>
<a href="#s13">13. Limiti</a><br><a href="#s14">14. Indice sintetico</a><br>
<a href="#s15">15. Conclusioni</a><br><a href="#app">Appendice: riproducibilità</a>
</nav>
"""

html = f"""<!DOCTYPE html>
<html lang="it"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rischio climatico e quotazioni immobiliari ufficiali — Milano, Napoli, Roma, Torino</title>
<style>{CSS}</style></head><body><main>
<h1>Il rischio climatico-ambientale è scontato nelle quotazioni immobiliari
ufficiali?<br>Evidenza dalle province di Milano, Napoli, Roma e Torino</h1>
<p class="meta">Studio quantitativo su quotazioni OMI e rischi fisici
(alluvione, frana, sisma, subsidenza, incendio, precipitazioni estreme, erosione
costiera) — dataset Zornade, {pd.Timestamp.today():%d/%m/%Y}. Protocollo a livelli
guidato dalla diagnostica; linguaggio associativo, non causale.</p>
{TOC}
{S["s1"]}{S["s2"]}{S["s3"]}{S["s4"]}{S["s5"]}{S["s6"]}{S["s7"]}{S["s8"]}
{S["s9"]}{S["s10"]}{S["s11"]}{S["s12"]}{S["s13"]}{S["s14"]}{S["s15"]}{S["app"]}
</main></body></html>
"""

with open(DEST, "w", encoding="utf-8") as f:
    f.write(html)
print(f"[OK] Report scritto: {DEST}  ({os.path.getsize(DEST)/1e6:.1f} MB)")
