# Rischio climatico-ambientale e prezzi immobiliari — dati OMI e pipeline di integrazione dei rischi

Dati e script usati nell'analisi empirica di una tesi sui metodi di valutazione immobiliare e sulla trasmissione ai prezzi dei rischi climatico-ambientali (rischio alluvionale, sismico, subsidenza, eventi meteo estremi, calore/siccità) in Italia.

## Contenuto

### `data/`

| File | Descrizione | Unità di analisi |
|---|---|---|
| `celle_con_calore.csv` | Dataset finale: quotazioni OMI (prezzo/m², affitto/m²) incrociate con tutti gli indicatori di rischio (sismico, alluvionale, frana, subsidenza, incendio, precipitazioni estreme, calore, siccità, costa) e con l'indice di rischio climatico | zona OMI |
| `particelle.csv` | Stessi indicatori di rischio attribuiti a livello di particella catastale (più granulare della zona OMI) | particella catastale |
| `indice_esposizione_zone.csv` | Indice di esposizione fisica al rischio climatico-ambientale per zona, con relativo valore immobiliare esposto | zona OMI |

Fonti dei dati sottostanti: Osservatorio del Mercato Immobiliare (OMI, Agenzia delle Entrate); mappe di pericolosità sismica, alluvionale, da frana e di subsidenza; EFFIS (European Forest Fire Information System, JRC/Copernicus) per il rischio incendio; E-OBS (ECA&D) per le precipitazioni estreme; indicatori di calore e siccità da rianalisi climatiche; indice di rischio climatico regionale.

### `scripts/`

Pipeline Python che integra le diverse fonti di rischio nel dataset OMI e produce i modelli e le tabelle di analisi (scaricamento dati OMI, integrazione dei singoli rischi, costruzione dell'indice di esposizione, stima dei modelli edonici, test di robustezza).


## Nota

Repository di accompagnamento a un lavoro di tesi in corso. I dati sono forniti così come usati nell'analisi, senza garanzie di completezza o accuratezza per altri usi.
