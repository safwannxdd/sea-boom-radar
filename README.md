# SEA Boom Radar

**Which Silicon Valley trends will become Southeast Asia's next venture booms, and which are just noise?**

The project uses free, public data to build a reproducible machine-learning pipeline. It does two things:

1. It separates **durable SF trends** from **hype** using two independent methods.
2. It **backtests** both methods on real history. Would they have flagged ride-hailing, e-wallets and BNPL before
   they boomed in SEA, and stayed sceptical of daily deals, NFTs and the metaverse?

## Results (Oct 2026 run)

**Pre-registered test (the headline):** at the moment a VC had to decide, neither composite score reliably
separated the 7 SF→SEA booms from 12 hype waves.

| Score | AUC | 95% CI (sector bootstrap) | permutation p |
|---|---|---|---|
| Method A (convergence) | 0.62 | 0.45–0.77 | 0.10 |
| Method B (time-series + ML) | 0.52 | 0.31–0.75 | 0.43 |
| Ensemble | 0.60 | 0.39–0.81 | 0.21 |
| "Follow the buzz" baseline | 0.37 | 0.18–0.58 | — |

**What held up (exploratory, needs out-of-sample confirmation):**
- **Synchronisation is the tell.** Hype moves SF and SEA search interest in the same month (cross-corr 0.73 at lag 0).
  Real booms reach SEA ~10–14 months later. As a single signal: AUC 0.77, permutation p = 0.03 (13 sectors, before
  multiple-testing correction).
- **The SF→SEA link is sector-specific, not just background co-movement.** SF search interest Granger-predicts
  SEA interest in the same sector for 83% of sectors (67% after FDR correction), against 54% for placebo pairs that match
  one sector's SF data with a different sector's SEA data (Fisher exact p = 0.012, odds ratio 4.2). The link is tighter
  for hype (11/11) than for booms (4/7). This shows prediction, not causation, and covers search attention only.
- **Crowd signals are contrarian.** Attention momentum and YC founder momentum are both *higher* for hype than for booms.
- **Shape is decisive in hindsight only.** Two to three years after the peak, booms retain 63% of peak interest and hype retains 18%.

A first run reported Method A AUC = 0.73. That came from an evaluation-window bug and is retracted; see notebook 05.

**Data status:** Google Trends rate-limited this run. 18 of 19 labelled sectors use Google Trends, but most 2026
candidates fall back to Wikipedia pageviews (2015 onward). Re-run after `python -m radar.fetch gtrends` completes.

## Notebooks (run in order)

| # | Notebook | What it does |
|---|---|---|
| 01 | `01_data_collection.ipynb` | Pulls 7 free sources into one tidy sector × time panel. Converts counts to shares. |
| 02 | `02_explore_signals.ipynb` | Hindsight exploration: diffusion lag, synchronised hype, spike shape, signal independence. |
| 03 | `03_method_A_convergence.ipynb` | **Method A:** pre-registered, explainable multi-signal convergence score. |
| 04 | `04_method_B_timeseries_ml.ipynb` | **Method B:** STL, ADF, structural breaks, a gradient-boosted persistence classifier trained on YC history, and Granger SF→SEA. |
| 05 | `05_backtest_and_compare.ipynb` | Point-in-time backtest, sector bootstrap CIs, permutation test, weight robustness, A vs B. |
| 06 | `06_sea_radar_2026.ipynb` | Applies everything to 18 candidate sectors as of Oct 2026. Radar chart and ranked table. |

The notebook sources are readable `.py` files in `notebooks/_src/` (jupytext percent format).
Each `# %%` block is one notebook cell.

## Data sources (all free, no API keys)

| Source | Proxy for | Notes |
|---|---|---|
| Google Trends (SF Bay Area metro + SG, ID, VN, PH, TH, MY) | consumer attention, SF vs SEA | via `pytrends`; heavily rate-limited, so the first run is slow |
| Hacker News (Algolia API) | builder/investor chatter | quarterly, normalised by total HN activity |
| Y Combinator directory (yc-oss mirror) | where founders place bets | share of each year's YC companies |
| GitHub search API | developers building | yearly, normalised |
| OpenAlex | research activity | yearly, normalised |
| Wikipedia pageviews (en + id/vi/th/ms/tl) | public curiosity, Anglosphere vs SEA | from Jul 2015 |
| World Bank | SEA readiness (internet, banking, income, urbanisation) | context layer |

## Run it

```bash
pip install -r requirements.txt
# 1. collect data (cached; Google Trends and GitHub take ~1-2h because of rate limits)
for s in hn gtrends github openalex wiki denominators; do python -m radar.fetch $s & done; wait
# 2. build + execute all notebooks
./build_notebooks.sh
```

## Rigour checklist (what a sceptical investor should ask)

- **No look-ahead:** every score for year *t* uses only data up to *t*. The ML model is retrained each year on
  examples whose outcome was already known.
- **Scale-free features:** Google Trends rescales each series to its all-time max. All features are ratios or growth
  rates, so that rescaling cannot leak future information.
- **Shares, not counts:** HN, GitHub and research volumes grew 10–100×. Raw counts would make everything a "trend".
- **Labels fixed in advance:** see `radar/config.py`. Each label has a one-line rationale and ambiguous cases are excluded.
- **Weights fixed in advance:** Method A's weights are tested against 2,000 random weightings.
- **Small-N honesty:** 19 labelled sectors. Confidence intervals use a sector-level bootstrap, plus a permutation test.

## Limitations

- Labels are judgement calls, not a dataset. Disagree with one? Change it in `config.py` and re-run.
- Google Trends is noisy for low-volume terms and SEA countries. Sparse countries are dropped per sector.
- No paid deal data (Crunchbase/PitchBook). Adding SF and SEA funding rounds would be the biggest upgrade.
- The radar answers *"is this trend real, and has SEA moved yet?"* It does not answer *"will a specific startup win?"*
- Some SEA booms are SEA-native (super-apps, social commerce) and have no SF lead. The radar is blind to those by design.
