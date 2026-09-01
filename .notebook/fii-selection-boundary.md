# FII Selection Boundary
> Individual FII selection remains separate from allocation class weighting

Entry: `py/run_fii.py:main()` → `py/fii_selection.py:run_fii_selection()`

Stock flow remains: `py/core/preprocessing.py` → `py/core/scoring.py` →
`py/core/optimizer.py`. FII flow is isolated in
`py/fetch_status_invest_fii.py` and `py/fii_selection.py`; it reuses only
`core.optimizer.GeneticAlgorithm` and `core.metrics.hhi_sector`.

Source contract: `AdvancedSearchResultExport` with
`search={"Segment":"<id>"}` and `CategoryType=2`; FII page referer is
`/fundos-imobiliarios/busca-avancada`. Verified `Agências de Bancos` as ID 91.

Current populated segment map lives in `py/fetch_status_invest_fii.py:FII_SEGMENTS`:
IDs 23, 62, 87–100 (with current gaps), 103, and 108; 18 labels total.
Live refresh on 2026-08-10: 588 rows, 18 segments, no duplicate tickers.

FII output: `data/raw/status_invest_fii.csv` →
`data/processed/fii_clean_<profile>.csv` →
`outputs/carteira_fii_<profile>_consensus.json`. Parser canonicalizes source
typo `CAGR VALOR CORA 3 ANOS` to `CAGR VALOR COTA 3 ANOS` and preserves text
`GESTAO`.

Eligibility: positive `PRECO`, `P/VP`, `LIQUIDEZ MEDIA DIARIA`, `PATRIMONIO`,
`N COTISTAS`, and `N COTAS`; non-negative `DY`. Live `caio` run: 364 eligible,
284 scoreable, 10 selected. The selector does not assign final portfolio
weights; the later allocation optimizer owns that decision.

FII score groups/config are independent in `py/fii_selection.py`; defaults map
existing profile proportions to liquidity, size/cash, value, growth, and
dividend groups. `py/run_fii.py` exposes stock-style `--once`, `--quick`,
`--production`, and `--max-quality` presets; `--runs` remains the custom
sequential override. Allocation weights are intentionally absent from this
output.

Do not merge FII rows into stock source or change stock defaults. Allocation
reads the fixed FII consensus artifact as its `fiis` sleeve; IFIX remains
optional benchmark-only data. Historical FII re-selection remains deferred.

Updated: 2026-08-11
