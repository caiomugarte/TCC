# Allocation snapshot layout

The first allocation runner is offline by design. Put a reproducible snapshot
in this directory (or pass another directory to the CLI) with these files:

```text
metadata.json
caio_stocks.csv
caio_fiis.csv
[ifix.csv]                 # optional benchmark-only data
sp500_total_return_usd.csv
di.csv
btc_usd.csv
ptax.csv
```

Every CSV has a `date` column in ISO format and one or more positive level
columns. `caio_stocks.csv` must contain one column for every ticker in
`outputs/carteira_caio_consensus.json`. `caio_fiis.csv` must contain one
column for every ticker in `outputs/carteira_fii_caio_consensus.json`.
The other files normally use a single `value` column. The international and
crypto files are native USD levels;
`ptax.csv` is BRL per USD and is multiplied on exact common dates.

`metadata.json` must contain at least:

```json
{
  "source": "documented source or provider",
  "retrieved_at": "2026-07-21T12:00:00-03:00",
  "cutoff_date": "2026-07-20",
  "notes": "distribution and currency treatment"
}
```

Missing dates are intersected, never forward-filled. A permanent Yahoo no-data
response (for example, HTTP 404) is recorded separately for stock and FII
tickers in `metadata.json`; remaining valid tickers are reweighted equally
inside their respective sleeves. Transient fetch failures remain fatal. No
ticker is replaced in the stock optimizer, FII selector, or consensus
portfolio. IFIX is optional benchmark-only data and never supplies the `fiis`
class when `caio_fiis.csv` is present. Keep the snapshot with the analysis
outputs so you can reproduce the result.

Suggested source documentation for the first snapshot is the [B3 IFIX
methodology](https://www.b3.com.br/data/files/04/E6/A1/D3/762915107623A41592D828A8/IFIX-Metodologia-en-us.pdf),
[B3 DI methodology](https://www.b3.com.br/pt_br/market-data-e-indices/indices/indices-de-segmentos-e-setoriais/metodologia-do-di.htm),
[S&P 500 official index page](https://www.spglobal.com/spdji/en/indices/equity/sp-500/),
the [BCB SGS CDI series 12 API](https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados?formato=json),
and [BCB PTAX open data](https://dadosabertos.bcb.gov.br/dataset/?res_format=API&res_format=OData&tags=ptax).
The crypto provider is intentionally selected and recorded per snapshot rather
than hard-coded into the optimizer.

To acquire the first snapshot and run the analysis:

```bash
python3 py/fetch_allocation_snapshot.py \
  --start-date 2016-07-21 --end-date 2026-07-21
python3 py/run_allocation.py
```

The CLIs default to `outputs/carteira_fii_caio_consensus.json`. Pass
`--fii-portfolio PATH` to use another fixed FII artifact. Pass `--skip-ifix`
when you do not need the optional benchmark file.

For a personalized allocation, pass the continuous suitability score. The
runner interpolates the documented allocation-only anchor policies and records
the derived limits in the output:

```bash
python3 py/run_allocation.py \
  --snapshot-dir data/allocation_caio_new \
  --portfolio outputs/carteira_caio_new_consensus.json \
  --suitability-score 0.831 \
  --profile-name caio_new \
  --output-prefix allocation_caio_new_personalized
```

The fixed-income file uses the BCB SGS 12 daily CDI factor. A sample was
cross-checked against the B3 DI FTP annual-rate file using B3's 252-business-day
convention. The model remains gross and excludes taxes, fees, spreads, and
cash flows. Allocation candidates currently require a 5% minimum in every
sleeve.
