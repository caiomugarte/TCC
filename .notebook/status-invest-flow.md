# Status Invest sync
> Manual sector export refresh for stock fundamentals

Entry: `py/fetch_status_invest.py:refresh()`

Flow: ten fixed sector IDs → `category/AdvancedSearchResultExport` → semicolon
CSV → Brazilian numeric normalization → `data/raw/status_invest_fundamentals.csv`

Sector IDs: 2, 3, 10, 1, 5, 4, 8, 7, 6, 9. Source export has no `SETOR`; the
client assigns the existing project label by ID.

Gotchas:
- Export needs browser-like `User-Agent` and advanced-search `Referer` headers.
- Export endpoint takes filter and category type, not pagination; pagination is
  only for displayed results.
- Output replacement is atomic and happens only after all sectors validate.

Latest verified refresh: 617 ticker rows, 10 sectors, 31 columns, no duplicate
tickers.

Updated: 2026-08-09
