# ADR 0001: Use benchmark sleeves and a deterministic frontier for class allocation

## Status

Accepted

## Context

The repository's genetic algorithm selects individual Brazilian stocks from fundamental scores. The new question is how to distribute Caio's whole capital across asset classes. Reusing the stock GA would mix security-selection signals with heterogeneous benchmark series and would not provide a transparent diversification trade-off.

## Decision

Represent each class with one documented total-return benchmark in BRL, keep the existing Caio stock consensus as a fixed sleeve, and search non-negative weight vectors on a deterministic simplex grid. Apply volatility and drawdown caps to the training period, expose the non-dominated return/concentration frontier, and use the mathematically defined knee as the default target.

The first proxies are the B3 IFIX total-return index for FIIs, the BCB SGS 12 daily CDI factor cross-checked against B3 DI for post-fixed fixed income, S&P 500 Total Return converted with BCB PTAX for international equity, and BTC/USD converted with the same PTAX rule for crypto. Taxes, costs, and cash flows remain outside this version.

## Consequences

Positive:

- The allocation objective is independent from the stock-selection objective.
- The output is reproducible, inspectable, and shows alternative risk/diversification trade-offs.
- A zero allocation is possible without adding arbitrary class caps.

Trade-offs:

- A single benchmark can misrepresent the full class, especially FIIs, fixed income, and crypto.
- Historical optimization is exposed to regime and proxy risk; walk-forward evaluation and 5-year robustness reporting are required.
- Grid search is less flexible than continuous optimization, but its small five-class simplex is easy to audit and deterministic.

## Revisit when

The user wants multiple instruments per class, continuous constraints, taxes/costs, or a global equity universe. Those changes require a new specification rather than silently changing the meaning of this analysis.

## Source references

- [B3 IFIX methodology](https://www.b3.com.br/data/files/04/E6/A1/D3/762915107623A41592D828A8/IFIX-Metodologia-en-us.pdf)
- [B3 DI methodology](https://www.b3.com.br/pt_br/market-data-e-indices/indices/indices-de-segmentos-e-setoriais/metodologia-do-di.htm)
- [BCB SGS CDI series 12 API](https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados?formato=json)
- [S&P 500 official index page](https://www.spglobal.com/spdji/en/indices/equity/sp-500/)
- [BCB PTAX open data](https://dadosabertos.bcb.gov.br/dataset/?res_format=API&res_format=OData&tags=ptax)
