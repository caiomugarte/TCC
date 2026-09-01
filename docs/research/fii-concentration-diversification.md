# FII concentration and diversification research

**Accessed:** August 11, 2026
**Scope:** Whether the FII sleeve should contain fewer funds than the stock
sleeve, and whether FII holdings should use unequal weights.

## Conclusion

Research does not support a universal rule such as “FIIs must be 40% of the
number of stocks.” The relevant unit is effective exposure, not ticker count:
one FII can hold one property, many properties, CRIs, or other FIIs, and two
different FIIs can share the same sector, manager, issuer, tenant, or credit
risk.

Do not change the optimizer yet. Test a smaller FII candidate set and unequal
weights as separate alternatives. Keep the smallest configuration that holds
up out of sample on return, volatility, drawdown, concentration, turnover, and
selection stability.

## What the sources say

### Regulation and market structure

- The current [CVM Anexo Normativo III to Resolution 175](https://conteudo.cvm.gov.br/cvm_institucional/export/sites/cvm/legislacao/resolucoes/anexos/100/resol175consolid_Anexo03.pdf)
  requires each FII regulation to state its eligible assets and diversification
  requirements (Article 11, II, c). It does not prescribe one investor-level
  number of FIIs. The same rule permits an FII to buy other FII quotas (Article
  40, VI).
- The [CVM investor overview](https://www.gov.br/investidor/pt-br/investir/tipos-de-investimentos/fundos-de-investimentos-imobiliarios-fii)
  describes FIIs as vehicles that may invest in real estate and in securities
  related to the real-estate market. Therefore, equal ticker counts do not imply
  equal underlying diversification.
- The [B3 IFIX methodology](https://www.b3.com.br/data/files/2A/56/E3/DD/A3943710DB551337AC094EA8/IFIX-Metodologia-pt-br.pdf)
  weights funds by market capitalization and caps an individual fund at 20%
  during inclusion or periodic reviews. This is a benchmark rule, not proof
  that a personal portfolio should use market-cap weights, but it confirms that
  equal weighting is a design choice rather than a market requirement.

### Evidence specific to Brazilian FIIs

- [Moraes and Serra (2017)](https://revistas.unisinos.br/index.php/base/article/view/base.2017.141.05)
  studied 22 multi-property Brazilian FIIs. Larger funds were more diversified;
  the number of properties and property concentration were not significant in
  their model. The sample covered one period, so this does not establish a
  universal rule.
- [Bortoluzzo, Silva Neto, and Bortoluzzo (2020)](https://periodicos.ufpb.br/index.php/recfin/article/view/44848)
  used a panel of 110 Brazilian REITs. After excluding mortgage REITs,
  property-type diversification had a positive and statistically relevant
  relation with performance. “Diversification” here means property type, not
  number of listed FII tickers.
- [Teixeira, Forte, and Louzada (2024)](https://revistas.metodista.br/index.php/organizacoesemcontexto/article/view/24)
  compared specialized and diversified Brazilian FIIs from 2011 to 2018. The
  diversified group had higher volatility and higher return in that sample.
  More diversification therefore is not automatically better.
- [Bernardo, Campani, and Roquete (2023)](https://doi.org/10.1080/10835547.2023.2189509)
  found that Brazilian REIT correlations with stocks and government bonds vary
  over time. FII diversification benefits should therefore be evaluated with
  time-varying data, not only with a static ticker count.

### General portfolio evidence

- [Evans and Archer (1968)](https://doi.org/10.1111/j.1540-6261.1968.tb00315.x)
  and [Elton and Gruber (1977)](https://doi.org/10.1086/295964) show declining
  marginal risk-reduction benefits as more securities are added. These studies
  concern stocks, not Brazilian FIIs.
- [Statman (1987)](https://www.cambridge.org/core/product/identifier/S0022109000012680/type/journal_article)
  reached a much higher 30-to-40-stock range under its assumptions. The
  disagreement itself matters: there is no defensible magic number independent
  of asset correlations, investor constraints, costs, and objective.
- [Woerheide and Persson (1992)](https://openjournals.libs.uga.edu/fsr/article/view/3710)
  found the complement of the Herfindahl index more useful than raw security
  count for unevenly weighted portfolios. Use effective number of holdings,
  `N_eff = 1 / sum(weight_i ** 2)`, alongside the actual number of funds.

## Implications for this repository

The current [FII selector](../../py/fii_selection.py) uses fixed counts of 10,
12, and 15 for the generic profiles, and 10, 14, and 11 for the Caio profiles.
Those values mirror the stock configurations. The binary GA selects exactly
`n_assets`, and the result contains no `WEIGHT` column; the [sector HHI metric](../../py/core/metrics.py)
explicitly assumes equal asset weights.

That creates three limitations:

1. The optimizer controls sector counts, not total FII exposure concentration.
2. Equal weights make 11 FIIs look like 11 effective holdings even when their
   underlying exposures overlap heavily.
3. The current dataset has sector labels, but not enough look-through data to
   measure common issuers, properties, tenants, geography, or FOF overlap.

The concern is valid, but the first change should not be “copy stock count to a
smaller FII count.” It should decouple FII `n_assets` from stock profiles and
measure the exposures that actually drive risk.

## Minimal experiment before changing production defaults

Run the existing pipeline with `n_assets` in `{6, 8, 10, 12, 14}`. For each
value, compare:

- equal weights;
- score-tilted weights with explicit per-FII caps, for example 10%, 15%, and
  20% of the FII sleeve;
- out-of-sample annualized return, volatility, maximum drawdown, and turnover;
- ticker HHI, sector HHI, effective number of FIIs, and selection stability;
- overlap with the stock sleeve and the IFIX benchmark.

Use the [CVM open portfolio data](https://dados.cvm.gov.br/dataset/fi-doc-cda)
for look-through fields where available. Prefer the smallest FII count that
does not materially worsen out-of-sample risk-adjusted results or stability.

Until that experiment exists, retain current defaults and treat “fewer FIIs
than stocks” as a hypothesis, not a rule.

## Limits of this search

The Brazilian FII studies use older samples, small samples, or definitions of
diversification that differ from this optimizer. None estimates an optimal
number of FIIs for this portfolio. The recommendation is an inference from
those sources and the repository's current equal-weight implementation.
