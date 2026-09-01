# Project

## Purpose

This repository analyzes Brazilian equities for a personal investment profile using fundamental data, score-based ranking, and a genetic algorithm. The next capability is a separate analysis of how Caio's total capital could be distributed across asset classes.

## Current feature

`caio-asset-allocation` evaluates five benchmark sleeves in BRL:

- Brazilian stocks, represented by the existing Caio consensus portfolio.
- Brazilian real-estate funds, represented by B3 IFIX total return.
- International equity exposure, represented by S&P 500 total return converted from USD to BRL.
- Post-fixed fixed income, represented by BCB SGS 12 daily CDI factors cross-checked against B3 DI.
- Crypto, represented by Bitcoin converted to BRL.

The feature is independent of the stock-selection objective. It consumes the existing Caio portfolio as a fixed sleeve and does not select individual stocks again.

## Explicit non-goals

The first version excludes taxes, operating costs, contribution/withdrawal flows, class-specific maximums, a detailed tax engine, and a global-equity benchmark. BDRs and ETFs are access instruments, not additional classes.

## SaaS direction (draft)

The product direction is a Brazil-first portfolio decision assistant for individual investors. It generates a general allocation across five asset classes, performs deeper selection and weighting inside the Brazilian-stock sleeve, and helps the user monitor drift and review rebalancing actions.

The SaaS MVP is intentionally narrow:

- one user and one primary portfolio;
- five-class target allocation using the existing allocation core;
- stock-level analysis only for Brazilian equities;
- manual portfolio input first;
- monthly analysis and drift review, without order execution;
- recurring subscription with provider-neutral entitlements.

The personalized recommendation release remains subject to Brazilian securities-law and compliance review. B3/Área do Investidor integration is a later import feature, not an MVP dependency.
