# Context

**Status:** Draft; awaiting user confirmation before Design.

## User direction

- The MVP should optimize five asset classes and perform deeper analysis only
  inside the Brazilian-stock sleeve.
- The user wants exact target amounts, not only percentages:
  - amount per asset class;
  - amount per supported stock;
  - percentage inside the stock sleeve and percentage in the total portfolio.
- Generic profiles are candidates for Basic; a customized continuous profile is
  a candidate for Premium.

## Proposed decisions

- Treat the exact personalized output as the regulated-advice product lane for
  launch planning.
- Ask for investable capital separately from total patrimony.
- Publish `target_weight`, `target_amount_brl`, and model metadata for every
  class; publish the equivalent fields for every stock, including the two
  percentage bases.
- Use a target amount plus a tolerance band for tracking. The band is for drift
  review, not a replacement for the exact target.
- Keep automatic execution, broker credentials, tax optimization, and stock
  selection for other classes out of this MVP.

## Compliance gate

Before public personalized recommendations, obtain Brazil-first legal review
and choose one operating model: authorized own activity, authorized partner,
or a controlled private pilot with approved wording and responsibility. A
disclaimer alone is not the product boundary.

## Open decisions

- Legal operating model and responsible regulated professional.
- Initial supported stock universe and data freshness rule.
- Initial drift tolerance band.
- Whether the Basic tier may show only generic model portfolios or also
  personalized diagnostics.
