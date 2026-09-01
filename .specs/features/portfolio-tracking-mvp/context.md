# Context

**Status:** Draft; awaiting recommendation and pricing confirmation before Design.

## User direction

- Tracking the current portfolio makes the recurring product useful and is in
  scope for the MVP.
- Manual input is acceptable first; B3/Área do Investidor import is later.
- Rebalancing should compare current holdings with the class and stock targets.

## Proposed decisions

- Support one primary portfolio with manual positions and CSV import.
- Show current weight, target weight, signed drift, current value, and target
  value when prices are available.
- Keep stock sleeve weight and total-portfolio weight separate in the UI.
- Recommend new contributions before suggesting sales. Sales remain review
  actions and never become orders in this MVP.
- Make tracking a Premium value driver after the user has a valid target and
  portfolio snapshot.

## Open decisions

- Initial drift band; proposed default is five percentage points.
- Class-level representation for non-stock holdings.
- Manual refresh versus scheduled monthly refresh.
- Basic versus Premium tracking boundary.
