# LESSONS - auto-maintained by scripts/lessons.py

> Machine-owned. Do NOT hand-edit. Changes are overwritten on the next `lessons.py` write.
> Canonical state lives in `.specs/lessons.json`. Edit lessons only via the script.
> promote_threshold=2 distinct features · window_days=45 · quarantine_threshold=2

## Confirmed (load these at Specify/Design)

Corroborated across multiple features. Safe to apply as guidance.

_none_

## Candidates (under observation - do NOT load as guidance yet)

Seen once or not yet corroborated. Tracked, not trusted.

### L-001 - Give each refusal guard its own test input that only that guard rejects, so an earlier guard cannot mask it
- signal: `surviving_mutant` · recurrence: 1 feature(s) · scope: `scripts` · harmful: 0
- features: protection-zone
- evidence: M65 tests/test_protection_zone_turnover.py:246 (scripts)
- last seen: 2026-11-05T01:03:15Z

### L-002 - When implementation narrows a spec rule to stay consistent with another AC, update the spec text in the same feature instead of leaving only a code marker
- signal: `spec_deviation` · recurrence: 1 feature(s) · scope: `specs` · harmful: 0
- features: protection-zone
- evidence: SPEC_DEVIATION py/core/protection_zone.py:116 (specs)
- last seen: 2026-11-05T01:03:15Z

### L-003 - Assert the exact arguments of a defensive query filter even when another filter currently makes it redundant
- signal: `surviving_mutant` · recurrence: 1 feature(s) · scope: `api` · harmful: 0
- features: protection-zone
- evidence: M46 api/app/services/premium_executor.py:176 (api)
- last seen: 2026-11-05T01:03:16Z

### L-004 - Source-regex web tests must assert every call site and every rendered field a UI criterion names, not one shared component body
- signal: `surviving_mutant` · recurrence: 1 feature(s) · scope: `web` · harmful: 0
- features: premium-whole-units
- evidence: validation.md W1/W3/W4/W5 (web/components/recommendation/recommendation-view.tsx:84,96,289,299) (web)
- last seen: 2026-11-09T17:49:05Z

## Quarantined (failed when applied - ignore)

A confirmed lesson that recurred alongside failure. Kept for the maintainer to review.

_none_
