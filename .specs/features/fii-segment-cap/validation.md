# FII Segment Cap and Policy v4 Validation

**Date**: 2026-11-06
**Spec**: `.specs/features/fii-segment-cap/spec.md`
**Diff range**: a61164a..725daf4 (branch t3/issue-56)
**Verifier**: independent sub-agent (author != verifier), request iw-56-verify-r1

**Verdict**: PASS

## Task Completion

T1-T14 all `**Status**: Done` in tasks.md (header still says `Draft`, see N1).

## Spec-Anchored Acceptance Criteria

| Req | Spec outcome | Evidence | Result |
| --- | --- | --- | --- |
| SEG-01/02 | `premium-policy-v4`; v3 rejected | `api/tests/test_premium_policy.py:414`, `:267`; impl `api/app/services/premium_policy.py:12` | PASS |
| SEG-03/05/06 | FIIs 4/5/6, stocks 10/12/15; owner 0.70 -> fiis 5, cap 2, zone {k_offset 3, K 8, delta 0.11}, stocks 13 | `api/tests/test_premium_policy.py:397`, `:404`, `:325` | PASS |
| SEG-04 | fiis.max_per_segment 2 every score; stocks None, absent from dict | `api/tests/test_premium_policy.py:377`, `:387` | PASS |
| SEG-07 | model_versions fiis `fii-selection-v2` | `api/tests/test_premium_policy.py:422`; impl `api/app/services/premium_policy.py:102` | PASS |
| SEG-08 | Basic v2, 10/12/15, golden JSON, fii-selection-v1 | `api/tests/test_premium_policy.py:361` (unchanged), `api/tests/test_premium_optimization.py` basic_versions subtest | PASS |
| SEG-09..14 | GA -inf over cap, same fitness at cap, optimize capped, ValueError, golden uncapped | `tests/test_segment_cap_optimizer.py:63-115`; impl `py/core/optimizer.py:142-146` | PASS |
| SEG-15..18 | consensus skip; two of three; Indefinido; uncapped == pre-change oracle | `tests/test_segment_cap_fii.py:191`, `:197`, `:203`; `tests/test_segment_cap_zone.py:28-46` | PASS |
| SEG-19..22 | teto departure; segment cut before capacity; fill skip; challenger skip | `tests/test_segment_cap_zone.py:146-196` | PASS |
| SEG-23/24 | capped K=0 consensus recorded and used; None cap unchanged | `tests/test_segment_cap_zone.py:343`, `:352`, `:238`, `:396` | PASS |
| SEG-25 | ValueError naming field | `tests/test_segment_cap_zone.py:53` | PASS |
| SEG-26 | run_fii_selection departure `saiu: teto do segmento` | `tests/test_segment_cap_fii.py:337` | PASS |
| SEG-27..29 | cap to every GA; capacity check before GA (+ exact boundary); post-check | `tests/test_segment_cap_fii.py:102`, `:353`, `:363`, `:371`; `api/tests/test_premium_policy.py` V4FiiExecutabilityTests | PASS |
| SEG-30..33 | adapter forwards int; policy_invalid; no key without cap; e2e v4 run | `api/tests/test_premium_optimization.py:1004`, `:1016`, `:1031`; `api/tests/test_premium_integration.py:490`, `:514` | PASS |
| SEG-34..36 | protocol guard exit 2; turnover v4; docs/validation untouched | `tests/test_multirun_convergence.py:531`; `tests/test_protection_zone_turnover.py:164`; `git diff --name-only a61164a..HEAD -- docs/validation` = 0 | PASS |
| SEG-37..43 | HHI skip in consensus/fill/challenger/protection walk; raise not short sleeve; no hhi_max without cap; real data owner+limitar 8x40x30 seeds 0-3; propagate SegmentCapInfeasibleError | `tests/test_segment_cap_zone.py:67-104`, `:258-306`, `:367`, `:381`; `tests/test_segment_cap_fii.py:226-250`; `api/tests/test_premium_policy.py:793`, `:804` | PASS |

## Discrimination Sensor

40 behaviour mutants on throwaway copies (git archive HEAD), never the real tree. 40/40 killed.
Critical paths: `py/core/optimizer.py` M01-M04, `py/core/protection_zone.py` M05-M23, `api/app/adapters/premium_optimization.py` M24-M26 — all killed.
Non-critical: `py/fii_selection.py` M27-M34, `api/app/services/premium_policy.py` M35-M40 — all killed.
Real-tree `git status --porcelain` empty before and after.

## Gate Check

- Root: `python -m unittest discover -s tests -p "test_*.py"` -> Ran 587, OK (skipped=2); baseline 535 (+52).
- API: `PYTHONPATH=.:../py python -m unittest discover -s tests -t .` -> Ran 174, OK; baseline 160 (+14).
- Existing tests changed only where A-12/A-10 allow; Basic tests byte-identical.
- No diff under data/, outputs/, docs/validation/, baselines/.

## Nonblocking notes

- N1 tasks.md header `**Status**: Draft` while all tasks are Done.
- N2 `run_fii_selection` does not type-check `max_per_segment` before `_require_segment_capacity` (`py/fii_selection.py:747`); a non-int from an offline caller reaches the GA's ValueError instead of FiiSelectionError. The API adapter rejects it first (SEG-31).
- N3 spec Goals/Success checkboxes left unticked; validate_spec warns on open-question wording (Q1/Q2 carry defaults).
