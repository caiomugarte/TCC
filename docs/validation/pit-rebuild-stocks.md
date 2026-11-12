# Reconstrução PIT dos indicadores de ações

- Esquema: `pit-rebuild-stocks/v3` (issue #59, correções da #64)
- Commit da ferramenta: `550c89f312a3bb50dc66c5f23b48b3d813872d40`
- Datas de decisão: 2026-08-07, 2026-11-05

## Veredito

Regra de veredito da #59, aplicada aos valores com as correções da #64; os números da v2 estão no commit `ebaf1fd`.

**Veredito da classe:** `não reconstruível`. Os tickers não classificados não mudam o veredito.

**Grupo financeiro:** `não reconstruível`. Ele é calculado da mesma forma e não altera a classe.

Escopo da classe: 16 indicadores de METRIC_GROUPS, grupo não financeiro incluindo os não classificados.

### Veredito por indicador

| indicador | não financeiro | não financeiro sem não classificados | financeiro |
| --- | --- | --- | --- |
| `LIQ. CORRENTE` | com ressalvas | com ressalvas | não reconstruível |
| `DIVIDA LIQUIDA / EBIT` | com ressalvas | com ressalvas | não reconstruível |
| `DIV. LIQ. / PATRI.` | com ressalvas | com ressalvas | não reconstruível |
| `ROE` | com ressalvas | com ressalvas | não reconstruível |
| `ROA` | com ressalvas | com ressalvas | não reconstruível |
| `ROIC` | não reconstruível | não reconstruível | não reconstruível |
| `MARG. LIQUIDA` | com ressalvas | com ressalvas | não reconstruível |
| `MARGEM EBIT` | com ressalvas | com ressalvas | não reconstruível |
| `P/L` | não reconstruível | não reconstruível | não reconstruível |
| `P/VP` | com ressalvas | com ressalvas | não reconstruível |
| `EV/EBIT` | não reconstruível | não reconstruível | não reconstruível |
| `PSR` | com ressalvas | com ressalvas | com ressalvas |
| `CAGR RECEITAS 5 ANOS` | com ressalvas | com ressalvas | com ressalvas |
| `CAGR LUCROS 5 ANOS` | com ressalvas | com ressalvas | não reconstruível |
| `PEG RATIO` | não reconstruível | não reconstruível | não reconstruível |
| `DY` | não reconstruível | com ressalvas | com ressalvas |
| `PRECO` | reconstruível | reconstruível | reconstruível |
| `VALOR DE MERCADO` | com ressalvas | com ressalvas | com ressalvas |
| `LIQUIDEZ MEDIA DIARIA` | com ressalvas | com ressalvas | com ressalvas |

## Veredito ajustado (#64)

14 indicadores do veredito ajustado (sem ROIC e PEG RATIO), grupo não financeiro incluindo os não classificados. Por data, os tickers elegíveis sem os tickers de referência obsoleta (`reference_stale`). Os tickers com `si_lag` ficam no denominador como fora da tolerância.

| indicador | 2026-08-07: % na tolerância (n) | 2026-08-07: si_lag | 2026-11-05: % na tolerância (n) | 2026-11-05: si_lag | veredito |
| --- | --- | --- | --- | --- | --- |
| `LIQ. CORRENTE` | 97,69% (216) | 1 | 84,82% (224) | 30 | com ressalvas |
| `DIVIDA LIQUIDA / EBIT` | 94,44% (216) | 4 | 77,68% (224) | 35 | com ressalvas |
| `DIV. LIQ. / PATRI.` | 96,23% (212) | 1 | 84,72% (216) | 27 | com ressalvas |
| `ROE` | 93,52% (216) | 1 | 75,89% (224) | 40 | com ressalvas |
| `ROA` | 95,37% (216) | 3 | 81,25% (224) | 31 | com ressalvas |
| `MARG. LIQUIDA` | 92,99% (214) | 2 | 77,48% (222) | 36 | com ressalvas |
| `MARGEM EBIT` | 94,86% (214) | 1 | 80,18% (222) | 30 | com ressalvas |
| `P/L` | 87,50% (216) | 5 | 72,77% (224) | 35 | com ressalvas |
| `P/VP` | 90,74% (216) | 4 | 86,16% (224) | 13 | com ressalvas |
| `EV/EBIT` | 87,96% (216) | 4 | 74,11% (224) | 31 | com ressalvas |
| `PSR` | 88,79% (214) | 4 | 86,04% (222) | 14 | com ressalvas |
| `CAGR RECEITAS 5 ANOS` | 96,70% (212) | 0 | 96,82% (220) | 0 | reconstruível |
| `CAGR LUCROS 5 ANOS` | 93,92% (148) | 2 | 76,97% (152) | 28 | com ressalvas |
| `DY` | 76,39% (216) | 0 | 82,14% (224) | 0 | com ressalvas |

Critério de liberação (#64), parte 1 (≥ 70 % nas duas datas): cumprida.

Parte 2 (10 piores conferidos e explicados): ver `pit-rebuild-stocks-audit.md`.

<details>
<summary>LIQ. CORRENTE, 2026-08-07</summary>

| ticker | referência | reconstruído | diferença | si_lag | arquivo | valor anterior |
| --- | --- | --- | --- | --- | --- | --- |
| `VTRU3` | 4,48 | 2,13 | 52,46% | não | ITR 2026-03-31 v1, recebido em 2026-05-06 | 2,23 |
| `ONCO3` | 0,46 | 0,56 | 21,74% | não | DFP 2025-12-31 v1, recebido em 2026-04-09 | - |
| `BRST3` | 1,54 | 1,31 | 14,94% | sim | ITR 2026-03-31 v1, recebido em 2026-05-12 | 1,49 |
| `BHIA3` | 0,76 | 0,72 | 5,26% | não | ITR 2026-03-31 v1, recebido em 2026-05-13 | 0,66 |
| `TIMS3` | 0,89 | 0,85 | 4,49% | não | ITR 2026-06-30 v1, recebido em 2026-07-27 | 0,92 |
| `CAML3` | 1,95 | 1,87 | 4,10% | não | ITR 2026-05-31 v1, recebido em 2026-07-14 | 1,95 |
| `AALR3` | 0,28 | 0,28 | 0,00% | não | DFP 2025-12-31 v1, recebido em 2026-07-30 | - |
| `ABEV3` | 1,02 | 1,02 | 0,00% | não | ITR 2026-06-30 v1, recebido em 2026-07-30 | 1,03 |
| `AGRO3` | 1,47 | 1,47 | 0,00% | não | ITR 2026-03-31 v1, recebido em 2026-05-07 | 1,61 |
| `ALLD3` | 1,74 | 1,74 | 0,00% | não | ITR 2026-03-31 v1, recebido em 2026-05-07 | 1,80 |

Sem valor reconstruído: `AMBP3` (stale_filing).

</details>

<details>
<summary>LIQ. CORRENTE, 2026-11-05</summary>

| ticker | referência | reconstruído | diferença | si_lag | arquivo | valor anterior |
| --- | --- | --- | --- | --- | --- | --- |
| `VTRU3` | 4,48 | 2,56 | 42,86% | não | ITR 2026-06-30 v1, recebido em 2026-08-11 | 2,13 |
| `PRIO3` | 1,34 | 1,83 | 36,57% | sim | ITR 2026-09-30 v1, recebido em 2026-11-04 | 1,34 |
| `SUZB3` | 3,38 | 2,54 | 24,85% | sim | ITR 2026-09-30 v1, recebido em 2026-11-05 | 3,38 |
| `DASA3` | 1,27 | 1,58 | 24,41% | sim | ITR 2026-09-30 v1, recebido em 2026-11-05 | 1,27 |
| `RECV3` | 2,91 | 2,28 | 21,65% | sim | ITR 2026-09-30 v1, recebido em 2026-11-05 | 2,91 |
| `CASH3` | 2,68 | 3,24 | 20,90% | sim | ITR 2026-09-30 v1, recebido em 2026-11-04 | 2,68 |
| `SMTO3` | 2,62 | 3,16 | 20,61% | sim | ITR 2026-06-30 v1, recebido em 2026-08-10 | 2,62 |
| `ALUP11` | 3,17 | 2,53 | 20,19% | sim | ITR 2026-09-30 v1, recebido em 2026-11-05 | 3,17 |
| `ALUP3` | 3,17 | 2,53 | 20,19% | sim | ITR 2026-09-30 v1, recebido em 2026-11-05 | 3,17 |
| `ALUP4` | 3,17 | 2,53 | 20,19% | sim | ITR 2026-09-30 v1, recebido em 2026-11-05 | 3,17 |

Sem valor reconstruído: `AALR3` (stale_filing), `AMBP3` (stale_filing).

</details>

<details>
<summary>DIVIDA LIQUIDA / EBIT, 2026-08-07</summary>

| ticker | referência | reconstruído | diferença | si_lag | arquivo | valor anterior |
| --- | --- | --- | --- | --- | --- | --- |
| `BHIA3` | -14,38 | 4,48 | 131,15% | não | ITR 2026-03-31 v1, recebido em 2026-05-13 | 3,54 |
| `TIMS3` | -0,73 | -0,28 | 61,64% | não | ITR 2026-06-30 v1, recebido em 2026-07-27 | -0,46 |
| `CAML3` | 5,02 | 7,45 | 48,41% | sim | ITR 2026-05-31 v1, recebido em 2026-07-14 | 5,16 |
| `VTRU3` | 3,68 | 2,31 | 37,23% | não | ITR 2026-03-31 v1, recebido em 2026-05-06 | 2,73 |
| `BRST3` | 5,03 | 6,90 | 37,18% | não | ITR 2026-03-31 v1, recebido em 2026-05-12 | 6,81 |
| `EMAE4` | 1,28 | 1,45 | 13,28% | sim | ITR 2026-06-30 v1, recebido em 2026-08-07 | 1,24 |
| `SOJA3` | 7,81 | 7,14 | 8,58% | não | ITR 2026-03-31 v1, recebido em 2026-05-13 | 3,49 |
| `MOTV3` | 4,18 | 3,88 | 7,18% | sim | ITR 2026-06-30 v1, recebido em 2026-07-29 | 4,18 |
| `TPIS3` | -2,47 | -2,62 | 6,07% | sim | ITR 2026-03-31 v1, recebido em 2026-05-15 | -2,58 |
| `AMER3` | 3,69 | 3,88 | 5,15% | não | ITR 2026-03-31 v1, recebido em 2026-05-13 | 2,84 |

Sem valor reconstruído: `AMBP3` (stale_filing), `EQPA3` (missing_input: EBIT).

</details>

<details>
<summary>DIVIDA LIQUIDA / EBIT, 2026-11-05</summary>

| ticker | referência | reconstruído | diferença | si_lag | arquivo | valor anterior |
| --- | --- | --- | --- | --- | --- | --- |
| `EQPA3` | 0,32 | 2,61 | 715,62% | não | ITR 2026-06-30 v1, recebido em 2026-08-12 | 2,51 |
| `CASH3` | 1,63 | 13,20 | 709,82% | sim | ITR 2026-09-30 v1, recebido em 2026-11-04 | 1,63 |
| `AGRO3` | -23,56 | 54,38 | 330,81% | sim | ITR 2026-09-30 v1, recebido em 2026-11-03 | -23,56 |
| `LJQQ3` | 30,40 | -26,50 | 187,17% | sim | ITR 2026-09-30 v1, recebido em 2026-11-05 | 30,40 |
| `MDNE3` | 0,37 | 0,86 | 132,43% | sim | ITR 2026-09-30 v1, recebido em 2026-11-05 | 0,37 |
| `TGMA3` | 0,08 | 0,01 | 87,50% | sim | ITR 2026-09-30 v1, recebido em 2026-11-04 | 0,08 |
| `BHIA3` | -14,38 | -2,55 | 82,27% | não | ITR 2026-06-30 v1, recebido em 2026-08-16 | 4,48 |
| `GRND3` | -1,26 | -0,39 | 69,05% | sim | ITR 2026-09-30 v1, recebido em 2026-11-05 | -1,26 |
| `TIMS3` | -0,73 | -0,28 | 61,64% | não | ITR 2026-06-30 v1, recebido em 2026-07-27 | -0,46 |
| `TUPY3` | -5,30 | -8,42 | 58,87% | sim | ITR 2026-09-30 v1, recebido em 2026-11-05 | -5,30 |

Sem valor reconstruído: `AALR3` (stale_filing), `AMBP3` (stale_filing).

</details>

<details>
<summary>DIV. LIQ. / PATRI., 2026-08-07</summary>

| ticker | referência | reconstruído | diferença | si_lag | arquivo | valor anterior |
| --- | --- | --- | --- | --- | --- | --- |
| `AALR3` | 0,34 | -1,27 | 473,53% | não | DFP 2025-12-31 v1, recebido em 2026-07-30 | - |
| `VTRU3` | -0,54 | 0,39 | 172,22% | não | ITR 2026-03-31 v1, recebido em 2026-05-06 | 0,55 |
| `TIMS3` | -0,39 | -0,08 | 79,49% | não | ITR 2026-06-30 v1, recebido em 2026-07-27 | -0,12 |
| `BRST3` | 0,66 | 1,08 | 63,64% | não | ITR 2026-03-31 v1, recebido em 2026-05-12 | 1,07 |
| `BHIA3` | 2,15 | 3,29 | 53,02% | não | ITR 2026-03-31 v1, recebido em 2026-05-13 | 1,72 |
| `ONCO3` | 6,99 | 3,65 | 47,78% | não | DFP 2025-12-31 v1, recebido em 2026-04-09 | - |
| `CAML3` | 1,08 | 1,50 | 38,89% | sim | ITR 2026-05-31 v1, recebido em 2026-07-14 | 1,08 |
| `ABEV3` | -0,17 | -0,17 | 0,00% | não | ITR 2026-06-30 v1, recebido em 2026-07-30 | -0,18 |
| `AGRO3` | 0,66 | 0,66 | 0,00% | não | ITR 2026-03-31 v1, recebido em 2026-05-07 | 0,61 |
| `ALLD3` | 0,09 | 0,09 | 0,00% | não | ITR 2026-03-31 v1, recebido em 2026-05-07 | 0,03 |

Sem valor reconstruído: `AMBP3` (stale_filing).

</details>

<details>
<summary>DIV. LIQ. / PATRI., 2026-11-05</summary>

| ticker | referência | reconstruído | diferença | si_lag | arquivo | valor anterior |
| --- | --- | --- | --- | --- | --- | --- |
| `PCAR3` | 8,95 | -11,67 | 230,39% | sim | ITR 2026-09-30 v1, recebido em 2026-11-04 | 8,95 |
| `LJQQ3` | -0,16 | 0,20 | 225,00% | sim | ITR 2026-09-30 v1, recebido em 2026-11-05 | -0,16 |
| `VTRU3` | -0,54 | 0,29 | 153,70% | não | ITR 2026-06-30 v1, recebido em 2026-08-11 | 0,39 |
| `BHIA3` | 2,15 | -0,69 | 132,09% | não | ITR 2026-06-30 v1, recebido em 2026-08-16 | 3,29 |
| `MDNE3` | 0,09 | 0,20 | 122,22% | sim | ITR 2026-09-30 v1, recebido em 2026-11-05 | 0,09 |
| `TGMA3` | 0,03 | 0,00 | 100,00% | sim | ITR 2026-09-30 v1, recebido em 2026-11-04 | 0,03 |
| `TIMS3` | -0,39 | -0,08 | 79,49% | não | ITR 2026-06-30 v1, recebido em 2026-07-27 | -0,12 |
| `GRND3` | -0,11 | -0,03 | 72,73% | sim | ITR 2026-09-30 v1, recebido em 2026-11-05 | -0,11 |
| `BRST3` | 0,66 | 1,07 | 62,12% | não | ITR 2026-06-30 v1, recebido em 2026-08-11 | 1,08 |
| `CAML3` | 1,08 | 1,43 | 32,41% | não | ITR 2026-08-31 v1, recebido em 2026-10-08 | 1,50 |

Sem valor reconstruído: `AMBP3` (stale_filing).

</details>

<details>
<summary>ROE, 2026-08-07</summary>

| ticker | referência | reconstruído | diferença | si_lag | arquivo | valor anterior |
| --- | --- | --- | --- | --- | --- | --- |
| `ONCO3` | -720,24 | -402,23 | 318,01 p.p. | não | DFP 2025-12-31 v1, recebido em 2026-04-09 | - |
| `BHIA3` | -49,14 | -204,95 | 155,81 p.p. | não | ITR 2026-03-31 v1, recebido em 2026-05-13 | -107,71 |
| `VTRU3` | -29,57 | 29,95 | 59,52 p.p. | não | ITR 2026-03-31 v1, recebido em 2026-05-06 | 12,64 |
| `AALR3` | -544,08 | -494,46 | 49,62 p.p. | não | DFP 2025-12-31 v1, recebido em 2026-07-30 | - |
| `RAIZ4` | -326,87 | -309,75 | 17,12 p.p. | não | DFP 2026-03-31 v1, recebido em 2026-06-29 | -1393,79 |
| `INTB3` | 18,07 | 8,63 | 9,44 p.p. | sim | ITR 2026-06-30 v1, recebido em 2026-07-29 | 18,29 |
| `BRST3` | 7,44 | 2,53 | 4,91 p.p. | não | ITR 2026-03-31 v1, recebido em 2026-05-12 | 2,65 |
| `EMAE4` | -18,27 | -14,30 | 3,97 p.p. | não | ITR 2026-06-30 v1, recebido em 2026-08-07 | -20,58 |
| `CAML3` | 6,46 | 3,69 | 2,77 p.p. | não | ITR 2026-05-31 v1, recebido em 2026-07-14 | 4,92 |
| `TIMS3` | 15,91 | 17,29 | 1,38 p.p. | não | ITR 2026-06-30 v1, recebido em 2026-07-27 | 17,74 |

Sem valor reconstruído: `AMBP3` (stale_filing), `EQPA3` (missing_input: LL).

</details>

<details>
<summary>ROE, 2026-11-05</summary>

| ticker | referência | reconstruído | diferença | si_lag | arquivo | valor anterior |
| --- | --- | --- | --- | --- | --- | --- |
| `PCAR3` | -520,29 | -932,09 | 411,80 p.p. | sim | ITR 2026-09-30 v1, recebido em 2026-11-04 | -520,29 |
| `EQPA3` | 366,78 | 29,89 | 336,89 p.p. | não | ITR 2026-06-30 v1, recebido em 2026-08-12 | 27,75 |
| `BHIA3` | -49,14 | -159,69 | 110,55 p.p. | não | ITR 2026-06-30 v1, recebido em 2026-08-16 | -204,95 |
| `RAIZ4` | -325,38 | -241,22 | 84,16 p.p. | não | ITR 2026-06-30 v1, recebido em 2026-08-13 | -309,75 |
| `AZUL3` | -31,48 | 31,48 | 62,96 p.p. | não | ITR 2026-06-30 v2, recebido em 2026-08-19 | 119,00 |
| `VTRU3` | -29,57 | 27,23 | 56,80 p.p. | não | ITR 2026-06-30 v1, recebido em 2026-08-11 | 29,95 |
| `EMAE4` | -18,27 | -48,30 | 30,03 p.p. | não | ITR 2026-09-30 v1, recebido em 2026-11-04 | -14,30 |
| `CASH3` | -23,45 | -0,47 | 22,98 p.p. | sim | ITR 2026-09-30 v1, recebido em 2026-11-04 | -23,45 |
| `AMER3` | 9,10 | -6,74 | 15,84 p.p. | não | ITR 2026-06-30 v1, recebido em 2026-08-12 | 12,51 |
| `LJQQ3` | -67,62 | -82,54 | 14,92 p.p. | sim | ITR 2026-09-30 v1, recebido em 2026-11-05 | -67,62 |

Sem valor reconstruído: `AALR3` (stale_filing), `AMBP3` (stale_filing).

</details>

<details>
<summary>ROA, 2026-08-07</summary>

| ticker | referência | reconstruído | diferença | si_lag | arquivo | valor anterior |
| --- | --- | --- | --- | --- | --- | --- |
| `VTRU3` | -24,02 | 16,06 | 40,08 p.p. | não | ITR 2026-03-31 v1, recebido em 2026-05-06 | 6,07 |
| `ONCO3` | -57,34 | -50,82 | 6,52 p.p. | não | DFP 2025-12-31 v1, recebido em 2026-04-09 | - |
| `BHIA3` | -5,11 | -10,93 | 5,82 p.p. | não | ITR 2026-03-31 v1, recebido em 2026-05-13 | -8,88 |
| `INTB3` | 10,75 | 5,13 | 5,62 p.p. | sim | ITR 2026-06-30 v1, recebido em 2026-07-29 | 10,64 |
| `TIMS3` | 3,43 | 7,56 | 4,13 p.p. | não | ITR 2026-06-30 v1, recebido em 2026-07-27 | 7,49 |
| `BRST3` | 2,86 | 0,85 | 2,01 p.p. | não | ITR 2026-03-31 v1, recebido em 2026-05-12 | 0,87 |
| `EMAE4` | -6,52 | -5,11 | 1,41 p.p. | sim | ITR 2026-06-30 v1, recebido em 2026-08-07 | -7,02 |
| `CAML3` | 1,81 | 0,92 | 0,89 p.p. | sim | ITR 2026-05-31 v1, recebido em 2026-07-14 | 1,38 |
| `HBSA3` | -3,20 | -2,80 | 0,40 p.p. | não | ITR 2026-03-31 v1, recebido em 2026-05-04 | -2,17 |
| `VIVA3` | 12,00 | 12,40 | 0,40 p.p. | não | ITR 2026-06-30 v1, recebido em 2026-08-05 | 12,52 |

Sem valor reconstruído: `AMBP3` (stale_filing), `EQPA3` (missing_input: LL).

</details>

<details>
<summary>ROA, 2026-11-05</summary>

| ticker | referência | reconstruído | diferença | si_lag | arquivo | valor anterior |
| --- | --- | --- | --- | --- | --- | --- |
| `EQPA3` | 96,34 | 7,85 | 88,49 p.p. | não | ITR 2026-06-30 v1, recebido em 2026-08-12 | 7,79 |
| `BHIA3` | -5,11 | -51,00 | 45,89 p.p. | não | ITR 2026-06-30 v1, recebido em 2026-08-16 | -10,93 |
| `VTRU3` | -24,02 | 16,51 | 40,53 p.p. | não | ITR 2026-06-30 v1, recebido em 2026-08-11 | 16,06 |
| `CASH3` | -18,87 | -0,40 | 18,47 p.p. | sim | ITR 2026-09-30 v1, recebido em 2026-11-04 | -18,87 |
| `EMAE4` | -6,52 | -17,51 | 10,99 p.p. | não | ITR 2026-09-30 v1, recebido em 2026-11-04 | -5,11 |
| `PCAR3` | -13,68 | -19,91 | 6,23 p.p. | sim | ITR 2026-09-30 v1, recebido em 2026-11-04 | -13,68 |
| `EALT4` | 8,97 | 3,93 | 5,04 p.p. | sim | ITR 2026-09-30 v1, recebido em 2026-11-05 | 8,97 |
| `AMER3` | 2,60 | -1,92 | 4,52 p.p. | não | ITR 2026-06-30 v1, recebido em 2026-08-12 | 3,43 |
| `TIMS3` | 3,43 | 7,56 | 4,13 p.p. | não | ITR 2026-06-30 v1, recebido em 2026-07-27 | 7,49 |
| `AGRO3` | -2,43 | 1,44 | 3,87 p.p. | sim | ITR 2026-09-30 v1, recebido em 2026-11-03 | -2,43 |

Sem valor reconstruído: `AALR3` (stale_filing), `AMBP3` (stale_filing).

</details>

<details>
<summary>MARG. LIQUIDA, 2026-08-07</summary>

| ticker | referência | reconstruído | diferença | si_lag | arquivo | valor anterior |
| --- | --- | --- | --- | --- | --- | --- |
| `VTRU3` | -14,81 | 48,63 | 63,44 p.p. | não | ITR 2026-03-31 v1, recebido em 2026-05-06 | 16,37 |
| `EMAE4` | -29,53 | -19,87 | 9,66 p.p. | não | ITR 2026-06-30 v1, recebido em 2026-08-07 | -26,20 |
| `TIMS3` | 7,22 | 15,79 | 8,57 p.p. | não | ITR 2026-06-30 v1, recebido em 2026-07-27 | 16,02 |
| `ONCO3` | -71,05 | -63,14 | 7,91 p.p. | não | DFP 2025-12-31 v1, recebido em 2026-04-09 | - |
| `INTB3` | 13,11 | 6,26 | 6,85 p.p. | não | ITR 2026-06-30 v1, recebido em 2026-07-29 | 12,36 |
| `BHIA3` | -5,98 | -12,30 | 6,32 p.p. | não | ITR 2026-03-31 v1, recebido em 2026-05-13 | -10,23 |
| `BRST3` | 8,46 | 2,27 | 6,19 p.p. | não | ITR 2026-03-31 v1, recebido em 2026-05-12 | 2,43 |
| `TPIS3` | -42,83 | -39,30 | 3,53 p.p. | não | ITR 2026-03-31 v1, recebido em 2026-05-15 | -36,60 |
| `MOTV3` | 20,63 | 19,27 | 1,36 p.p. | não | ITR 2026-06-30 v1, recebido em 2026-07-29 | 17,40 |
| `HBSA3` | -8,99 | -7,87 | 1,12 p.p. | não | ITR 2026-03-31 v1, recebido em 2026-05-04 | -6,29 |

Sem valor reconstruído: `AMBP3` (stale_filing), `EQPA3` (missing_input: LL).

</details>

<details>
<summary>MARG. LIQUIDA, 2026-11-05</summary>

| ticker | referência | reconstruído | diferença | si_lag | arquivo | valor anterior |
| --- | --- | --- | --- | --- | --- | --- |
| `EQPA3` | 150,85 | 12,29 | 138,56 p.p. | não | ITR 2026-06-30 v1, recebido em 2026-08-12 | 12,31 |
| `VTRU3` | -14,81 | 46,56 | 61,37 p.p. | não | ITR 2026-06-30 v1, recebido em 2026-08-11 | 48,63 |
| `EMAE4` | -29,53 | -70,02 | 40,49 p.p. | não | ITR 2026-09-30 v1, recebido em 2026-11-04 | -19,87 |
| `BHIA3` | -5,98 | -44,42 | 38,44 p.p. | não | ITR 2026-06-30 v1, recebido em 2026-08-16 | -12,30 |
| `CASH3` | -20,31 | -0,51 | 19,80 p.p. | sim | ITR 2026-09-30 v1, recebido em 2026-11-04 | -20,31 |
| `AGRO3` | -9,73 | 5,73 | 15,46 p.p. | sim | ITR 2026-09-30 v1, recebido em 2026-11-03 | -9,73 |
| `AXIA3` | 27,05 | 41,74 | 14,69 p.p. | sim | ITR 2026-09-30 v1, recebido em 2026-11-04 | 27,05 |
| `AXIA7` | 27,05 | 41,74 | 14,69 p.p. | sim | ITR 2026-09-30 v1, recebido em 2026-11-04 | 27,05 |
| `TIMS3` | 7,22 | 15,79 | 8,57 p.p. | não | ITR 2026-06-30 v1, recebido em 2026-07-27 | 16,02 |
| `ALUP11` | 27,68 | 19,73 | 7,95 p.p. | sim | ITR 2026-09-30 v1, recebido em 2026-11-05 | 27,68 |

Sem valor reconstruído: `AALR3` (stale_filing), `AMBP3` (stale_filing).

</details>

<details>
<summary>MARGEM EBIT, 2026-08-07</summary>

| ticker | referência | reconstruído | diferença | si_lag | arquivo | valor anterior |
| --- | --- | --- | --- | --- | --- | --- |
| `VTRU3` | -7,31 | 27,71 | 35,02 p.p. | não | ITR 2026-03-31 v1, recebido em 2026-05-06 | 26,03 |
| `EMAE4` | -85,26 | -65,00 | 20,26 p.p. | não | ITR 2026-06-30 v1, recebido em 2026-08-07 | -68,80 |
| `ONCO3` | -43,01 | -34,68 | 8,33 p.p. | não | DFP 2025-12-31 v1, recebido em 2026-04-09 | - |
| `TPIS3` | -52,90 | -45,84 | 7,06 p.p. | não | ITR 2026-03-31 v1, recebido em 2026-05-15 | -43,62 |
| `BHIA3` | -1,82 | 4,41 | 6,23 p.p. | não | ITR 2026-03-31 v1, recebido em 2026-05-13 | 4,60 |
| `ISAE3` | 48,44 | 46,88 | 1,56 p.p. | não | ITR 2026-06-30 v1, recebido em 2026-08-03 | 43,10 |
| `ISAE4` | 48,44 | 46,88 | 1,56 p.p. | não | ITR 2026-06-30 v1, recebido em 2026-08-03 | 43,10 |
| `BRST3` | 14,96 | 14,00 | 0,96 p.p. | não | ITR 2026-03-31 v1, recebido em 2026-05-12 | 14,43 |
| `VIVA3` | 23,33 | 22,40 | 0,93 p.p. | sim | ITR 2026-06-30 v1, recebido em 2026-08-05 | 23,54 |
| `EQTL3` | 15,18 | 15,53 | 0,35 p.p. | não | ITR 2026-03-31 v1, recebido em 2026-05-13 | 15,63 |

Sem valor reconstruído: `AMBP3` (stale_filing), `EQPA3` (missing_input: EBIT).

</details>

<details>
<summary>MARGEM EBIT, 2026-11-05</summary>

| ticker | referência | reconstruído | diferença | si_lag | arquivo | valor anterior |
| --- | --- | --- | --- | --- | --- | --- |
| `EQPA3` | 157,82 | 19,27 | 138,55 p.p. | não | ITR 2026-06-30 v1, recebido em 2026-08-12 | 18,62 |
| `VTRU3` | -7,31 | 28,25 | 35,56 p.p. | não | ITR 2026-06-30 v1, recebido em 2026-08-11 | 27,71 |
| `EMAE4` | -85,26 | -119,58 | 34,32 p.p. | não | ITR 2026-09-30 v1, recebido em 2026-11-04 | -65,00 |
| `CASH3` | -13,91 | -1,72 | 12,19 p.p. | sim | ITR 2026-09-30 v1, recebido em 2026-11-04 | -13,91 |
| `ALUP11` | 68,45 | 56,78 | 11,67 p.p. | sim | ITR 2026-09-30 v1, recebido em 2026-11-05 | 68,45 |
| `ALUP3` | 68,45 | 56,78 | 11,67 p.p. | sim | ITR 2026-09-30 v1, recebido em 2026-11-05 | 68,45 |
| `ALUP4` | 68,45 | 56,78 | 11,67 p.p. | sim | ITR 2026-09-30 v1, recebido em 2026-11-05 | 68,45 |
| `AXIA3` | 25,81 | 37,30 | 11,49 p.p. | sim | ITR 2026-09-30 v1, recebido em 2026-11-04 | 25,81 |
| `AXIA7` | 25,81 | 37,30 | 11,49 p.p. | sim | ITR 2026-09-30 v1, recebido em 2026-11-04 | 25,81 |
| `PRIO3` | 27,60 | 36,29 | 8,69 p.p. | sim | ITR 2026-09-30 v1, recebido em 2026-11-04 | 27,60 |

Sem valor reconstruído: `AALR3` (stale_filing), `AMBP3` (stale_filing).

</details>

<details>
<summary>P/L, 2026-08-07</summary>

| ticker | referência | reconstruído | diferença | si_lag | arquivo | valor anterior |
| --- | --- | --- | --- | --- | --- | --- |
| `VVEO3` | 10,25 | 102,46 | 899,61% | não | ITR 2026-03-31 v1, recebido em 2026-05-14 | 113,28 |
| `BRST3` | 10,58 | 31,04 | 193,38% | não | ITR 2026-03-31 v1, recebido em 2026-05-12 | 30,03 |
| `AZEV4` | -0,32 | -0,86 | 168,75% | não | ITR 2026-03-31 v2, recebido em 2026-05-15 | -0,84 |
| `AZEV3` | -0,68 | -1,82 | 167,65% | não | ITR 2026-03-31 v2, recebido em 2026-05-15 | -1,77 |
| `LIGT3` | 1,91 | 4,42 | 131,41% | não | ITR 2026-03-31 v1, recebido em 2026-05-14 | 54,20 |
| `VTRU3` | -6,32 | 1,53 | 124,21% | não | ITR 2026-03-31 v1, recebido em 2026-05-06 | 4,61 |
| `INTB3` | 8,08 | 16,92 | 109,41% | sim | ITR 2026-06-30 v1, recebido em 2026-07-29 | 8,39 |
| `CAML3` | 8,25 | 14,57 | 76,61% | não | ITR 2026-05-31 v1, recebido em 2026-07-14 | 10,83 |
| `BHIA3` | -0,46 | -0,19 | 58,70% | não | ITR 2026-03-31 v1, recebido em 2026-05-13 | -0,16 |
| `TIMS3` | 22,67 | 10,37 | 54,26% | não | ITR 2026-06-30 v1, recebido em 2026-07-27 | 10,36 |

Sem valor reconstruído: `AMBP3` (stale_filing), `EQPA3` (missing_input: LL).

</details>

<details>
<summary>P/L, 2026-11-05</summary>

| ticker | referência | reconstruído | diferença | si_lag | arquivo | valor anterior |
| --- | --- | --- | --- | --- | --- | --- |
| `CASH3` | -4,09 | -173,75 | 4148,17% | não | ITR 2026-09-30 v1, recebido em 2026-11-04 | -4,43 |
| `EQPA3` | 0,57 | 6,97 | 1122,81% | não | ITR 2026-06-30 v1, recebido em 2026-08-12 | 7,08 |
| `SIMH3` | -17,22 | 31,70 | 284,09% | sim | ITR 2026-09-30 v1, recebido em 2026-11-04 | -17,22 |
| `AGRO3` | -20,56 | 33,31 | 262,01% | sim | ITR 2026-09-30 v1, recebido em 2026-11-03 | -20,56 |
| `AMER3` | 2,22 | -3,00 | 235,14% | não | ITR 2026-06-30 v1, recebido em 2026-08-12 | 1,52 |
| `VVEO3` | 15,38 | 38,60 | 150,98% | não | ITR 2026-06-30 v1, recebido em 2026-08-11 | 108,86 |
| `LIGT3` | 1,65 | 3,83 | 132,12% | não | ITR 2026-06-30 v1, recebido em 2026-08-13 | 3,53 |
| `VTRU3` | -6,09 | 1,68 | 127,59% | não | ITR 2026-06-30 v1, recebido em 2026-08-11 | 1,47 |
| `MYPK3` | 15,83 | 35,92 | 126,91% | sim | ITR 2026-09-30 v1, recebido em 2026-11-04 | 15,83 |
| `CAML3` | 9,20 | 19,89 | 116,20% | não | ITR 2026-08-31 v1, recebido em 2026-10-08 | 13,11 |

Sem valor reconstruído: `AALR3` (stale_filing), `AMBP3` (stale_filing).

</details>

<details>
<summary>P/VP, 2026-08-07</summary>

| ticker | referência | reconstruído | diferença | si_lag | arquivo | valor anterior |
| --- | --- | --- | --- | --- | --- | --- |
| `VVEO3` | 0,11 | 1,10 | 900,00% | não | ITR 2026-03-31 v1, recebido em 2026-05-14 | 1,07 |
| `AZEV3` | 3,65 | 9,73 | 166,58% | não | ITR 2026-03-31 v2, recebido em 2026-05-15 | 9,82 |
| `AZEV4` | 1,73 | 4,60 | 165,90% | não | ITR 2026-03-31 v2, recebido em 2026-05-15 | 4,64 |
| `LIGT3` | 0,60 | 1,39 | 131,67% | não | ITR 2026-03-31 v1, recebido em 2026-05-14 | 2,12 |
| `VTRU3` | 1,87 | 0,46 | 75,40% | não | ITR 2026-03-31 v1, recebido em 2026-05-06 | 0,58 |
| `BHIA3` | 0,23 | 0,38 | 65,22% | não | ITR 2026-03-31 v1, recebido em 2026-05-13 | 0,17 |
| `TIMS3` | 3,61 | 1,79 | 50,42% | não | ITR 2026-06-30 v1, recebido em 2026-07-27 | 1,84 |
| `ONCO3` | 1,32 | 0,78 | 40,91% | não | DFP 2025-12-31 v1, recebido em 2026-04-09 | - |
| `ORVR3` | 6,38 | 4,47 | 29,94% | não | ITR 2026-03-31 v1, recebido em 2026-05-14 | 4,52 |
| `SIMH3` | 0,97 | 0,71 | 26,80% | não | ITR 2026-03-31 v1, recebido em 2026-05-07 | 0,68 |

Sem valor reconstruído: `AMBP3` (stale_filing).

</details>

<details>
<summary>P/VP, 2026-11-05</summary>

| ticker | referência | reconstruído | diferença | si_lag | arquivo | valor anterior |
| --- | --- | --- | --- | --- | --- | --- |
| `PCAR3` | 2,88 | -3,67 | 227,43% | sim | ITR 2026-09-30 v1, recebido em 2026-11-04 | 2,88 |
| `VVEO3` | 0,48 | 1,20 | 150,00% | não | ITR 2026-06-30 v1, recebido em 2026-08-11 | 1,17 |
| `BHIA3` | 0,22 | -0,08 | 136,36% | não | ITR 2026-06-30 v1, recebido em 2026-08-16 | 0,37 |
| `LIGT3` | 0,43 | 0,99 | 130,23% | não | ITR 2026-06-30 v1, recebido em 2026-08-13 | 1,11 |
| `VTRU3` | 1,80 | 0,46 | 74,44% | não | ITR 2026-06-30 v1, recebido em 2026-08-11 | 0,44 |
| `RAIZ4` | -0,38 | -0,12 | 68,42% | não | ITR 2026-06-30 v1, recebido em 2026-08-13 | -0,36 |
| `VIVA3` | 1,35 | 0,45 | 66,67% | sim | ITR 2026-09-30 v1, recebido em 2026-11-05 | 1,35 |
| `OPCT3` | 4,16 | 1,75 | 57,93% | não | ITR 2026-06-30 v1, recebido em 2026-08-06 | 1,81 |
| `TIMS3` | 3,82 | 1,91 | 50,00% | não | ITR 2026-06-30 v1, recebido em 2026-07-27 | 1,96 |
| `EALT4` | 0,68 | 0,41 | 39,71% | sim | ITR 2026-09-30 v1, recebido em 2026-11-05 | 0,66 |

Sem valor reconstruído: `AALR3` (stale_filing), `AMBP3` (stale_filing).

</details>

<details>
<summary>EV/EBIT, 2026-08-07</summary>

| ticker | referência | reconstruído | diferença | si_lag | arquivo | valor anterior |
| --- | --- | --- | --- | --- | --- | --- |
| `TFCO4` | 10,90 | 64,31 | 490,00% | não | ITR 2026-03-31 v1, recebido em 2026-05-11 | 65,47 |
| `VTRU3` | -9,11 | 4,99 | 154,77% | não | ITR 2026-03-31 v1, recebido em 2026-05-06 | 5,63 |
| `BHIA3` | -15,89 | 5,00 | 131,47% | não | ITR 2026-03-31 v1, recebido em 2026-05-13 | 3,89 |
| `AZEV3` | -1,26 | -2,46 | 95,24% | não | ITR 2026-03-31 v2, recebido em 2026-05-15 | -2,40 |
| `AZEV4` | -1,26 | -2,46 | 95,24% | não | ITR 2026-03-31 v2, recebido em 2026-05-15 | -2,40 |
| `SHUL4` | 23,73 | 6,81 | 71,30% | não | ITR 2026-03-31 v1, recebido em 2026-05-14 | 5,93 |
| `VVEO3` | 5,40 | 8,65 | 60,19% | não | ITR 2026-03-31 v1, recebido em 2026-05-14 | 9,12 |
| `LIGT3` | 16,63 | 24,76 | 48,89% | não | ITR 2026-03-31 v1, recebido em 2026-05-14 | 15,38 |
| `CAML3` | 7,50 | 10,12 | 34,93% | sim | ITR 2026-05-31 v1, recebido em 2026-07-14 | 7,71 |
| `ORVR3` | 31,80 | 23,50 | 26,10% | não | ITR 2026-03-31 v1, recebido em 2026-05-14 | 25,14 |

Sem valor reconstruído: `AMBP3` (stale_filing), `EQPA3` (missing_input: EBIT).

</details>

<details>
<summary>EV/EBIT, 2026-11-05</summary>

| ticker | referência | reconstruído | diferença | si_lag | arquivo | valor anterior |
| --- | --- | --- | --- | --- | --- | --- |
| `CASH3` | -4,33 | -37,88 | 774,83% | não | ITR 2026-09-30 v1, recebido em 2026-11-04 | -4,83 |
| `EQPA3` | 0,86 | 7,06 | 720,93% | não | ITR 2026-06-30 v1, recebido em 2026-08-12 | 7,20 |
| `TFCO4` | 7,40 | 42,72 | 477,30% | não | ITR 2026-06-30 v1, recebido em 2026-08-12 | 44,40 |
| `AGRO3` | -58,52 | 138,78 | 337,15% | sim | ITR 2026-09-30 v1, recebido em 2026-11-03 | -58,52 |
| `VTRU3` | -8,65 | 4,52 | 152,25% | não | ITR 2026-06-30 v1, recebido em 2026-08-11 | 4,90 |
| `BHIA3` | -15,82 | -2,85 | 81,98% | não | ITR 2026-06-30 v1, recebido em 2026-08-16 | 4,98 |
| `TUPY3` | -10,99 | -18,34 | 66,88% | sim | ITR 2026-09-30 v1, recebido em 2026-11-05 | -10,99 |
| `VIVA3` | 5,97 | 2,52 | 57,79% | sim | ITR 2026-09-30 v1, recebido em 2026-11-05 | 5,85 |
| `LIGT3` | 17,61 | 25,79 | 46,45% | não | ITR 2026-06-30 v1, recebido em 2026-08-13 | 21,90 |
| `OPCT3` | 14,12 | 8,18 | 42,07% | não | ITR 2026-06-30 v1, recebido em 2026-08-06 | 9,69 |

Sem valor reconstruído: `AALR3` (stale_filing), `AMBP3` (stale_filing).

</details>

<details>
<summary>PSR, 2026-08-07</summary>

| ticker | referência | reconstruído | diferença | si_lag | arquivo | valor anterior |
| --- | --- | --- | --- | --- | --- | --- |
| `VVEO3` | 0,02 | 0,18 | 800,00% | não | ITR 2026-03-31 v1, recebido em 2026-05-14 | 0,18 |
| `AZEV3` | 0,92 | 2,46 | 167,39% | não | ITR 2026-03-31 v2, recebido em 2026-05-15 | 3,14 |
| `AZEV4` | 0,44 | 1,16 | 163,64% | não | ITR 2026-03-31 v2, recebido em 2026-05-15 | 1,48 |
| `LIGT3` | 0,33 | 0,75 | 127,27% | não | ITR 2026-03-31 v1, recebido em 2026-05-14 | 0,77 |
| `BHIA3` | 0,03 | 0,02 | 33,33% | não | ITR 2026-03-31 v1, recebido em 2026-05-13 | 0,02 |
| `ORVR3` | 8,17 | 5,72 | 29,99% | não | ITR 2026-03-31 v1, recebido em 2026-05-14 | 6,21 |
| `SIMH3` | 0,09 | 0,07 | 22,22% | não | ITR 2026-03-31 v1, recebido em 2026-05-07 | 0,07 |
| `VTRU3` | 0,94 | 0,74 | 21,28% | não | ITR 2026-03-31 v1, recebido em 2026-05-06 | 0,75 |
| `BRST3` | 0,89 | 0,71 | 20,22% | não | ITR 2026-03-31 v1, recebido em 2026-05-12 | 0,73 |
| `EGIE3` | 3,08 | 2,48 | 19,48% | não | ITR 2026-06-30 v1, recebido em 2026-08-05 | 2,56 |

Sem valor reconstruído: `AMBP3` (stale_filing), `EQPA3` (missing_input: REC).

</details>

<details>
<summary>PSR, 2026-11-05</summary>

| ticker | referência | reconstruído | diferença | si_lag | arquivo | valor anterior |
| --- | --- | --- | --- | --- | --- | --- |
| `VVEO3` | 0,07 | 0,19 | 171,43% | não | ITR 2026-06-30 v1, recebido em 2026-08-11 | 0,19 |
| `LIGT3` | 0,26 | 0,59 | 126,92% | não | ITR 2026-06-30 v1, recebido em 2026-08-13 | 0,60 |
| `VIVA3` | 1,36 | 0,47 | 65,44% | não | ITR 2026-09-30 v1, recebido em 2026-11-05 | 1,28 |
| `OPCT3` | 1,80 | 0,76 | 57,78% | não | ITR 2026-06-30 v1, recebido em 2026-08-06 | 0,84 |
| `EALT4` | 0,47 | 0,28 | 40,43% | sim | ITR 2026-09-30 v1, recebido em 2026-11-05 | 0,46 |
| `BHIA3` | 0,03 | 0,02 | 33,33% | não | ITR 2026-06-30 v1, recebido em 2026-08-16 | 0,02 |
| `PMAM3` | 1,84 | 1,29 | 29,89% | não | ITR 2026-06-30 v1, recebido em 2026-08-31 | 0,64 |
| `BRST3` | 0,75 | 0,57 | 24,00% | não | ITR 2026-06-30 v1, recebido em 2026-08-11 | 0,59 |
| `EMAE4` | 3,75 | 4,52 | 20,53% | não | ITR 2026-09-30 v1, recebido em 2026-11-04 | 3,22 |
| `EGIE3` | 2,76 | 2,22 | 19,57% | não | ITR 2026-06-30 v1, recebido em 2026-08-05 | 2,30 |

Sem valor reconstruído: `AALR3` (stale_filing), `AMBP3` (stale_filing).

</details>

<details>
<summary>CAGR RECEITAS 5 ANOS, 2026-08-07</summary>

| ticker | referência | reconstruído | diferença | si_lag | arquivo | valor anterior |
| --- | --- | --- | --- | --- | --- | --- |
| `MLAS3` | 6,65 | 4,98 | 1,67 p.p. | não | ITR 2026-03-31 v1, recebido em 2026-05-13 | 4,98 |
| `EQPA3` | 15,78 | 14,19 | 1,59 p.p. | não | ITR 2026-03-31 v1, recebido em 2026-05-13 | - |
| `BHIA3` | 1,39 | 0,20 | 1,19 p.p. | não | ITR 2026-03-31 v1, recebido em 2026-05-13 | 0,20 |
| `VALE3` | 0,72 | 0,48 | 0,24 p.p. | não | ITR 2026-06-30 v3, recebido em 2026-07-31 | 0,48 |
| `AALR3` | 4,43 | 4,43 | 0,00 p.p. | não | DFP 2025-12-31 v1, recebido em 2026-07-30 | - |
| `ABEV3` | 8,61 | 8,61 | 0,00 p.p. | não | ITR 2026-06-30 v1, recebido em 2026-07-30 | 8,61 |
| `AGRO3` | 10,70 | 10,70 | 0,00 p.p. | não | ITR 2026-03-31 v1, recebido em 2026-05-07 | 10,70 |
| `ALLD3` | 3,11 | 3,11 | 0,00 p.p. | não | ITR 2026-03-31 v1, recebido em 2026-05-07 | 3,11 |
| `ALOS3` | 29,09 | 29,09 | 0,00 p.p. | não | ITR 2026-06-30 v1, recebido em 2026-08-06 | 29,09 |
| `ALPA3` | 6,29 | 6,29 | 0,00 p.p. | não | ITR 2026-06-30 v1, recebido em 2026-08-06 | 6,29 |

Sem valor reconstruído: `AMBP3` (stale_filing), `AURE3` (short_history), `BRST3` (short_history), `LAND3` (short_history).

</details>

<details>
<summary>CAGR RECEITAS 5 ANOS, 2026-11-05</summary>

| ticker | referência | reconstruído | diferença | si_lag | arquivo | valor anterior |
| --- | --- | --- | --- | --- | --- | --- |
| `MLAS3` | 6,65 | 4,98 | 1,67 p.p. | não | ITR 2026-06-30 v1, recebido em 2026-08-12 | 4,98 |
| `BHIA3` | 1,39 | 0,20 | 1,19 p.p. | não | ITR 2026-06-30 v1, recebido em 2026-08-16 | 0,20 |
| `VALE3` | 0,72 | 0,48 | 0,24 p.p. | não | ITR 2026-09-30 v2, recebido em 2026-10-29 | 0,48 |
| `ABEV3` | 8,61 | 8,61 | 0,00 p.p. | não | ITR 2026-09-30 v1, recebido em 2026-10-29 | 8,61 |
| `AGRO3` | -5,40 | -5,40 | 0,00 p.p. | não | ITR 2026-09-30 v1, recebido em 2026-11-03 | -5,40 |
| `ALLD3` | 3,11 | 3,11 | 0,00 p.p. | não | ITR 2026-06-30 v1, recebido em 2026-08-12 | 3,11 |
| `ALOS3` | 29,09 | 29,09 | 0,00 p.p. | não | ITR 2026-09-30 v1, recebido em 2026-11-05 | 29,09 |
| `ALPA3` | 6,29 | 6,29 | 0,00 p.p. | não | ITR 2026-06-30 v1, recebido em 2026-08-06 | 6,29 |
| `ALPA4` | 6,29 | 6,29 | 0,00 p.p. | não | ITR 2026-06-30 v1, recebido em 2026-08-06 | 6,29 |
| `ALPK3` | 23,59 | 23,59 | 0,00 p.p. | não | ITR 2026-09-30 v1, recebido em 2026-11-04 | 23,59 |

Sem valor reconstruído: `AALR3` (stale_filing), `AMBP3` (stale_filing), `AURE3` (short_history), `BRST3` (short_history), `LAND3` (short_history).

</details>

<details>
<summary>CAGR LUCROS 5 ANOS, 2026-08-07</summary>

| ticker | referência | reconstruído | diferença | si_lag | arquivo | valor anterior |
| --- | --- | --- | --- | --- | --- | --- |
| `TIMS3` | 1,43 | 18,60 | 17,17 p.p. | não | ITR 2026-06-30 v1, recebido em 2026-07-27 | 18,63 |
| `INTB3` | 12,71 | -2,77 | 15,48 p.p. | não | ITR 2026-06-30 v1, recebido em 2026-07-29 | 11,88 |
| `CAML3` | -15,89 | -24,93 | 9,04 p.p. | não | ITR 2026-05-31 v1, recebido em 2026-07-14 | -20,34 |
| `VIVA3` | 31,91 | 32,77 | 0,86 p.p. | sim | ITR 2026-06-30 v1, recebido em 2026-08-05 | 31,89 |
| `GMAT3` | 18,99 | 19,56 | 0,57 p.p. | não | ITR 2026-03-31 v1, recebido em 2026-05-14 | 20,36 |
| `SOJA3` | -18,88 | -18,62 | 0,26 p.p. | não | ITR 2026-03-31 v1, recebido em 2026-05-13 | -22,21 |
| `MOTV3` | 82,59 | 82,60 | 0,01 p.p. | não | ITR 2026-06-30 v1, recebido em 2026-07-29 | 77,45 |
| `ABEV3` | 7,38 | 7,38 | 0,00 p.p. | não | ITR 2026-06-30 v1, recebido em 2026-07-30 | 6,48 |
| `ALLD3` | 15,35 | 15,35 | 0,00 p.p. | não | ITR 2026-03-31 v1, recebido em 2026-05-07 | 14,69 |
| `ALOS3` | 40,77 | 40,77 | 0,00 p.p. | não | ITR 2026-06-30 v1, recebido em 2026-08-06 | 37,40 |

Sem valor reconstruído: `BRST3` (short_history), `DXCO3` (non_positive_base), `EQPA3` (missing_input: LL_TTM), `LAND3` (short_history).

</details>

<details>
<summary>CAGR LUCROS 5 ANOS, 2026-11-05</summary>

| ticker | referência | reconstruído | diferença | si_lag | arquivo | valor anterior |
| --- | --- | --- | --- | --- | --- | --- |
| `EQPA3` | 93,04 | 16,91 | 76,13 p.p. | não | ITR 2026-06-30 v1, recebido em 2026-08-12 | 16,53 |
| `EALT4` | 18,53 | 1,27 | 17,26 p.p. | sim | ITR 2026-09-30 v1, recebido em 2026-11-05 | 18,53 |
| `TIMS3` | 1,43 | 18,60 | 17,17 p.p. | não | ITR 2026-06-30 v1, recebido em 2026-07-27 | 18,63 |
| `PRIO3` | 53,58 | 68,53 | 14,95 p.p. | sim | ITR 2026-09-30 v1, recebido em 2026-11-04 | 53,58 |
| `CAML3` | -19,43 | -30,94 | 11,51 p.p. | não | ITR 2026-08-31 v1, recebido em 2026-10-08 | -24,93 |
| `AXIA3` | 13,72 | 24,05 | 10,33 p.p. | sim | ITR 2026-09-30 v1, recebido em 2026-11-04 | 13,72 |
| `AXIA7` | 13,72 | 24,05 | 10,33 p.p. | sim | ITR 2026-09-30 v1, recebido em 2026-11-04 | 13,72 |
| `VBBR3` | 5,30 | 12,72 | 7,42 p.p. | sim | ITR 2026-09-30 v1, recebido em 2026-11-04 | 5,30 |
| `ALUP11` | 8,04 | 1,81 | 6,23 p.p. | sim | ITR 2026-09-30 v1, recebido em 2026-11-05 | 8,04 |
| `ALUP3` | 8,04 | 1,81 | 6,23 p.p. | sim | ITR 2026-09-30 v1, recebido em 2026-11-05 | 8,04 |

Sem valor reconstruído: `BRST3` (short_history), `DXCO3` (non_positive_base), `SCAR3` (non_positive_base).

</details>

<details>
<summary>DY, 2026-08-07</summary>

| ticker | referência | reconstruído | diferença | si_lag | arquivo | valor anterior |
| --- | --- | --- | --- | --- | --- | --- |
| `HBRE3` | 52,50 | 21,87 | 30,63 p.p. | não | - | - |
| `RIAA3` | 31,29 | 41,94 | 10,65 p.p. | não | - | - |
| `MDNE3` | 14,37 | 7,24 | 7,13 p.p. | não | - | - |
| `VLID3` | 9,90 | 3,27 | 6,63 p.p. | não | - | - |
| `EQPA3` | 2,53 | 8,49 | 5,96 p.p. | não | - | - |
| `HBOR3` | 1,05 | 6,94 | 5,89 p.p. | não | - | - |
| `PFRM3` | 4,63 | 9,14 | 4,51 p.p. | não | - | - |
| `COCE5` | 0,00 | 4,36 | 4,36 p.p. | não | - | - |
| `CLSC4` | 0,36 | 4,48 | 4,12 p.p. | não | - | - |
| `TGMA3` | 10,84 | 14,87 | 4,03 p.p. | não | - | - |

</details>

<details>
<summary>DY, 2026-11-05</summary>

| ticker | referência | reconstruído | diferença | si_lag | arquivo | valor anterior |
| --- | --- | --- | --- | --- | --- | --- |
| `MBRF3` | 18,41 | 0,00 | 18,41 p.p. | não | - | - |
| `JSLG3` | 48,30 | 34,82 | 13,48 p.p. | não | - | - |
| `RIAA3` | 40,52 | 53,98 | 13,46 p.p. | não | - | - |
| `MDNE3` | 22,31 | 11,24 | 11,07 p.p. | não | - | - |
| `VLID3` | 13,48 | 5,32 | 8,16 p.p. | não | - | - |
| `EQPA3` | 4,65 | 11,05 | 6,40 p.p. | não | - | - |
| `COCE5` | 0,00 | 5,30 | 5,30 p.p. | não | - | - |
| `PFRM3` | 2,42 | 7,13 | 4,71 p.p. | não | - | - |
| `PASS3` | 2,90 | 6,48 | 3,58 p.p. | não | - | - |
| `EQTL3` | 1,36 | 4,77 | 3,41 p.p. | não | - | - |

</details>

## Sensibilidades (não são o veredito oficial)

Estas linhas não são o veredito oficial. O veredito oficial é o da seção Veredito, e nenhuma sensibilidade entra nele.

| sensibilidade | veredito da classe | veredito do grupo financeiro | o que muda |
| --- | --- | --- | --- |
| Sem referências obsoletas (D-A) | `não reconstruível` | `não reconstruível` | excluídos: 2026-08-07 `AESB3`, `APER3`, `AXIA6`, `AZUL53`, `AZUL54`, `BPAN4`, `BRFS3`, `CLSA3`, `CPLE5`, `CPLE6`, `CRFB3`, `ELMD3`, `ENAT3`, `GPIV33`, `JBSS3`, `NEOE3`, `NTCO3`, `PETZ3`, `PORT3`, `SOMA3`, `STBP3`, `SULA11`, `ZAMP3`; 2026-11-05 `AESB3`, `AXIA6`, `AZUL54`, `BPAN4`, `BRFS3`, `CPLE5`, `CPLE6`, `ENAT3`, `GPIV33`, `JBSS3`, `NEOE3`, `NTCO3`, `PETZ3`, `PORT3`, `STBP3`, `SULA11`, `ZAMP3` |
| Data 2026-11-05 com filings até 2026-11-02 (D-B) | `não reconstruível` | `não reconstruível` | filings de 2026-11-05 com DT_RECEB ≤ 2026-11-02; as outras datas ficam como no veredito oficial |
| Sem ROIC e PEG RATIO (D-C) | `não reconstruível` | - | fora da classe: `PEG RATIO`, `ROIC` |

Nota (D-C): ROIC e PEG RATIO falham porque as fórmulas do Status Invest para eles não são públicas (A-8 é uma leitura), não por falta de dados.

## Investigações

- `MBRF3`, `DY` em 2026-11-05: A busca do histórico não é a causa: em 2026-08-07 o mesmo histórico dá DY 17,49, igual ao da Status Invest. O último provento que a B3 publica para a companhia, de R$ 2,8105 com data-ex 2025-09-19, está a 412 dias de 2026-11-05 e sai da janela (2025-11-05, 2026-11-05]; por isso o DY é 0,00. A Status Invest mantém 18,41 nessa data, valor que nenhuma fonte em cache reproduz. Evidência: b3_company_MBRF.json (tradingName MARFRIG, totalNumberShares 1380439543, retrieved_at 2026-11-10T00:37:01Z, dia de São Paulo 2026-11-09) e b3_cash_MARFRIG_p1.json (totalRecords 13, completo; o registro com lastDatePriorEx 18/09/2025 e valueCash 2,8105011104 é o único com data-ex depois de 2025-08-07, e nenhum tem data-ex em (2025-11-05, 2026-11-05]). Com esse registro: 100 × 2,8105011104 / 16,07 = 17,49 em 2026-08-07 (janela (2025-08-07, 2026-08-07]); 100 × 2,8105011104 / 15,63 = 17,98 em 2026-11-05 (CSV de 2026-11-06: PRECO 15,63, DY 18,41).

## Parâmetros

- Universo elegível: `VALOR DE MERCADO` ≥ 200000000 e `LIQUIDEZ MEDIA DIARIA` ≥ 50000. Fontes: `py/core/preprocessing.py:103-106`, `api/app/services/premium_policy.py:56`.
- Tolerância: 5% relativa; 0,50 p.p. para os indicadores em percentual (`CAGR LUCROS 5 ANOS`, `CAGR RECEITAS 5 ANOS`, `DY`, `MARG. LIQUIDA`, `MARGEM EBIT`, `ROA`, `ROE`, `ROIC`). A comparação usa o valor arredondado a 2 casas.
- Veredito por indicador: `reconstruível` com 90% ou mais dentro da tolerância; `com ressalvas` com 70% ou mais; `não reconstruível` abaixo disso.
- Liquidez: média de VOLTOT das últimas 21 sessões até d.
- Cotações (`quote_codbdi`, CODBDI): `02`, `07`, `08`.
- Cortes de filings (`filings_cutoffs`): `2026-11-05` com filings recebidos até `2026-11-02`.
- Escala das ações: A CVM não informa escala em `composicao_capital`. Cada contagem de classe em `d` é N = arquivado × f × eventos da B3 da classe com data-ex em (DT_REFER, d], com cada evento entrando pelo seu múltiplo (AD-033). Para cada arquivamento, o fator f ∈ {10^-2, 10^-1, 1, 10, 10^2, 10^3} é a potência de dez que deixa o arquivado vezes os eventos com data-ex em (DT_REFER, R] mais perto da `totalNumberShares` da B3, medido pela distância em log10; R é a data em que o registro da B3 foi obtido. Os eventos com data-ex depois de d só escolhem a potência de dez. A contagem da B3 só escolhe o fator e nunca entra num valor. Sem registro da B3, ou com contagem zero, `f` é 1 e a empresa aparece em `share_scale.unverified`. As contagens escalonadas são arredondadas. Limite (√10), só para mudança sem evento da B3: se a contagem mudou menos de √10 (3,16 vezes) entre o arquivamento e a obtenção do registro e nenhum evento a explica, ela fica como arquivada; se mudou √10 ou mais, `f` absorve a mudança como a potência de dez mais próxima. Isso é exato só para grupamento de 10:1; sem evento da B3, um desdobramento de 1:5 resulta em ×10, errado por 2 vezes. O resíduo (`residual`), arquivado × eventos até R × f / `totalNumberShares`, de cada empresa mostra cada caso.

### Fórmulas

| indicador | fórmula |
| --- | --- |
| `LIQ. CORRENTE` | AC / PC |
| `DIVIDA LIQUIDA / EBIT` | DL / EBIT |
| `DIV. LIQ. / PATRI.` | DL / PL_ctrl (A-21) |
| `ROE` | 100 × LL / abs(PL_ctrl), PL_ctrl = 2.03 − 2.03.09 (A-21) |
| `ROA` | 100 × LL / AT |
| `ROIC` | 100 × (EBIT − abs(IR)) / (PL + DB) |
| `MARG. LIQUIDA` | 100 × LL / REC |
| `MARGEM EBIT` | 100 × EBIT / REC |
| `P/L` | p_eq / (LL / N_total), p_eq = p / k, N_total = N_ON + N_PREF |
| `P/VP` | p_eq / (PL_ctrl / N_total), PL_ctrl = 2.03 − 2.03.09 (A-21) |
| `EV/EBIT` | (VALOR DE MERCADO + DL) / EBIT |
| `PSR` | p_eq / (REC / N_total) |
| `CAGR RECEITAS 5 ANOS` | 100 × ((REC_FY / REC_FY−5)^(1/5) − 1), exercícios fiscais da DFP; vazio se algum <= 0 |
| `CAGR LUCROS 5 ANOS` | 100 × ((LL_TTM / LL_FY−5)^(1/5) − 1), LL_FY−5 = exercício fiscal cinco anos antes da DFP usável mais recente; vazio se algum <= 0 (A-21) |
| `PEG RATIO` | P/L / (100 × (LL_TTM / LL_TTM,1 ano antes − 1)); vazio se LL_TTM,1 ano antes <= 0 |
| `DY` | 100 × DPS_12m / p, DPS_12m = soma dos dividendos com d − 365 dias < data-ex <= d (por unidade para units) |
| `PRECO` | p: último pregão em COTAHIST com data <= d |
| `VALOR DE MERCADO` | N_ON × p_ON + N_PREF × p_PREF (A-17): p_ON do ticker ON; p_PREF do preferencial mais líquido; classe com N > 0 e sem cotação usa a outra (vm_price_fallback); N = arquivado × f (A-18) × eventos B3 da classe com data-ex em (DT_REFER, d] (AD-033) |
| `LIQUIDEZ MEDIA DIARIA` | média de VOLTOT das últimas 21 sessões <= d, CODBDI 02, 07 e 08 (A-7) |

### Entradas que não são ponto no tempo

Estas entradas são lidas como foram obtidas, e não como estavam em cada data. O papel de cada uma limita o que ela pode alterar.

| entrada | papel | uso | verificação na reconstrução completa |
| --- | --- | --- | --- |
| B3 `totalNumberShares` | `unit_only` | escolhe a escala das contagens (A-18), sobre a contagem ajustada pelos eventos até a obtenção do registro da B3 (AD-033); nunca entra num valor | escala datada por arquivamento, com a contagem da B3 de cada data |
| Cadastro CNPJ → CD_CVM | `identity` | identifica a empresa (A-9) | empresas canceladas no cadastro |
| FCA `valor_mobiliario` (ticker → CNPJ, composição da unidade) | `identity` | mapeia ticker e unidade (A-9, MAP-03); a linha vem da maior Data_Referencia até d | linhas escolhidas pela data de entrega, não pela data de referência |
| B3 `codeCVM` da empresa | `identity` | fallback de mapeamento para tickers fora da FCA (A-9) | empresas sem registro atual na B3 |
| Cadastro `SETOR_ATIV` | `classification` | decide o grupo financeiro e a classe (A-10) | setor datado; reclassificações ao longo dos anos |
| B3 `tradingName` | `source_selection` | escolhe o histórico de dividendos buscado (A-15) | empresas renomeadas e canceladas sem registro atual |
| Valores de dividendos (B3 `valueCash`, Yahoo) | `dated_value_current_amount` | a data-ex é PIT (A-16); o valor é o atual | valores como eram conhecidos em cada data de decisão |
| B3 `stockDividends` | `dated_event_current_record` | ajusta valores de dividendos por desdobramento, bonificação e grupamento com data-ex depois do dividendo e ≤ d (A-20) e contagens de ações (AD-033): eventos com data-ex em (DT_REFER, d] entram na contagem; eventos com data-ex depois de d e até a obtenção do registro da B3 só escolhem a potência de dez | eventos societários como eram conhecidos em cada data |

## Cobertura do mapeamento

### 2026-08-07

- Elegíveis: 269. Mapeados: 246. Não mapeados: 23.
- Mapeados pela B3 (`mapped_via_b3`): `ALUP3`, `ALUP4`, `AMAR3`, `BPAC11`, `BPAC3`, `BPAC5`, `CMIN3`, `CSNA3`, `EQPA3`, `KLBN3`, `KLBN4`, `MBRF3`, `TGMA3`.
- Caíram para o período anterior porque o mais recente foi reapresentado depois de d, com uma versão já recebida até d (`refiled_after_d`): `ONCO3`.
- Usaram o preço da outra classe (`vm_price_fallback`): 7 empresa(s).
- Sem cotação recente (`reference_stale`, sem registro em [d − 30 dias, d]): `AESB3`, `APER3`, `AXIA6`, `AZUL53`, `AZUL54`, `BPAN4`, `BRFS3`, `CLSA3`, `CPLE5`, `CPLE6`, `CRFB3`, `ELMD3`, `ENAT3`, `GPIV33`, `JBSS3`, `NEOE3`, `NTCO3`, `PETZ3`, `PORT3`, `SOMA3`, `STBP3`, `SULA11`, `ZAMP3`.
- Preço de classe de pregão antigo (`vm_stale_class_price`): 3 classe(s).
- Escala das contagens (`share_scale.scaled`): 66 empresa(s). Não verificada (`share_scale.unverified`): nenhum.

Escala das contagens, com o fator e o resíduo de cada empresa:

| cd_cvm | fator | resíduo |
| --- | --- | --- |
| `1155` | 10^3 | 1,00 |
| `11762` | 10^3 | 1,00 |
| `11932` | 10^3 | 1,00 |
| `11975` | 10^1 | 2,66 |
| `1210` | 10^3 | 1,00 |
| `1325` | 10^3 | 1,00 |
| `13773` | 10^3 | 1,00 |
| `13781` | 10^3 | 1,00 |
| `14109` | 10^3 | 1,00 |
| `14460` | 10^3 | 1,01 |
| `14826` | 10^3 | 1,00 |
| `16993` | 10^3 | 1,00 |
| `18724` | 10^3 | 1,00 |
| `19330` | 10^3 | 1,00 |
| `19348` | 10^3 | 1,00 |
| `19445` | 10^3 | 1,00 |
| `19453` | 10^3 | 1,00 |
| `19550` | 10^3 | 1,00 |
| `19615` | 10^3 | 1,00 |
| `19879` | 10^1 | 2,31 |
| `20044` | 10^3 | 1,00 |
| `20087` | 10^3 | 1,00 |
| `20257` | 10^3 | 1,00 |
| `20346` | 10^3 | 1,00 |
| `20362` | 10^3 | 1,00 |
| `20370` | 10^3 | 1,00 |
| `20532` | 10^3 | 1,00 |
| `20877` | 10^3 | 1,00 |
| `20915` | 10^3 | 1,00 |
| `21016` | 10^3 | 1,00 |
| `21040` | 10^3 | 1,00 |
| `21091` | 10^3 | 1,00 |
| `21130` | 10^3 | 1,04 |
| `21148` | 10^3 | 1,00 |
| `21237` | 10^3 | 1,00 |
| `21440` | 10^3 | 1,00 |
| `22012` | 10^3 | 1,00 |
| `22675` | 10^3 | 1,00 |
| `23264` | 10^3 | 1,02 |
| `23272` | 10^3 | 1,00 |
| `23795` | 10^3 | 1,00 |
| `2429` | 10^3 | 1,00 |
| `24295` | 10^3 | 1,00 |
| `2437` | 10^3 | 1,07 |
| `2453` | 10^3 | 1,00 |
| `2461` | 10^3 | 1,00 |
| `25038` | 10^3 | 1,00 |
| `25046` | 10^3 | 1,00 |
| `25062` | 10^-1 | 1,00 |
| `25330` | 10^3 | 1,00 |
| `25372` | 10^3 | 1,00 |
| `25550` | 10^3 | 0,70 |
| `25682` | 10^1 | 2,51 |
| `25984` | 10^3 | 1,00 |
| `26000` | 10^3 | 1,00 |
| `26620` | 10^3 | 1,00 |
| `4170` | 10^3 | 0,96 |
| `5380` | 10^3 | 0,97 |
| `5762` | 10^3 | 1,00 |
| `5770` | 10^3 | 1,00 |
| `6211` | 10^3 | 0,99 |
| `6343` | 10^3 | 1,00 |
| `6505` | 10^3 | 0,94 |
| `7617` | 10^3 | 1,00 |
| `7870` | 10^3 | 1,00 |
| `8133` | 10^3 | 1,00 |

Empresas com preço de classe de pregão antigo:

| cd_cvm | classe | ticker | último pregão |
| --- | --- | --- | --- |
| `14311` | PREF | `CPLE5` | 2025-12-19 |
| `16993` | ON | `EMAE3` | 2025-08-18 |
| `8753` | ON | `MTSA3` | 2025-06-23 |

Eventos da B3 que ajustam a contagem, por classe (`share_events.adjusted`):

| cd_cvm | classe | ISINs | DT_REFER | fator de eventos | eventos |
| --- | --- | --- | --- | --- | --- |
| `11975` | ON | `BRAZEVACNOR0` | 2026-03-31 | 0,05 | `BRAZEVACNOR0` GRUPAMENTO, data-ex 2026-06-26, ×0,05 |
| `11975` | PREF | `BRAZEVACNPR7` | 2026-03-31 | 0,05 | `BRAZEVACNPR7` GRUPAMENTO, data-ex 2026-06-26, ×0,05 |
| `14443` | ON | `BRSBSPACNOR5` | 2026-03-31 | 5 | `BRSBSPACNOR5` DESDOBRAMENTO, data-ex 2026-04-29, ×5 |
| `24961` | ON | `BRAMBPACNOR6` | 2025-06-30 | 10 | `BRAMBPACNOR6` DESDOBRAMENTO, data-ex 2025-08-05, ×10 |
| `25445` | ON | `BRESPAACNOR7` | 2026-03-31 | 0,1 | `BRESPAACNOR7` GRUPAMENTO, data-ex 2026-06-15, ×0,1 |
| `25763` | ON | `BRVITTACNOR4` | 2026-03-31 | 1,1 | `BRVITTACNOR4` BONIFICACAO, data-ex 2026-04-10, ×1,1 |

Classes com ações e sem ISIN em d (`share_events.no_isin`):

| cd_cvm | classe |
| --- | --- |
| `14443` | PREF |
| `14664` | ON |
| `20958` | ON |
| `24600` | ON |
| `25127` | PREF |
| `25208` | ON |
| `25860` | ON |
| `25860` | PREF |
| `25917` | ON |

Contagens suspeitas, sem evento da B3 que as explique (`share_count_suspect`):

| cd_cvm | DT_REFER anterior | total anterior | DT_REFER | total | razão |
| --- | --- | --- | --- | --- | --- |
| `19550` | 2025-12-31 | 1773371000 | 2026-03-31 | 1374558000 | 0,7751 |
| `20125` | 2026-03-31 | 5458252860 | 2026-06-30 | 2924199731 | 0,5357 |
| `21067` | 2025-12-31 | 84909375 | 2026-03-31 | 104214394 | 1,2274 |
| `26123` | 2025-09-30 | 651756313 | 2025-12-31 | 1132929494 | 1,7383 |
| `6505` | 2025-12-31 | 653878000 | 2026-03-31 | 950628000 | 1,4538 |

Fluxos de 12 meses refeitos com o ITR do ano anterior (`ttm_prior_from_itr`):

| cd_cvm | DT_REFER | contas |
| --- | --- | --- |
| `1023` | 2026-03-31 | `3.08` |
| `11398` | 2026-03-31 | `3.08` |
| `1155` | 2026-03-31 | `3.08` |
| `1210` | 2026-03-31 | `3.08` |
| `1325` | 2026-06-30 | `3.08` |
| `18724` | 2026-03-31 | `3.01` |
| `19348` | 2026-06-30 | `3.08` |
| `20532` | 2026-06-30 | `3.08` |
| `20567` | 2026-03-31 | `3.08` |
| `20958` | 2026-06-30 | `3.08` |
| `21440` | 2026-03-31 | `3.08` |
| `22616` | 2026-03-31 | `3.08` |
| `23159` | 2026-06-30 | `3.01` |
| `23795` | 2026-03-31 | `3.01` |
| `24600` | 2026-03-31 | `3.08` |
| `24929` | 2026-06-30 | `3.01`, `3.05`, `3.08`, `LL` |
| `906` | 2026-06-30 | `3.08` |
| `922` | 2026-03-31 | `3.08` |

Tickers com filing antigo demais para os indicadores contábeis (`stale_filing`):

| ticker | cd_cvm | DT_REFER | idade (dias) |
| --- | --- | --- | --- |
| `AMBP3` | `24961` | 2025-06-30 | 403 |

Tickers não mapeados:

| ticker | motivo |
| --- | --- |
| `AESB3` | `no_quote` |
| `APER3` | `no_quote` |
| `AXIA6` | `no_quote` |
| `AZUL53` | `no_quote` |
| `AZUL54` | `no_quote` |
| `BPAN4` | `no_quote` |
| `BRFS3` | `no_quote` |
| `CLSA3` | `no_quote` |
| `CPLE5` | `no_quote` |
| `CPLE6` | `no_quote` |
| `CRFB3` | `no_quote` |
| `ELMD3` | `no_quote` |
| `ENAT3` | `no_quote` |
| `GPIV33` | `no_cvm_code` |
| `JBSS3` | `no_quote` |
| `NEOE3` | `no_quote` |
| `NTCO3` | `no_quote` |
| `PETZ3` | `no_quote` |
| `PORT3` | `no_quote` |
| `SOMA3` | `no_quote` |
| `STBP3` | `no_quote` |
| `SULA11` | `no_quote` |
| `ZAMP3` | `no_quote` |

Grupo financeiro, com a regra que classificou cada ticker:

| ticker | regra |
| --- | --- |
| `ABCB4` | `setor_ativ` |
| `APER3` | `setor_ativ` |
| `BAZA3` | `setor_ativ` |
| `BBAS3` | `setor_ativ` |
| `BBDC3` | `setor_ativ` |
| `BBDC4` | `setor_ativ` |
| `BBSE3` | `setor_ativ` |
| `BEES3` | `setor_ativ` |
| `BMEB3` | `setor_ativ` |
| `BMEB4` | `setor_ativ` |
| `BMGB4` | `setor_ativ` |
| `BPAC11` | `setor_ativ` |
| `BPAC3` | `setor_ativ` |
| `BPAC5` | `setor_ativ` |
| `BPAN4` | `setor_ativ` |
| `BRBI11` | `setor_ativ` |
| `BRSR3` | `setor_ativ` |
| `BRSR6` | `setor_ativ` |
| `CXSE3` | `setor_ativ` |
| `IRBR3` | `setor_ativ` |
| `ITSA3` | `setor_ativ` |
| `ITSA4` | `setor_ativ` |
| `ITUB3` | `setor_ativ` |
| `ITUB4` | `setor_ativ` |
| `PINE3` | `setor_ativ` |
| `PINE4` | `setor_ativ` |
| `PSSA3` | `setor_ativ` |
| `QUAL3` | `setor_ativ` |
| `SANB11` | `setor_ativ` |
| `SANB3` | `setor_ativ` |
| `SANB4` | `setor_ativ` |
| `SULA11` | `setor_ativ` |
| `WIZC3` | `setor_ativ` |

Tickers não classificados:

| ticker | SETOR do Status Invest | motivo |
| --- | --- | --- |
| `GPIV33` | `Financeiro e Outros` | `no_cvm_code` |

Empresas com preço de outra classe:

| cd_cvm | classe sem cotação | preço de |
| --- | --- | --- |
| `14443` | `PREF` | `SBSP3` |
| `14664` | `ON` | `SHUL4` |
| `20958` | `ON` | `ABCB4` |
| `24600` | `ON` | `BMGB4` |
| `25127` | `PREF` | `PASS3` |
| `25208` | `ON` | `TFCO4` |
| `25917` | `ON` | `RAIZ4` |

### 2026-11-05

- Elegíveis: 274. Mapeados: 257. Não mapeados: 17.
- Mapeados pela B3 (`mapped_via_b3`): `ALUP3`, `ALUP4`, `AMAR3`, `BPAC11`, `BPAC3`, `BPAC5`, `CMIN3`, `CSNA3`, `EQPA3`, `KLBN3`, `KLBN4`, `MBRF3`, `TGMA3`.
- Caíram para o período anterior porque o mais recente foi reapresentado depois de d, com uma versão já recebida até d (`refiled_after_d`): nenhum.
- Usaram o preço da outra classe (`vm_price_fallback`): 8 empresa(s).
- Sem cotação recente (`reference_stale`, sem registro em [d − 30 dias, d]): `AESB3`, `AXIA6`, `AZUL54`, `BPAN4`, `BRFS3`, `CPLE5`, `CPLE6`, `ENAT3`, `GPIV33`, `JBSS3`, `NEOE3`, `NTCO3`, `PETZ3`, `PORT3`, `STBP3`, `SULA11`, `ZAMP3`.
- Preço de classe de pregão antigo (`vm_stale_class_price`): 4 classe(s).
- Escala das contagens (`share_scale.scaled`): 66 empresa(s). Não verificada (`share_scale.unverified`): nenhum.

Escala das contagens, com o fator e o resíduo de cada empresa:

| cd_cvm | fator | resíduo |
| --- | --- | --- |
| `11762` | 10^3 | 1,00 |
| `11932` | 10^3 | 1,00 |
| `1210` | 10^3 | 1,00 |
| `1228` | 10^3 | 1,00 |
| `1325` | 10^3 | 1,00 |
| `13773` | 10^3 | 1,00 |
| `13781` | 10^3 | 1,00 |
| `14109` | 10^3 | 1,00 |
| `14346` | 10^3 | 1,00 |
| `14460` | 10^3 | 1,00 |
| `14524` | 10^3 | 1,00 |
| `14826` | 10^3 | 1,00 |
| `16993` | 10^3 | 1,40 |
| `18724` | 10^3 | 1,00 |
| `19330` | 10^3 | 1,00 |
| `19348` | 10^3 | 1,00 |
| `19445` | 10^3 | 1,00 |
| `19453` | 10^3 | 1,00 |
| `19550` | 10^3 | 1,00 |
| `19615` | 10^3 | 1,00 |
| `19879` | 10^1 | 2,31 |
| `20044` | 10^3 | 1,00 |
| `20087` | 10^3 | 1,00 |
| `20257` | 10^3 | 1,00 |
| `20346` | 10^3 | 1,00 |
| `20362` | 10^3 | 1,00 |
| `20532` | 10^3 | 1,00 |
| `20915` | 10^3 | 1,00 |
| `21016` | 10^3 | 1,00 |
| `21040` | 10^3 | 1,00 |
| `21091` | 10^3 | 1,00 |
| `21130` | 10^3 | 1,00 |
| `21148` | 10^3 | 1,00 |
| `21237` | 10^3 | 1,00 |
| `21440` | 10^3 | 1,00 |
| `22012` | 10^3 | 1,00 |
| `22675` | 10^3 | 1,00 |
| `23264` | 10^3 | 1,00 |
| `23272` | 10^3 | 1,00 |
| `23574` | 10^-2 | 1,10 |
| `23795` | 10^3 | 1,00 |
| `2429` | 10^3 | 1,00 |
| `24295` | 10^3 | 1,00 |
| `2437` | 10^3 | 1,05 |
| `2453` | 10^3 | 1,00 |
| `2461` | 10^3 | 1,00 |
| `25038` | 10^3 | 1,00 |
| `25046` | 10^3 | 1,00 |
| `25330` | 10^3 | 1,00 |
| `25372` | 10^3 | 1,00 |
| `25496` | 10^-1 | 1,00 |
| `25550` | 10^3 | 1,00 |
| `25682` | 10^1 | 2,51 |
| `25984` | 10^3 | 1,00 |
| `26000` | 10^3 | 1,00 |
| `26620` | 10^3 | 1,00 |
| `4170` | 10^3 | 0,96 |
| `5380` | 10^3 | 0,61 |
| `5770` | 10^3 | 1,00 |
| `6211` | 10^3 | 1,00 |
| `6343` | 10^3 | 1,00 |
| `6505` | 10^3 | 0,96 |
| `7617` | 10^3 | 1,00 |
| `7870` | 10^3 | 1,00 |
| `8133` | 10^3 | 1,00 |
| `9393` | 10^1 | 0,70 |

Empresas com preço de classe de pregão antigo:

| cd_cvm | classe | ticker | último pregão |
| --- | --- | --- | --- |
| `11398` | ON | `CRPG3` | 2026-07-29 |
| `14311` | PREF | `CPLE5` | 2025-12-19 |
| `16993` | ON | `EMAE3` | 2025-08-18 |
| `24112` | PREF | `AZUL4` | 2025-05-28 |

Eventos da B3 que ajustam a contagem, por classe (`share_events.adjusted`):

| cd_cvm | classe | ISINs | DT_REFER | fator de eventos | eventos |
| --- | --- | --- | --- | --- | --- |
| `24961` | ON | `BRAMBPACNOR6` | 2025-06-30 | 10 | `BRAMBPACNOR6` DESDOBRAMENTO, data-ex 2025-08-05, ×10 |
| `25160` | ON | `BRSEQLACNOR2` | 2026-06-30 | 0,00666667 | `BRSEQLACNOR2` GRUPAMENTO, data-ex 2026-11-03, ×0,00666667 |
| `25550` | ON | `BRORVRACNOR0` | 2026-06-30 | 4 | `BRORVRACNOR0` DESDOBRAMENTO, data-ex 2026-08-11, ×4 |

Classes com ações e sem ISIN em d (`share_events.no_isin`):

| cd_cvm | classe |
| --- | --- |
| `14443` | PREF |
| `20958` | ON |
| `23574` | PREF |
| `24180` | PREF |
| `24600` | ON |
| `25208` | ON |
| `25860` | ON |
| `25860` | PREF |
| `25917` | ON |
| `6211` | PREF |

Contagens suspeitas, sem evento da B3 que as explique (`share_count_suspect`):

| cd_cvm | DT_REFER anterior | total anterior | DT_REFER | total | razão |
| --- | --- | --- | --- | --- | --- |
| `16993` | 2026-06-30 | 36924000 | 2026-09-30 | 51629000 | 1,3983 |
| `23574` | 2026-03-31 | 573353080 | 2026-06-30 | 315344194 | 0,5500 |
| `24112` | 2026-03-31 | 547308517788 | 2026-06-30 | 368557924 | 0,0007 |
| `24805` | 2026-06-30 | 236197769 | 2026-09-30 | 82118477 | 0,3477 |
| `25550` | 2026-03-31 | 96127000 | 2026-06-30 | 137324000 | 1,4286 |
| `25917` | 2026-03-31 | 10352509484 | 2026-06-30 | 4557597117 | 0,4402 |
| `5380` | 2026-06-30 | 22500000 | 2026-09-30 | 14170000 | 0,6298 |
| `9393` | 2026-03-31 | 1394546420 | 2026-06-30 | 2971982330 | 2,1311 |

Fluxos de 12 meses refeitos com o ITR do ano anterior (`ttm_prior_from_itr`):

| cd_cvm | DT_REFER | contas |
| --- | --- | --- |
| `1023` | 2026-06-30 | `3.08` |
| `11258` | 2026-06-30 | `3.08` |
| `1155` | 2026-06-30 | `3.08` |
| `1210` | 2026-06-30 | `3.08` |
| `1228` | 2026-06-30 | `3.08` |
| `1325` | 2026-09-30 | `3.08` |
| `14206` | 2025-06-30 | `3.08` |
| `18724` | 2026-06-30 | `3.01` |
| `19348` | 2026-09-30 | `3.08` |
| `20532` | 2026-09-30 | `3.08` |
| `20567` | 2026-06-30 | `3.08` |
| `20958` | 2026-09-30 | `3.08` |
| `21440` | 2026-06-30 | `3.08` |
| `22616` | 2026-06-30 | `3.08` |
| `23159` | 2026-09-30 | `3.01` |
| `23795` | 2026-06-30 | `3.01` |
| `24600` | 2026-09-30 | `3.08` |
| `24929` | 2026-06-30 | `3.01`, `3.05`, `3.08`, `LL` |
| `906` | 2026-09-30 | `3.08` |
| `922` | 2026-06-30 | `3.08` |

Tickers com filing antigo demais para os indicadores contábeis (`stale_filing`):

| ticker | cd_cvm | DT_REFER | idade (dias) |
| --- | --- | --- | --- |
| `AALR3` | `24058` | 2025-12-31 | 309 |
| `AMBP3` | `24961` | 2025-06-30 | 493 |
| `BSLI3` | `14206` | 2025-06-30 | 493 |

Tickers não mapeados:

| ticker | motivo |
| --- | --- |
| `AESB3` | `no_quote` |
| `AXIA6` | `no_quote` |
| `AZUL54` | `no_quote` |
| `BPAN4` | `no_quote` |
| `BRFS3` | `no_quote` |
| `CPLE5` | `no_quote` |
| `CPLE6` | `no_quote` |
| `ENAT3` | `no_quote` |
| `GPIV33` | `no_cvm_code` |
| `JBSS3` | `no_quote` |
| `NEOE3` | `no_quote` |
| `NTCO3` | `no_quote` |
| `PETZ3` | `no_quote` |
| `PORT3` | `no_quote` |
| `STBP3` | `no_quote` |
| `SULA11` | `no_quote` |
| `ZAMP3` | `no_quote` |

Grupo financeiro, com a regra que classificou cada ticker:

| ticker | regra |
| --- | --- |
| `ABCB4` | `setor_ativ` |
| `BAZA3` | `setor_ativ` |
| `BBAS3` | `setor_ativ` |
| `BBDC3` | `setor_ativ` |
| `BBDC4` | `setor_ativ` |
| `BBSE3` | `setor_ativ` |
| `BEES3` | `setor_ativ` |
| `BEES4` | `setor_ativ` |
| `BMEB3` | `setor_ativ` |
| `BMEB4` | `setor_ativ` |
| `BMGB4` | `setor_ativ` |
| `BNBR3` | `setor_ativ` |
| `BPAC11` | `setor_ativ` |
| `BPAC3` | `setor_ativ` |
| `BPAC5` | `setor_ativ` |
| `BPAN4` | `setor_ativ` |
| `BRBI11` | `setor_ativ` |
| `BRSR3` | `setor_ativ` |
| `BRSR6` | `setor_ativ` |
| `BSLI3` | `setor_ativ` |
| `CXSE3` | `setor_ativ` |
| `IRBR3` | `setor_ativ` |
| `ITSA3` | `setor_ativ` |
| `ITSA4` | `setor_ativ` |
| `ITUB3` | `setor_ativ` |
| `ITUB4` | `setor_ativ` |
| `PINE3` | `setor_ativ` |
| `PINE4` | `setor_ativ` |
| `PSSA3` | `setor_ativ` |
| `QUAL3` | `setor_ativ` |
| `SANB11` | `setor_ativ` |
| `SANB3` | `setor_ativ` |
| `SANB4` | `setor_ativ` |
| `SULA11` | `setor_ativ` |
| `WIZC3` | `setor_ativ` |

Tickers não classificados:

| ticker | SETOR do Status Invest | motivo |
| --- | --- | --- |
| `GPIV33` | `Financeiro e Outros` | `no_cvm_code` |

Empresas com preço de outra classe:

| cd_cvm | classe sem cotação | preço de |
| --- | --- | --- |
| `14443` | `PREF` | `SBSP3` |
| `20958` | `ON` | `ABCB4` |
| `23574` | `PREF` | `MEAL3` |
| `24180` | `PREF` | `IRBR3` |
| `24600` | `ON` | `BMGB4` |
| `25208` | `ON` | `TFCO4` |
| `25917` | `ON` | `RAIZ4` |
| `6211` | `PREF` | `FRAS3` |

## Resultados por data

### 2026-08-07

| indicador | grupo | n | % na tolerância | mediana | p90 | veredito |
| --- | --- | --- | --- | --- | --- | --- |
| `LIQ. CORRENTE` | não financeiro | 236 | 89,41% | 0,00% | 0,00% | com ressalvas |
| `LIQ. CORRENTE` | não financeiro sem não classificados | 235 | 89,79% | 0,00% | 0,00% | com ressalvas |
| `LIQ. CORRENTE` | financeiro | 29 | 58,62% | 0,00% | 700,00% | não reconstruível |
| `DIVIDA LIQUIDA / EBIT` | não financeiro | 236 | 86,44% | 0,00% | 0,00% | com ressalvas |
| `DIVIDA LIQUIDA / EBIT` | não financeiro sem não classificados | 235 | 86,81% | 0,00% | 0,00% | com ressalvas |
| `DIVIDA LIQUIDA / EBIT` | financeiro | 10 | 70,00% | 0,00% | 5,26% | com ressalvas |
| `DIV. LIQ. / PATRI.` | não financeiro | 230 | 88,70% | 0,00% | 0,00% | com ressalvas |
| `DIV. LIQ. / PATRI.` | não financeiro sem não classificados | 229 | 89,08% | 0,00% | 0,00% | com ressalvas |
| `DIV. LIQ. / PATRI.` | financeiro | 10 | 70,00% | 0,00% | 53,85% | com ressalvas |
| `ROE` | não financeiro | 236 | 85,59% | 0,00 p.p. | 0,00 p.p. | com ressalvas |
| `ROE` | não financeiro sem não classificados | 235 | 85,96% | 0,00 p.p. | 0,00 p.p. | com ressalvas |
| `ROE` | financeiro | 33 | 27,27% | 8,29 p.p. | 202,25 p.p. | não reconstruível |
| `ROA` | não financeiro | 236 | 87,29% | 0,00 p.p. | 0,00 p.p. | com ressalvas |
| `ROA` | não financeiro sem não classificados | 235 | 87,66% | 0,00 p.p. | 0,00 p.p. | com ressalvas |
| `ROA` | financeiro | 33 | 63,64% | 0,00 p.p. | 0,05 p.p. | não reconstruível |
| `ROIC` | não financeiro | 236 | 48,31% | 0,44 p.p. | 4,00 p.p. | não reconstruível |
| `ROIC` | não financeiro sem não classificados | 235 | 48,51% | 0,44 p.p. | 4,00 p.p. | não reconstruível |
| `ROIC` | financeiro | 10 | 50,00% | 0,20 p.p. | 43,90 p.p. | não reconstruível |
| `MARG. LIQUIDA` | não financeiro | 234 | 85,04% | 0,00 p.p. | 0,02 p.p. | com ressalvas |
| `MARG. LIQUIDA` | não financeiro sem não classificados | 233 | 85,41% | 0,00 p.p. | 0,02 p.p. | com ressalvas |
| `MARG. LIQUIDA` | financeiro | 32 | 56,25% | 0,01 p.p. | 0,21 p.p. | não reconstruível |
| `MARGEM EBIT` | não financeiro | 234 | 86,75% | 0,00 p.p. | 0,01 p.p. | com ressalvas |
| `MARGEM EBIT` | não financeiro sem não classificados | 233 | 87,12% | 0,00 p.p. | 0,01 p.p. | com ressalvas |
| `MARGEM EBIT` | financeiro | 32 | 68,75% | 0,00 p.p. | 39,02 p.p. | não reconstruível |
| `P/L` | não financeiro | 236 | 80,08% | 0,00% | 5,57% | com ressalvas |
| `P/L` | não financeiro sem não classificados | 235 | 80,43% | 0,00% | 5,57% | com ressalvas |
| `P/L` | financeiro | 33 | 60,61% | 0,00% | 3,30% | não reconstruível |
| `P/VP` | não financeiro | 236 | 83,05% | 0,00% | 3,51% | com ressalvas |
| `P/VP` | não financeiro sem não classificados | 235 | 83,40% | 0,00% | 3,51% | com ressalvas |
| `P/VP` | financeiro | 33 | 27,27% | 86,75% | 1265,32% | não reconstruível |
| `EV/EBIT` | não financeiro | 236 | 80,51% | 0,00% | 5,92% | com ressalvas |
| `EV/EBIT` | não financeiro sem não classificados | 235 | 80,85% | 0,00% | 5,92% | com ressalvas |
| `EV/EBIT` | financeiro | 33 | 18,18% | 0,47% | 356,17% | não reconstruível |
| `PSR` | não financeiro | 233 | 81,55% | 0,00% | 5,95% | com ressalvas |
| `PSR` | não financeiro sem não classificados | 233 | 81,55% | 0,00% | 5,95% | com ressalvas |
| `PSR` | financeiro | 32 | 75,00% | 0,00% | 111,32% | com ressalvas |
| `CAGR RECEITAS 5 ANOS` | não financeiro | 230 | 89,13% | 0,00 p.p. | 0,00 p.p. | com ressalvas |
| `CAGR RECEITAS 5 ANOS` | não financeiro sem não classificados | 230 | 89,13% | 0,00 p.p. | 0,00 p.p. | com ressalvas |
| `CAGR RECEITAS 5 ANOS` | financeiro | 31 | 80,65% | 0,00 p.p. | 7,77 p.p. | com ressalvas |
| `CAGR LUCROS 5 ANOS` | não financeiro | 160 | 86,88% | 0,00 p.p. | 0,00 p.p. | com ressalvas |
| `CAGR LUCROS 5 ANOS` | não financeiro sem não classificados | 160 | 86,88% | 0,00 p.p. | 0,00 p.p. | com ressalvas |
| `CAGR LUCROS 5 ANOS` | financeiro | 29 | 51,72% | 0,00 p.p. | 0,03 p.p. | não reconstruível |
| `PEG RATIO` | não financeiro | 235 | 68,94% | 0,00% | 4,00% | não reconstruível |
| `PEG RATIO` | não financeiro sem não classificados | 234 | 69,23% | 0,00% | 4,00% | não reconstruível |
| `PEG RATIO` | financeiro | 33 | 54,55% | 0,00% | 0,00% | não reconstruível |
| `DY` | não financeiro | 236 | 69,92% | 0,00 p.p. | 2,01 p.p. | não reconstruível |
| `DY` | não financeiro sem não classificados | 235 | 70,21% | 0,00 p.p. | 2,01 p.p. | com ressalvas |
| `DY` | financeiro | 33 | 78,79% | 0,00 p.p. | 0,56 p.p. | com ressalvas |
| `PRECO` | não financeiro | 236 | 91,53% | 0,00% | 0,00% | reconstruível |
| `PRECO` | não financeiro sem não classificados | 235 | 91,91% | 0,00% | 0,00% | reconstruível |
| `PRECO` | financeiro | 33 | 90,91% | 0,00% | 0,00% | reconstruível |
| `VALOR DE MERCADO` | não financeiro | 236 | 83,05% | 0,00% | 4,13% | com ressalvas |
| `VALOR DE MERCADO` | não financeiro sem não classificados | 235 | 83,40% | 0,00% | 4,13% | com ressalvas |
| `VALOR DE MERCADO` | financeiro | 33 | 75,76% | 0,00% | 14,56% | com ressalvas |
| `LIQUIDEZ MEDIA DIARIA` | não financeiro | 236 | 86,86% | 0,35% | 2,02% | com ressalvas |
| `LIQUIDEZ MEDIA DIARIA` | não financeiro sem não classificados | 235 | 87,23% | 0,35% | 2,02% | com ressalvas |
| `LIQUIDEZ MEDIA DIARIA` | financeiro | 33 | 87,88% | 0,35% | 1,50% | com ressalvas |

Referências em branco no Status Invest (grupo não financeiro):

| indicador | em branco | com valor reconstruído | DY tratado como 0,00 |
| --- | --- | --- | --- |
| `DIV. LIQ. / PATRI.` | 6 | 4 | 0 |
| `MARG. LIQUIDA` | 2 | 0 | 0 |
| `MARGEM EBIT` | 2 | 0 | 0 |
| `PSR` | 3 | 0 | 0 |
| `CAGR RECEITAS 5 ANOS` | 6 | 1 | 0 |
| `CAGR LUCROS 5 ANOS` | 76 | 1 | 0 |
| `PEG RATIO` | 1 | 1 | 0 |
| `DY` | 0 | 0 | 63 |

<details>
<summary>LIQ. CORRENTE, não financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `VTRU3` | 4,48 | 2,13 | 52,46% |
| `ONCO3` | 0,46 | 0,56 | 21,74% |
| `BRST3` | 1,54 | 1,31 | 14,94% |
| `BHIA3` | 0,76 | 0,72 | 5,26% |
| `TIMS3` | 0,89 | 0,85 | 4,49% |
| `CAML3` | 1,95 | 1,87 | 4,10% |
| `AALR3` | 0,28 | 0,28 | 0,00% |
| `ABEV3` | 1,02 | 1,02 | 0,00% |
| `AGRO3` | 1,47 | 1,47 | 0,00% |
| `ALLD3` | 1,74 | 1,74 | 0,00% |

Sem valor reconstruído: `AESB3` (no_quote), `AMBP3` (stale_filing), `AXIA6` (no_quote), `AZUL53` (no_quote), `AZUL54` (no_quote), `BRFS3` (no_quote), `CLSA3` (no_quote), `CPLE5` (no_quote), `CPLE6` (no_quote), `CRFB3` (no_quote), `ELMD3` (no_quote), `ENAT3` (no_quote), `GPIV33` (no_cvm_code), `JBSS3` (no_quote), `NEOE3` (no_quote), `NTCO3` (no_quote), `PETZ3` (no_quote), `PORT3` (no_quote), `SOMA3` (no_quote), `STBP3` (no_quote), `ZAMP3` (no_quote).

</details>

<details>
<summary>LIQ. CORRENTE, financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `BMGB4` | 0,01 | 2,81 | 28000,00% |
| `BPAC11` | 0,01 | 0,08 | 700,00% |
| `BPAC3` | 0,01 | 0,08 | 700,00% |
| `BPAC5` | 0,01 | 0,08 | 700,00% |
| `ITUB3` | 0,12 | 0,38 | 216,67% |
| `ITUB4` | 0,12 | 0,38 | 216,67% |
| `SANB11` | 0,74 | 0,38 | 48,65% |
| `SANB3` | 0,74 | 0,38 | 48,65% |
| `SANB4` | 0,74 | 0,38 | 48,65% |
| `ABCB4` | 0,19 | 0,19 | 0,00% |

Sem valor reconstruído: `APER3` (no_quote), `BPAN4` (no_quote), `SULA11` (no_quote).

</details>

<details>
<summary>DIVIDA LIQUIDA / EBIT, não financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `BHIA3` | -14,38 | 4,48 | 131,15% |
| `TIMS3` | -0,73 | -0,28 | 61,64% |
| `CAML3` | 5,02 | 7,45 | 48,41% |
| `VTRU3` | 3,68 | 2,31 | 37,23% |
| `BRST3` | 5,03 | 6,90 | 37,18% |
| `EMAE4` | 1,28 | 1,45 | 13,28% |
| `SOJA3` | 7,81 | 7,14 | 8,58% |
| `MOTV3` | 4,18 | 3,88 | 7,18% |
| `TPIS3` | -2,47 | -2,62 | 6,07% |
| `AMER3` | 3,69 | 3,88 | 5,15% |

Sem valor reconstruído: `AESB3` (no_quote), `AMBP3` (stale_filing), `AXIA6` (no_quote), `AZUL53` (no_quote), `AZUL54` (no_quote), `BRFS3` (no_quote), `CLSA3` (no_quote), `CPLE5` (no_quote), `CPLE6` (no_quote), `CRFB3` (no_quote), `ELMD3` (no_quote), `ENAT3` (no_quote), `EQPA3` (missing_input: EBIT), `GPIV33` (no_cvm_code), `JBSS3` (no_quote), `NEOE3` (no_quote), `NTCO3` (no_quote), `PETZ3` (no_quote), `PORT3` (no_quote), `SOMA3` (no_quote), `STBP3` (no_quote), `ZAMP3` (no_quote).

</details>

<details>
<summary>DIVIDA LIQUIDA / EBIT, financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `BBSE3` | -0,57 | -0,54 | 5,26% |
| `QUAL3` | 2,69 | 2,66 | 1,12% |
| `BRBI11` | -264,41 | -264,41 | 0,00% |
| `CXSE3` | -0,23 | -0,23 | 0,00% |
| `ITSA3` | 0,23 | 0,23 | 0,00% |
| `ITSA4` | 0,23 | 0,23 | 0,00% |
| `PSSA3` | -4,59 | -4,59 | 0,00% |
| `WIZC3` | -0,03 | -0,03 | 0,00% |

Sem valor reconstruído: `APER3` (no_quote), `SULA11` (no_quote).

</details>

<details>
<summary>DIV. LIQ. / PATRI., não financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `AALR3` | 0,34 | -1,27 | 473,53% |
| `VTRU3` | -0,54 | 0,39 | 172,22% |
| `TIMS3` | -0,39 | -0,08 | 79,49% |
| `BRST3` | 0,66 | 1,08 | 63,64% |
| `BHIA3` | 2,15 | 3,29 | 53,02% |
| `ONCO3` | 6,99 | 3,65 | 47,78% |
| `CAML3` | 1,08 | 1,50 | 38,89% |
| `ABEV3` | -0,17 | -0,17 | 0,00% |
| `AGRO3` | 0,66 | 0,66 | 0,00% |
| `ALLD3` | 0,09 | 0,09 | 0,00% |

Sem valor reconstruído: `AESB3` (no_quote), `AMBP3` (stale_filing), `AXIA6` (no_quote), `BRFS3` (no_quote), `CLSA3` (no_quote), `CPLE5` (no_quote), `CPLE6` (no_quote), `CRFB3` (no_quote), `ELMD3` (no_quote), `ENAT3` (no_quote), `GPIV33` (no_cvm_code), `JBSS3` (no_quote), `NEOE3` (no_quote), `NTCO3` (no_quote), `PETZ3` (no_quote), `PORT3` (no_quote), `SOMA3` (no_quote), `STBP3` (no_quote), `ZAMP3` (no_quote).

</details>

<details>
<summary>DIV. LIQ. / PATRI., financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `BBSE3` | -0,52 | -0,24 | 53,85% |
| `BRBI11` | -1,59 | -1,59 | 0,00% |
| `CXSE3` | -0,08 | -0,08 | 0,00% |
| `ITSA3` | 0,05 | 0,05 | 0,00% |
| `ITSA4` | 0,05 | 0,05 | 0,00% |
| `PSSA3` | -0,90 | -0,90 | 0,00% |
| `QUAL3` | 0,61 | 0,61 | 0,00% |
| `WIZC3` | -0,02 | -0,02 | 0,00% |

Sem valor reconstruído: `APER3` (no_quote), `SULA11` (no_quote).

</details>

<details>
<summary>ROE, não financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `ONCO3` | -720,24 | -402,23 | 318,01 p.p. |
| `BHIA3` | -49,14 | -204,95 | 155,81 p.p. |
| `VTRU3` | -29,57 | 29,95 | 59,52 p.p. |
| `AALR3` | -544,08 | -494,46 | 49,62 p.p. |
| `RAIZ4` | -326,87 | -309,75 | 17,12 p.p. |
| `INTB3` | 18,07 | 8,63 | 9,44 p.p. |
| `BRST3` | 7,44 | 2,53 | 4,91 p.p. |
| `EMAE4` | -18,27 | -14,30 | 3,97 p.p. |
| `CAML3` | 6,46 | 3,69 | 2,77 p.p. |
| `TIMS3` | 15,91 | 17,29 | 1,38 p.p. |

Sem valor reconstruído: `AESB3` (no_quote), `AMBP3` (stale_filing), `AXIA6` (no_quote), `AZUL53` (no_quote), `AZUL54` (no_quote), `BRFS3` (no_quote), `CLSA3` (no_quote), `CPLE5` (no_quote), `CPLE6` (no_quote), `CRFB3` (no_quote), `ELMD3` (no_quote), `ENAT3` (no_quote), `EQPA3` (missing_input: LL), `GPIV33` (no_cvm_code), `JBSS3` (no_quote), `NEOE3` (no_quote), `NTCO3` (no_quote), `PETZ3` (no_quote), `PORT3` (no_quote), `SOMA3` (no_quote), `STBP3` (no_quote), `ZAMP3` (no_quote).

</details>

<details>
<summary>ROE, financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `PINE3` | 30,83 | 5691,42 | 5660,59 p.p. |
| `PINE4` | 30,83 | 5691,42 | 5660,59 p.p. |
| `BMEB3` | 27,16 | 229,41 | 202,25 p.p. |
| `BMEB4` | 27,16 | 229,41 | 202,25 p.p. |
| `BEES3` | 13,26 | 180,94 | 167,68 p.p. |
| `BRSR3` | 15,11 | 69,17 | 54,06 p.p. |
| `BRSR6` | 15,11 | 69,17 | 54,06 p.p. |
| `BBAS3` | 8,20 | 32,36 | 24,16 p.p. |
| `BAZA3` | 11,77 | 20,83 | 9,06 p.p. |
| `BBDC3` | 13,56 | 5,27 | 8,29 p.p. |

Sem valor reconstruído: `ABCB4` (zero_divisor), `APER3` (no_quote), `BMGB4` (missing_input: LL), `BPAC11` (missing_input: LL), `BPAC3` (missing_input: LL), `BPAC5` (missing_input: LL), `BPAN4` (no_quote), `ITUB3` (missing_input: LL), `ITUB4` (missing_input: LL), `SANB11` (missing_input: LL), `SANB3` (missing_input: LL), `SANB4` (missing_input: LL), `SULA11` (no_quote).

</details>

<details>
<summary>ROA, não financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `VTRU3` | -24,02 | 16,06 | 40,08 p.p. |
| `ONCO3` | -57,34 | -50,82 | 6,52 p.p. |
| `BHIA3` | -5,11 | -10,93 | 5,82 p.p. |
| `INTB3` | 10,75 | 5,13 | 5,62 p.p. |
| `TIMS3` | 3,43 | 7,56 | 4,13 p.p. |
| `BRST3` | 2,86 | 0,85 | 2,01 p.p. |
| `EMAE4` | -6,52 | -5,11 | 1,41 p.p. |
| `CAML3` | 1,81 | 0,92 | 0,89 p.p. |
| `HBSA3` | -3,20 | -2,80 | 0,40 p.p. |
| `VIVA3` | 12,00 | 12,40 | 0,40 p.p. |

Sem valor reconstruído: `AESB3` (no_quote), `AMBP3` (stale_filing), `AXIA6` (no_quote), `AZUL53` (no_quote), `AZUL54` (no_quote), `BRFS3` (no_quote), `CLSA3` (no_quote), `CPLE5` (no_quote), `CPLE6` (no_quote), `CRFB3` (no_quote), `ELMD3` (no_quote), `ENAT3` (no_quote), `EQPA3` (missing_input: LL), `GPIV33` (no_cvm_code), `JBSS3` (no_quote), `NEOE3` (no_quote), `NTCO3` (no_quote), `PETZ3` (no_quote), `PORT3` (no_quote), `SOMA3` (no_quote), `STBP3` (no_quote), `ZAMP3` (no_quote).

</details>

<details>
<summary>ROA, financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `BBAS3` | 0,60 | 0,48 | 0,12 p.p. |
| `BMEB3` | 2,19 | 2,14 | 0,05 p.p. |
| `BMEB4` | 2,19 | 2,14 | 0,05 p.p. |
| `PINE3` | 1,60 | 1,56 | 0,04 p.p. |
| `PINE4` | 1,60 | 1,56 | 0,04 p.p. |
| `BBDC3` | 1,00 | 0,99 | 0,01 p.p. |
| `BBDC4` | 1,00 | 0,99 | 0,01 p.p. |
| `ABCB4` | 1,50 | 1,50 | 0,00 p.p. |
| `BAZA3` | 1,31 | 1,31 | 0,00 p.p. |
| `BBSE3` | 41,91 | 41,91 | 0,00 p.p. |

Sem valor reconstruído: `APER3` (no_quote), `BMGB4` (missing_input: LL), `BPAC11` (missing_input: LL), `BPAC3` (missing_input: LL), `BPAC5` (missing_input: LL), `BPAN4` (no_quote), `ITUB3` (missing_input: LL), `ITUB4` (missing_input: LL), `SANB11` (missing_input: LL), `SANB3` (missing_input: LL), `SANB4` (missing_input: LL), `SULA11` (no_quote).

</details>

<details>
<summary>ROIC, não financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `AALR3` | -547,76 | -520,46 | 27,30 p.p. |
| `VTRU3` | -29,52 | -2,93 | 26,59 p.p. |
| `BRKM3` | -56,79 | -30,26 | 26,53 p.p. |
| `BRKM5` | -56,79 | -30,26 | 26,53 p.p. |
| `EMAE4` | -81,96 | -59,18 | 22,78 p.p. |
| `ONCO3` | -69,32 | -51,40 | 17,92 p.p. |
| `BHIA3` | -11,40 | 3,19 | 14,59 p.p. |
| `LOGN3` | -4,33 | 5,75 | 10,08 p.p. |
| `CASH3` | -32,33 | -24,37 | 7,96 p.p. |
| `TGMA3` | 12,62 | 20,57 | 7,95 p.p. |

Sem valor reconstruído: `AESB3` (no_quote), `AMBP3` (stale_filing), `AXIA6` (no_quote), `AZUL53` (no_quote), `AZUL54` (no_quote), `BRFS3` (no_quote), `CLSA3` (no_quote), `CPLE5` (no_quote), `CPLE6` (no_quote), `CRFB3` (no_quote), `ELMD3` (no_quote), `ENAT3` (no_quote), `EQPA3` (missing_input: EBIT), `GPIV33` (no_cvm_code), `JBSS3` (no_quote), `NEOE3` (no_quote), `NTCO3` (no_quote), `PETZ3` (no_quote), `PORT3` (no_quote), `SOMA3` (no_quote), `STBP3` (no_quote), `ZAMP3` (no_quote).

</details>

<details>
<summary>ROIC, financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `BBSE3` | 68,52 | 24,62 | 43,90 p.p. |
| `QUAL3` | 4,11 | 6,12 | 2,01 p.p. |
| `WIZC3` | 21,48 | 22,60 | 1,12 p.p. |
| `CXSE3` | 30,17 | 30,54 | 0,37 p.p. |
| `ITSA3` | 17,44 | 17,41 | 0,03 p.p. |
| `ITSA4` | 17,44 | 17,41 | 0,03 p.p. |
| `BRBI11` | -0,19 | -0,20 | 0,01 p.p. |
| `PSSA3` | 13,32 | 13,33 | 0,01 p.p. |

Sem valor reconstruído: `APER3` (no_quote), `SULA11` (no_quote).

</details>

<details>
<summary>MARG. LIQUIDA, não financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `VTRU3` | -14,81 | 48,63 | 63,44 p.p. |
| `EMAE4` | -29,53 | -19,87 | 9,66 p.p. |
| `TIMS3` | 7,22 | 15,79 | 8,57 p.p. |
| `ONCO3` | -71,05 | -63,14 | 7,91 p.p. |
| `INTB3` | 13,11 | 6,26 | 6,85 p.p. |
| `BHIA3` | -5,98 | -12,30 | 6,32 p.p. |
| `BRST3` | 8,46 | 2,27 | 6,19 p.p. |
| `TPIS3` | -42,83 | -39,30 | 3,53 p.p. |
| `MOTV3` | 20,63 | 19,27 | 1,36 p.p. |
| `HBSA3` | -8,99 | -7,87 | 1,12 p.p. |

Sem valor reconstruído: `AESB3` (no_quote), `AMBP3` (stale_filing), `AXIA6` (no_quote), `AZUL53` (no_quote), `AZUL54` (no_quote), `BRFS3` (no_quote), `CLSA3` (no_quote), `CPLE5` (no_quote), `CPLE6` (no_quote), `CRFB3` (no_quote), `ELMD3` (no_quote), `ENAT3` (no_quote), `EQPA3` (missing_input: LL), `GPIV33` (no_cvm_code), `JBSS3` (no_quote), `NEOE3` (no_quote), `NTCO3` (no_quote), `PETZ3` (no_quote), `PORT3` (no_quote), `SOMA3` (no_quote), `STBP3` (no_quote), `ZAMP3` (no_quote).

</details>

<details>
<summary>MARG. LIQUIDA, financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `BBAS3` | 4,82 | 3,88 | 0,94 p.p. |
| `BMEB3` | 8,94 | 8,73 | 0,21 p.p. |
| `BMEB4` | 8,94 | 8,73 | 0,21 p.p. |
| `PINE3` | 10,22 | 10,09 | 0,13 p.p. |
| `PINE4` | 10,22 | 10,09 | 0,13 p.p. |
| `BBDC3` | 8,20 | 8,11 | 0,09 p.p. |
| `BBDC4` | 8,20 | 8,11 | 0,09 p.p. |
| `QUAL3` | 1,10 | 1,09 | 0,01 p.p. |
| `BRSR3` | 7,76 | 7,77 | 0,01 p.p. |
| `BRSR6` | 7,76 | 7,77 | 0,01 p.p. |

Sem valor reconstruído: `APER3` (no_quote), `BBSE3` (zero_divisor), `BMGB4` (missing_input: LL), `BPAC11` (missing_input: LL), `BPAC3` (missing_input: LL), `BPAC5` (missing_input: LL), `BPAN4` (no_quote), `ITUB3` (missing_input: LL), `ITUB4` (missing_input: LL), `SANB11` (missing_input: LL), `SANB3` (missing_input: LL), `SANB4` (missing_input: LL), `SULA11` (no_quote).

</details>

<details>
<summary>MARGEM EBIT, não financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `VTRU3` | -7,31 | 27,71 | 35,02 p.p. |
| `EMAE4` | -85,26 | -65,00 | 20,26 p.p. |
| `ONCO3` | -43,01 | -34,68 | 8,33 p.p. |
| `TPIS3` | -52,90 | -45,84 | 7,06 p.p. |
| `BHIA3` | -1,82 | 4,41 | 6,23 p.p. |
| `ISAE3` | 48,44 | 46,88 | 1,56 p.p. |
| `ISAE4` | 48,44 | 46,88 | 1,56 p.p. |
| `BRST3` | 14,96 | 14,00 | 0,96 p.p. |
| `VIVA3` | 23,33 | 22,40 | 0,93 p.p. |
| `EQTL3` | 15,18 | 15,53 | 0,35 p.p. |

Sem valor reconstruído: `AESB3` (no_quote), `AMBP3` (stale_filing), `AXIA6` (no_quote), `AZUL53` (no_quote), `AZUL54` (no_quote), `BRFS3` (no_quote), `CLSA3` (no_quote), `CPLE5` (no_quote), `CPLE6` (no_quote), `CRFB3` (no_quote), `ELMD3` (no_quote), `ENAT3` (no_quote), `EQPA3` (missing_input: EBIT), `GPIV33` (no_cvm_code), `JBSS3` (no_quote), `NEOE3` (no_quote), `NTCO3` (no_quote), `PETZ3` (no_quote), `PORT3` (no_quote), `SOMA3` (no_quote), `STBP3` (no_quote), `ZAMP3` (no_quote).

</details>

<details>
<summary>MARGEM EBIT, financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `BPAC11` | 29,09 | 68,11 | 39,02 p.p. |
| `BPAC3` | 29,09 | 68,11 | 39,02 p.p. |
| `BPAC5` | 29,09 | 68,11 | 39,02 p.p. |
| `IRBR3` | 8,01 | 0,19 | 7,82 p.p. |
| `PINE3` | 15,46 | 13,47 | 1,99 p.p. |
| `PINE4` | 15,46 | 13,47 | 1,99 p.p. |
| `QUAL3` | 21,25 | 21,34 | 0,09 p.p. |
| `BRSR3` | 9,22 | 9,23 | 0,01 p.p. |
| `BRSR6` | 9,22 | 9,23 | 0,01 p.p. |
| `ABCB4` | 15,46 | 15,46 | 0,00 p.p. |

Sem valor reconstruído: `APER3` (no_quote), `BBSE3` (zero_divisor), `BPAN4` (no_quote), `SULA11` (no_quote).

</details>

<details>
<summary>P/L, não financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `VVEO3` | 10,25 | 102,46 | 899,61% |
| `BRST3` | 10,58 | 31,04 | 193,38% |
| `AZEV4` | -0,32 | -0,86 | 168,75% |
| `AZEV3` | -0,68 | -1,82 | 167,65% |
| `LIGT3` | 1,91 | 4,42 | 131,41% |
| `VTRU3` | -6,32 | 1,53 | 124,21% |
| `INTB3` | 8,08 | 16,92 | 109,41% |
| `CAML3` | 8,25 | 14,57 | 76,61% |
| `BHIA3` | -0,46 | -0,19 | 58,70% |
| `TIMS3` | 22,67 | 10,37 | 54,26% |

Sem valor reconstruído: `AESB3` (no_quote), `AMBP3` (stale_filing), `AXIA6` (no_quote), `AZUL53` (no_quote), `AZUL54` (no_quote), `BRFS3` (no_quote), `CLSA3` (no_quote), `CPLE5` (no_quote), `CPLE6` (no_quote), `CRFB3` (no_quote), `ELMD3` (no_quote), `ENAT3` (no_quote), `EQPA3` (missing_input: LL), `GPIV33` (no_cvm_code), `JBSS3` (no_quote), `NEOE3` (no_quote), `NTCO3` (no_quote), `PETZ3` (no_quote), `PORT3` (no_quote), `SOMA3` (no_quote), `STBP3` (no_quote), `ZAMP3` (no_quote).

</details>

<details>
<summary>P/L, financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `BBAS3` | 7,32 | 9,08 | 24,04% |
| `PINE3` | 5,55 | 5,74 | 3,42% |
| `PINE4` | 5,76 | 5,95 | 3,30% |
| `BMEB4` | 8,68 | 8,86 | 2,07% |
| `BMEB3` | 7,38 | 7,53 | 2,03% |
| `BBDC4` | 7,46 | 7,54 | 1,07% |
| `BBDC3` | 6,53 | 6,60 | 1,07% |
| `ABCB4` | 6,01 | 6,01 | 0,00% |
| `BAZA3` | 4,19 | 4,19 | 0,00% |
| `BBSE3` | 8,11 | 8,11 | 0,00% |

Sem valor reconstruído: `APER3` (no_quote), `BMGB4` (missing_input: LL), `BPAC11` (missing_input: LL), `BPAC3` (missing_input: LL), `BPAC5` (missing_input: LL), `BPAN4` (no_quote), `ITUB3` (missing_input: LL), `ITUB4` (missing_input: LL), `SANB11` (missing_input: LL), `SANB3` (missing_input: LL), `SANB4` (missing_input: LL), `SULA11` (no_quote).

</details>

<details>
<summary>P/VP, não financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `VVEO3` | 0,11 | 1,10 | 900,00% |
| `AZEV3` | 3,65 | 9,73 | 166,58% |
| `AZEV4` | 1,73 | 4,60 | 165,90% |
| `LIGT3` | 0,60 | 1,39 | 131,67% |
| `VTRU3` | 1,87 | 0,46 | 75,40% |
| `BHIA3` | 0,23 | 0,38 | 65,22% |
| `TIMS3` | 3,61 | 1,79 | 50,42% |
| `ONCO3` | 1,32 | 0,78 | 40,91% |
| `ORVR3` | 6,38 | 4,47 | 29,94% |
| `SIMH3` | 0,97 | 0,71 | 26,80% |

Sem valor reconstruído: `AESB3` (no_quote), `AMBP3` (stale_filing), `AXIA6` (no_quote), `AZUL53` (no_quote), `AZUL54` (no_quote), `BRFS3` (no_quote), `CLSA3` (no_quote), `CPLE5` (no_quote), `CPLE6` (no_quote), `CRFB3` (no_quote), `ELMD3` (no_quote), `ENAT3` (no_quote), `GPIV33` (no_cvm_code), `JBSS3` (no_quote), `NEOE3` (no_quote), `NTCO3` (no_quote), `PETZ3` (no_quote), `PORT3` (no_quote), `SOMA3` (no_quote), `STBP3` (no_quote), `ZAMP3` (no_quote).

</details>

<details>
<summary>P/VP, financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `PINE3` | 1,71 | 326,71 | 19005,85% |
| `PINE4` | 1,78 | 338,64 | 18924,72% |
| `BEES3` | 1,24 | 16,93 | 1265,32% |
| `BMEB3` | 2,00 | 17,28 | 764,00% |
| `BMEB4` | 2,36 | 20,32 | 761,02% |
| `BBAS3` | 0,60 | 2,94 | 390,00% |
| `BRSR6` | 0,48 | 2,21 | 360,42% |
| `BRSR3` | 0,62 | 2,84 | 358,06% |
| `ITUB3` | 2,21 | 0,20 | 90,95% |
| `ITUB4` | 2,06 | 0,19 | 90,78% |

Sem valor reconstruído: `ABCB4` (zero_divisor), `APER3` (no_quote), `BPAN4` (no_quote), `SULA11` (no_quote).

</details>

<details>
<summary>EV/EBIT, não financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `TFCO4` | 10,90 | 64,31 | 490,00% |
| `VTRU3` | -9,11 | 4,99 | 154,77% |
| `BHIA3` | -15,89 | 5,00 | 131,47% |
| `AZEV3` | -1,26 | -2,46 | 95,24% |
| `AZEV4` | -1,26 | -2,46 | 95,24% |
| `SHUL4` | 23,73 | 6,81 | 71,30% |
| `VVEO3` | 5,40 | 8,65 | 60,19% |
| `LIGT3` | 16,63 | 24,76 | 48,89% |
| `CAML3` | 7,50 | 10,12 | 34,93% |
| `ORVR3` | 31,80 | 23,50 | 26,10% |

Sem valor reconstruído: `AESB3` (no_quote), `AMBP3` (stale_filing), `AXIA6` (no_quote), `AZUL53` (no_quote), `AZUL54` (no_quote), `BRFS3` (no_quote), `CLSA3` (no_quote), `CPLE5` (no_quote), `CPLE6` (no_quote), `CRFB3` (no_quote), `ELMD3` (no_quote), `ENAT3` (no_quote), `EQPA3` (missing_input: EBIT), `GPIV33` (no_cvm_code), `JBSS3` (no_quote), `NEOE3` (no_quote), `NTCO3` (no_quote), `PETZ3` (no_quote), `PORT3` (no_quote), `SOMA3` (no_quote), `STBP3` (no_quote), `ZAMP3` (no_quote).

</details>

<details>
<summary>EV/EBIT, financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `IRBR3` | 10,75 | 596,61 | 5449,86% |
| `BAZA3` | 8,83 | 40,28 | 356,17% |
| `ABCB4` | 4,90 | 11,37 | 132,04% |
| `BBSE3` | 6,90 | 15,02 | 117,68% |
| `QUAL3` | 4,24 | 4,20 | 0,94% |
| `CXSE3` | 11,98 | 11,98 | 0,00% |
| `ITSA3` | 8,25 | 8,25 | 0,00% |
| `ITSA4` | 8,25 | 8,25 | 0,00% |
| `PSSA3` | 5,74 | 5,74 | 0,00% |
| `WIZC3` | 2,22 | 2,22 | 0,00% |

Sem valor reconstruído: `APER3` (no_quote), `BBAS3` (missing_input: DB), `BBDC3` (missing_input: DB), `BBDC4` (missing_input: DB), `BEES3` (missing_input: DB), `BMEB3` (missing_input: DB), `BMEB4` (missing_input: DB), `BMGB4` (missing_input: DB), `BPAC11` (missing_input: DB), `BPAC3` (missing_input: DB), `BPAC5` (missing_input: DB), `BPAN4` (no_quote), `BRBI11` (no_quote), `BRSR3` (missing_input: DB), `BRSR6` (missing_input: DB), `ITUB3` (missing_input: DB), `ITUB4` (missing_input: DB), `PINE3` (missing_input: DB), `PINE4` (missing_input: DB), `SANB11` (missing_input: DB), `SANB3` (missing_input: DB), `SANB4` (missing_input: DB), `SULA11` (no_quote).

</details>

<details>
<summary>PSR, não financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `VVEO3` | 0,02 | 0,18 | 800,00% |
| `AZEV3` | 0,92 | 2,46 | 167,39% |
| `AZEV4` | 0,44 | 1,16 | 163,64% |
| `LIGT3` | 0,33 | 0,75 | 127,27% |
| `BHIA3` | 0,03 | 0,02 | 33,33% |
| `ORVR3` | 8,17 | 5,72 | 29,99% |
| `SIMH3` | 0,09 | 0,07 | 22,22% |
| `VTRU3` | 0,94 | 0,74 | 21,28% |
| `BRST3` | 0,89 | 0,71 | 20,22% |
| `EGIE3` | 3,08 | 2,48 | 19,48% |

Sem valor reconstruído: `AESB3` (no_quote), `AMBP3` (stale_filing), `AXIA6` (no_quote), `AZUL53` (no_quote), `AZUL54` (no_quote), `BRFS3` (no_quote), `CLSA3` (no_quote), `CPLE5` (no_quote), `CPLE6` (no_quote), `CRFB3` (no_quote), `ELMD3` (no_quote), `ENAT3` (no_quote), `EQPA3` (missing_input: REC), `JBSS3` (no_quote), `NEOE3` (no_quote), `NTCO3` (no_quote), `PETZ3` (no_quote), `PORT3` (no_quote), `SOMA3` (no_quote), `STBP3` (no_quote), `ZAMP3` (no_quote).

</details>

<details>
<summary>PSR, financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `BPAC11` | 3,19 | 20,24 | 534,48% |
| `BPAC3` | 3,94 | 8,33 | 111,42% |
| `BPAC5` | 3,18 | 6,72 | 111,32% |
| `BMGB4` | 0,39 | 0,36 | 7,69% |
| `BBDC3` | 0,54 | 0,53 | 1,85% |
| `PINE3` | 0,57 | 0,58 | 1,75% |
| `PINE4` | 0,59 | 0,60 | 1,69% |
| `BMEB4` | 0,78 | 0,77 | 1,28% |
| `ABCB4` | 0,76 | 0,76 | 0,00% |
| `BAZA3` | 0,53 | 0,53 | 0,00% |

Sem valor reconstruído: `APER3` (no_quote), `BBSE3` (zero_divisor), `BPAN4` (no_quote), `SULA11` (no_quote).

</details>

<details>
<summary>CAGR RECEITAS 5 ANOS, não financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `MLAS3` | 6,65 | 4,98 | 1,67 p.p. |
| `EQPA3` | 15,78 | 14,19 | 1,59 p.p. |
| `BHIA3` | 1,39 | 0,20 | 1,19 p.p. |
| `VALE3` | 0,72 | 0,48 | 0,24 p.p. |
| `AALR3` | 4,43 | 4,43 | 0,00 p.p. |
| `ABEV3` | 8,61 | 8,61 | 0,00 p.p. |
| `AGRO3` | 10,70 | 10,70 | 0,00 p.p. |
| `ALLD3` | 3,11 | 3,11 | 0,00 p.p. |
| `ALOS3` | 29,09 | 29,09 | 0,00 p.p. |
| `ALPA3` | 6,29 | 6,29 | 0,00 p.p. |

Sem valor reconstruído: `AMBP3` (stale_filing), `AURE3` (short_history), `AXIA6` (no_quote), `AZUL53` (no_quote), `AZUL54` (no_quote), `BRFS3` (no_quote), `BRST3` (short_history), `CLSA3` (no_quote), `CPLE5` (no_quote), `CPLE6` (no_quote), `CRFB3` (no_quote), `ELMD3` (no_quote), `ENAT3` (no_quote), `JBSS3` (no_quote), `LAND3` (short_history), `NEOE3` (no_quote), `NTCO3` (no_quote), `PETZ3` (no_quote), `PORT3` (no_quote), `SOMA3` (no_quote), `STBP3` (no_quote), `ZAMP3` (no_quote).

</details>

<details>
<summary>CAGR RECEITAS 5 ANOS, financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `BPAC11` | 16,36 | 8,59 | 7,77 p.p. |
| `BPAC3` | 16,36 | 8,59 | 7,77 p.p. |
| `BPAC5` | 16,36 | 8,59 | 7,77 p.p. |
| `ABCB4` | 28,96 | 28,96 | 0,00 p.p. |
| `BAZA3` | 46,47 | 46,47 | 0,00 p.p. |
| `BBAS3` | 26,49 | 26,49 | 0,00 p.p. |
| `BBDC3` | 22,38 | 22,38 | 0,00 p.p. |
| `BBDC4` | 22,38 | 22,38 | 0,00 p.p. |
| `BEES3` | 28,74 | 28,74 | 0,00 p.p. |
| `BMEB3` | 29,99 | 29,99 | 0,00 p.p. |

Sem valor reconstruído: `APER3` (no_quote), `BPAN4` (no_quote), `SULA11` (no_quote).

</details>

<details>
<summary>CAGR LUCROS 5 ANOS, não financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `TIMS3` | 1,43 | 18,60 | 17,17 p.p. |
| `INTB3` | 12,71 | -2,77 | 15,48 p.p. |
| `CAML3` | -15,89 | -24,93 | 9,04 p.p. |
| `VIVA3` | 31,91 | 32,77 | 0,86 p.p. |
| `GMAT3` | 18,99 | 19,56 | 0,57 p.p. |
| `SOJA3` | -18,88 | -18,62 | 0,26 p.p. |
| `MOTV3` | 82,59 | 82,60 | 0,01 p.p. |
| `ABEV3` | 7,38 | 7,38 | 0,00 p.p. |
| `ALLD3` | 15,35 | 15,35 | 0,00 p.p. |
| `ALOS3` | 40,77 | 40,77 | 0,00 p.p. |

Sem valor reconstruído: `AXIA6` (no_quote), `BRFS3` (no_quote), `BRST3` (short_history), `CPLE5` (no_quote), `CPLE6` (no_quote), `CRFB3` (no_quote), `DXCO3` (non_positive_base), `ELMD3` (no_quote), `ENAT3` (no_quote), `EQPA3` (missing_input: LL_TTM), `JBSS3` (no_quote), `LAND3` (short_history), `NEOE3` (no_quote), `PETZ3` (no_quote), `PORT3` (no_quote), `SOMA3` (no_quote).

</details>

<details>
<summary>CAGR LUCROS 5 ANOS, financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `BBAS3` | 3,40 | 1,32 | 2,08 p.p. |
| `BBDC3` | 8,93 | 8,96 | 0,03 p.p. |
| `BBDC4` | 8,93 | 8,96 | 0,03 p.p. |
| `ABCB4` | 24,52 | 24,52 | 0,00 p.p. |
| `BAZA3` | 26,06 | 26,06 | 0,00 p.p. |
| `BBSE3` | 19,00 | 19,00 | 0,00 p.p. |
| `BEES3` | 8,03 | 8,03 | 0,00 p.p. |
| `BRBI11` | 12,45 | 12,45 | 0,00 p.p. |
| `BRSR3` | 23,23 | 23,23 | 0,00 p.p. |
| `BRSR6` | 23,23 | 23,23 | 0,00 p.p. |

Sem valor reconstruído: `BMEB3` (short_history), `BMEB4` (short_history), `BMGB4` (missing_input: LL_TTM), `BPAC11` (missing_input: LL_TTM), `BPAC3` (missing_input: LL_TTM), `BPAC5` (missing_input: LL_TTM), `BPAN4` (no_quote), `ITUB3` (missing_input: LL_TTM), `ITUB4` (missing_input: LL_TTM), `SANB11` (missing_input: LL_TTM), `SANB3` (missing_input: LL_TTM), `SANB4` (missing_input: LL_TTM), `SULA11` (no_quote).

</details>

<details>
<summary>PEG RATIO, não financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `INTB3` | 0,26 | -0,45 | 273,08% |
| `LIGT3` | 0,24 | 0,55 | 129,17% |
| `EMAE4` | 0,04 | 0,06 | 50,00% |
| `TIMS3` | 1,06 | 0,54 | 49,06% |
| `ORVR3` | 0,56 | 0,40 | 28,57% |
| `VIVA3` | -0,68 | -0,84 | 23,53% |
| `EGIE3` | 0,34 | 0,27 | 20,59% |
| `MOVI3` | 0,22 | 0,18 | 18,18% |
| `GMAT3` | 0,21 | 0,18 | 14,29% |
| `VAMO3` | 21,72 | 18,95 | 12,75% |

Sem valor reconstruído: `AALR3` (non_positive_base), `AESB3` (no_quote), `ALPK3` (non_positive_base), `AMAR3` (non_positive_base), `AMBP3` (stale_filing), `AURE3` (non_positive_base), `AXIA6` (no_quote), `AZEV3` (non_positive_base), `AZEV4` (non_positive_base), `AZUL53` (no_quote), `AZUL54` (no_quote), `BEEF3` (non_positive_base), `BHIA3` (non_positive_base), `BIOM3` (non_positive_base), `BRFS3` (no_quote), `BRKM3` (non_positive_base), `BRKM5` (non_positive_base), `CLSA3` (no_quote), `CPLE5` (no_quote), `CPLE6` (no_quote), `CRFB3` (no_quote), `CRPG5` (non_positive_base), `CSAN3` (non_positive_base), `CSNA3` (non_positive_base), `CVCB3` (non_positive_base), `DASA3` (non_positive_base), `ELMD3` (no_quote), `ENAT3` (no_quote), `EQPA3` (missing_input: LL), `GPIV33` (no_cvm_code), `HAPV3` (non_positive_base), `HBSA3` (non_positive_base), `JALL3` (non_positive_base), `JBSS3` (no_quote), `LJQQ3` (non_positive_base), `MATD3` (non_positive_base), `MLAS3` (non_positive_base), `MRVE3` (non_positive_base), `NEOE3` (no_quote), `NGRD3` (non_positive_base), `NTCO3` (no_quote), `ONCO3` (non_positive_base), `OPCT3` (non_positive_base), `PCAR3` (non_positive_base), `PETZ3` (no_quote), `PORT3` (no_quote), `PTBL3` (non_positive_base), `RAIL3` (non_positive_base), `RAIZ4` (non_positive_base), `SCAR3` (non_positive_base), `SIMH3` (non_positive_base), `SOMA3` (no_quote), `STBP3` (no_quote), `SUZB3` (non_positive_base), `TUPY3` (non_positive_base), `VSTE3` (non_positive_base), `VVEO3` (non_positive_base), `ZAMP3` (no_quote).

</details>

<details>
<summary>PEG RATIO, financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `BBAS3` | -0,20 | -0,22 | 10,00% |
| `ABCB4` | 1,12 | 1,12 | 0,00% |
| `BAZA3` | -0,13 | -0,13 | 0,00% |
| `BBDC3` | 0,38 | 0,38 | 0,00% |
| `BBDC4` | 0,43 | 0,43 | 0,00% |
| `BBSE3` | 2,66 | 2,66 | 0,00% |
| `BEES3` | -0,30 | -0,30 | 0,00% |
| `BMEB3` | 0,14 | 0,14 | 0,00% |
| `BMEB4` | 0,17 | 0,17 | 0,00% |
| `BRBI11` | -0,74 | -0,74 | 0,00% |

Sem valor reconstruído: `APER3` (no_quote), `BMGB4` (missing_input: LL), `BPAC11` (missing_input: LL), `BPAC3` (missing_input: LL), `BPAC5` (missing_input: LL), `BPAN4` (no_quote), `ITUB3` (missing_input: LL), `ITUB4` (missing_input: LL), `PINE3` (short_history), `PINE4` (short_history), `SANB11` (missing_input: LL), `SANB3` (missing_input: LL), `SANB4` (missing_input: LL), `SULA11` (no_quote).

</details>

<details>
<summary>DY, não financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `HBRE3` | 52,50 | 21,87 | 30,63 p.p. |
| `RIAA3` | 31,29 | 41,94 | 10,65 p.p. |
| `MDNE3` | 14,37 | 7,24 | 7,13 p.p. |
| `VLID3` | 9,90 | 3,27 | 6,63 p.p. |
| `EQPA3` | 2,53 | 8,49 | 5,96 p.p. |
| `HBOR3` | 1,05 | 6,94 | 5,89 p.p. |
| `PFRM3` | 4,63 | 9,14 | 4,51 p.p. |
| `COCE5` | 0,00 | 4,36 | 4,36 p.p. |
| `CLSC4` | 0,36 | 4,48 | 4,12 p.p. |
| `TGMA3` | 10,84 | 14,87 | 4,03 p.p. |

Sem valor reconstruído: `AESB3` (no_quote), `AXIA6` (no_quote), `AZUL53` (no_quote), `AZUL54` (no_quote), `BRFS3` (no_quote), `CLSA3` (no_quote), `CPLE5` (no_quote), `CPLE6` (no_quote), `CRFB3` (no_quote), `ELMD3` (no_quote), `ENAT3` (no_quote), `GPIV33` (no_cvm_code), `JBSS3` (no_quote), `NEOE3` (no_quote), `NTCO3` (no_quote), `PETZ3` (no_quote), `PORT3` (no_quote), `SOMA3` (no_quote), `STBP3` (no_quote), `ZAMP3` (no_quote).

</details>

<details>
<summary>DY, financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `BMEB3` | 6,28 | 3,36 | 2,92 p.p. |
| `BMEB4` | 5,87 | 3,14 | 2,73 p.p. |
| `BEES3` | 7,81 | 8,88 | 1,07 p.p. |
| `QUAL3` | 1,12 | 0,56 | 0,56 p.p. |
| `BPAC5` | 2,65 | 2,32 | 0,33 p.p. |
| `BPAC11` | 2,63 | 2,31 | 0,32 p.p. |
| `BPAC3` | 2,13 | 1,87 | 0,26 p.p. |
| `ITUB4` | 8,70 | 8,47 | 0,23 p.p. |
| `ITUB3` | 8,12 | 7,90 | 0,22 p.p. |
| `ITSA3` | 9,87 | 9,77 | 0,10 p.p. |

Sem valor reconstruído: `APER3` (no_quote), `BPAN4` (no_quote), `SULA11` (no_quote).

</details>

<details>
<summary>PRECO, não financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `AALR3` | 3,04 | 3,04 | 0,00% |
| `ABEV3` | 15,48 | 15,48 | 0,00% |
| `AGRO3` | 18,07 | 18,07 | 0,00% |
| `ALLD3` | 4,60 | 4,60 | 0,00% |
| `ALOS3` | 26,94 | 26,94 | 0,00% |
| `ALPA3` | 9,87 | 9,87 | 0,00% |
| `ALPA4` | 13,53 | 13,53 | 0,00% |
| `ALPK3` | 4,35 | 4,35 | 0,00% |
| `ALUP11` | 32,06 | 32,06 | 0,00% |
| `ALUP3` | 11,33 | 11,33 | 0,00% |

Sem valor reconstruído: `AESB3` (no_quote), `AXIA6` (no_quote), `AZUL53` (no_quote), `AZUL54` (no_quote), `BRFS3` (no_quote), `CLSA3` (no_quote), `CPLE5` (no_quote), `CPLE6` (no_quote), `CRFB3` (no_quote), `ELMD3` (no_quote), `ENAT3` (no_quote), `GPIV33` (no_cvm_code), `JBSS3` (no_quote), `NEOE3` (no_quote), `NTCO3` (no_quote), `PETZ3` (no_quote), `PORT3` (no_quote), `SOMA3` (no_quote), `STBP3` (no_quote), `ZAMP3` (no_quote).

</details>

<details>
<summary>PRECO, financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `ABCB4` | 23,53 | 23,53 | 0,00% |
| `BAZA3` | 63,23 | 63,23 | 0,00% |
| `BBAS3` | 20,06 | 20,06 | 0,00% |
| `BBDC3` | 15,15 | 15,15 | 0,00% |
| `BBDC4` | 17,31 | 17,31 | 0,00% |
| `BBSE3` | 38,38 | 38,38 | 0,00% |
| `BEES3` | 8,78 | 8,78 | 0,00% |
| `BMEB3` | 50,86 | 50,86 | 0,00% |
| `BMEB4` | 59,80 | 59,80 | 0,00% |
| `BMGB4` | 5,20 | 5,20 | 0,00% |

Sem valor reconstruído: `APER3` (no_quote), `BPAN4` (no_quote), `SULA11` (no_quote).

</details>

<details>
<summary>VALOR DE MERCADO, não financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `VVEO3` | 206605189,12 | 2066051891,20 | 900,00% |
| `TFCO4` | 2238515001,92 | 13773493331,79 | 515,30% |
| `VTRU3` | 481531784,47 | 1705331559,88 | 254,15% |
| `AZEV3` | 268547946,85 | 715590439,05 | 166,47% |
| `AZEV4` | 268547946,85 | 715590439,05 | 166,47% |
| `LIGT3` | 4990409763,10 | 11549215044,00 | 131,43% |
| `SHUL4` | 5056312514,12 | 1547432797,40 | 69,40% |
| `COCE5` | 2063151133,37 | 2759315027,57 | 33,74% |
| `ORVR3` | 9317440252,85 | 6522216950,00 | 30,00% |
| `SIMH3` | 4109665408,64 | 3004656209,28 | 26,89% |

Sem valor reconstruído: `AESB3` (no_quote), `AXIA6` (no_quote), `AZUL53` (no_quote), `AZUL54` (no_quote), `BRFS3` (no_quote), `CLSA3` (no_quote), `CPLE5` (no_quote), `CPLE6` (no_quote), `CRFB3` (no_quote), `ELMD3` (no_quote), `ENAT3` (no_quote), `GPIV33` (no_cvm_code), `JBSS3` (no_quote), `NEOE3` (no_quote), `NTCO3` (no_quote), `PETZ3` (no_quote), `PORT3` (no_quote), `SOMA3` (no_quote), `STBP3` (no_quote), `ZAMP3` (no_quote).

</details>

<details>
<summary>VALOR DE MERCADO, financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `BPAC11` | 209711040484,02 | 240235321221,08 | 14,56% |
| `BPAC3` | 209711040484,02 | 240235321221,08 | 14,56% |
| `BPAC5` | 209711040484,02 | 240235321221,08 | 14,56% |
| `BMGB4` | 3371075219,20 | 3115259128,80 | 7,59% |
| `PINE3` | 2965720326,00 | 3038948907,90 | 2,47% |
| `PINE4` | 2965720326,00 | 3038948907,90 | 2,47% |
| `BMEB3` | 6647539892,20 | 6626690180,00 | 0,31% |
| `BMEB4` | 6647539892,20 | 6626690180,00 | 0,31% |
| `SANB11` | 109741001931,39 | 109587559090,00 | 0,14% |
| `SANB3` | 109741001931,39 | 109587559090,00 | 0,14% |

Sem valor reconstruído: `APER3` (no_quote), `BPAN4` (no_quote), `BRBI11` (no_quote), `SULA11` (no_quote).

</details>

<details>
<summary>LIQUIDEZ MEDIA DIARIA, não financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `NGRD3` | 220373,80 | 1186237,48 | 438,28% |
| `MTSA4` | 63382,48 | 55671,48 | 12,17% |
| `CEBR3` | 233757,48 | 257492,29 | 10,15% |
| `AMBP3` | 583141,19 | 640116,43 | 9,77% |
| `AZEV3` | 3728442,14 | 4046450,43 | 8,53% |
| `EMAE4` | 127553,71 | 138132,10 | 8,29% |
| `CEBR5` | 323088,24 | 346917,00 | 7,38% |
| `AZEV4` | 7134732,29 | 7609590,71 | 6,66% |
| `CLSC4` | 604634,00 | 640791,76 | 5,98% |
| `ALPA3` | 95663,29 | 101211,81 | 5,80% |

Sem valor reconstruído: `AESB3` (no_quote), `AXIA6` (no_quote), `AZUL53` (no_quote), `AZUL54` (no_quote), `BRFS3` (no_quote), `CLSA3` (no_quote), `CPLE5` (no_quote), `CPLE6` (no_quote), `CRFB3` (no_quote), `ELMD3` (no_quote), `ENAT3` (no_quote), `GPIV33` (no_cvm_code), `JBSS3` (no_quote), `NEOE3` (no_quote), `NTCO3` (no_quote), `PETZ3` (no_quote), `PORT3` (no_quote), `SOMA3` (no_quote), `STBP3` (no_quote), `ZAMP3` (no_quote).

</details>

<details>
<summary>LIQUIDEZ MEDIA DIARIA, financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `BAZA3` | 349372,76 | 367067,10 | 5,06% |
| `PINE3` | 126920,24 | 129981,10 | 2,41% |
| `BRSR3` | 69006,38 | 70105,19 | 1,59% |
| `BMEB3` | 291676,33 | 296042,10 | 1,50% |
| `BRBI11` | 3738115,05 | 3781457,57 | 1,16% |
| `BPAC5` | 284028,52 | 287191,05 | 1,11% |
| `BEES3` | 88902,24 | 89820,29 | 1,03% |
| `QUAL3` | 4370592,52 | 4410652,81 | 0,92% |
| `ITSA3` | 1408294,14 | 1419875,29 | 0,82% |
| `WIZC3` | 2186812,76 | 2202845,71 | 0,73% |

Sem valor reconstruído: `APER3` (no_quote), `BPAN4` (no_quote), `SULA11` (no_quote).

</details>

### 2026-11-05

| indicador | grupo | n | % na tolerância | mediana | p90 | veredito |
| --- | --- | --- | --- | --- | --- | --- |
| `LIQ. CORRENTE` | não financeiro | 239 | 79,50% | 0,00% | 10,13% | com ressalvas |
| `LIQ. CORRENTE` | não financeiro sem não classificados | 238 | 79,83% | 0,00% | 10,13% | com ressalvas |
| `LIQ. CORRENTE` | financeiro | 29 | 48,28% | 2,42% | 600,00% | não reconstruível |
| `DIVIDA LIQUIDA / EBIT` | não financeiro | 239 | 72,80% | 0,00% | 20,81% | com ressalvas |
| `DIVIDA LIQUIDA / EBIT` | não financeiro sem não classificados | 238 | 73,11% | 0,00% | 20,81% | com ressalvas |
| `DIVIDA LIQUIDA / EBIT` | financeiro | 9 | 44,44% | 12,78% | 366,67% | não reconstruível |
| `DIV. LIQ. / PATRI.` | não financeiro | 230 | 79,57% | 0,00% | 8,60% | com ressalvas |
| `DIV. LIQ. / PATRI.` | não financeiro sem não classificados | 229 | 79,91% | 0,00% | 8,60% | com ressalvas |
| `DIV. LIQ. / PATRI.` | financeiro | 9 | 55,56% | 0,31% | 350,00% | não reconstruível |
| `ROE` | não financeiro | 239 | 71,13% | 0,00 p.p. | 3,67 p.p. | com ressalvas |
| `ROE` | não financeiro sem não classificados | 238 | 71,43% | 0,00 p.p. | 3,67 p.p. | com ressalvas |
| `ROE` | financeiro | 35 | 14,29% | 12,37 p.p. | 258,67 p.p. | não reconstruível |
| `ROA` | não financeiro | 239 | 76,15% | 0,00 p.p. | 1,44 p.p. | com ressalvas |
| `ROA` | não financeiro sem não classificados | 238 | 76,47% | 0,00 p.p. | 1,44 p.p. | com ressalvas |
| `ROA` | financeiro | 35 | 62,86% | 0,02 p.p. | 0,12 p.p. | não reconstruível |
| `ROIC` | não financeiro | 239 | 43,93% | 0,57 p.p. | 5,29 p.p. | não reconstruível |
| `ROIC` | não financeiro sem não classificados | 238 | 44,12% | 0,57 p.p. | 5,29 p.p. | não reconstruível |
| `ROIC` | financeiro | 9 | 55,56% | 0,30 p.p. | 12,79 p.p. | não reconstruível |
| `MARG. LIQUIDA` | não financeiro | 237 | 72,57% | 0,00 p.p. | 3,37 p.p. | com ressalvas |
| `MARG. LIQUIDA` | não financeiro sem não classificados | 236 | 72,88% | 0,00 p.p. | 3,37 p.p. | com ressalvas |
| `MARG. LIQUIDA` | financeiro | 33 | 51,52% | 0,03 p.p. | 0,95 p.p. | não reconstruível |
| `MARGEM EBIT` | não financeiro | 237 | 75,11% | 0,00 p.p. | 1,86 p.p. | com ressalvas |
| `MARGEM EBIT` | não financeiro sem não classificados | 236 | 75,42% | 0,00 p.p. | 1,86 p.p. | com ressalvas |
| `MARGEM EBIT` | financeiro | 33 | 60,61% | 0,18 p.p. | 11,18 p.p. | não reconstruível |
| `P/L` | não financeiro | 239 | 68,20% | 0,00% | 34,49% | não reconstruível |
| `P/L` | não financeiro sem não classificados | 238 | 68,49% | 0,00% | 34,49% | não reconstruível |
| `P/L` | financeiro | 35 | 60,00% | 0,00% | 4,67% | não reconstruível |
| `P/VP` | não financeiro | 239 | 80,75% | 0,00% | 7,53% | com ressalvas |
| `P/VP` | não financeiro sem não classificados | 238 | 81,09% | 0,00% | 7,53% | com ressalvas |
| `P/VP` | financeiro | 35 | 20,00% | 87,20% | 1461,54% | não reconstruível |
| `EV/EBIT` | não financeiro | 239 | 69,46% | 0,00% | 23,48% | não reconstruível |
| `EV/EBIT` | não financeiro sem não classificados | 238 | 69,75% | 0,00% | 23,48% | não reconstruível |
| `EV/EBIT` | financeiro | 35 | 11,43% | 5,33% | 225,51% | não reconstruível |
| `PSR` | não financeiro | 236 | 80,93% | 0,00% | 6,49% | com ressalvas |
| `PSR` | não financeiro sem não classificados | 236 | 80,93% | 0,00% | 6,49% | com ressalvas |
| `PSR` | financeiro | 33 | 72,73% | 2,17% | 8,00% | com ressalvas |
| `CAGR RECEITAS 5 ANOS` | não financeiro | 233 | 91,42% | 0,00 p.p. | 0,00 p.p. | reconstruível |
| `CAGR RECEITAS 5 ANOS` | não financeiro sem não classificados | 233 | 91,42% | 0,00 p.p. | 0,00 p.p. | reconstruível |
| `CAGR RECEITAS 5 ANOS` | financeiro | 33 | 81,82% | 0,00 p.p. | 0,00 p.p. | com ressalvas |
| `CAGR LUCROS 5 ANOS` | não financeiro | 161 | 72,67% | 0,00 p.p. | 4,22 p.p. | com ressalvas |
| `CAGR LUCROS 5 ANOS` | não financeiro sem não classificados | 161 | 72,67% | 0,00 p.p. | 4,22 p.p. | com ressalvas |
| `CAGR LUCROS 5 ANOS` | financeiro | 32 | 50,00% | 0,00 p.p. | 2,27 p.p. | não reconstruível |
| `PEG RATIO` | não financeiro | 238 | 53,36% | 0,00% | 102,99% | não reconstruível |
| `PEG RATIO` | não financeiro sem não classificados | 237 | 53,59% | 0,00% | 102,99% | não reconstruível |
| `PEG RATIO` | financeiro | 35 | 40,00% | 0,00% | 30,77% | não reconstruível |
| `DY` | não financeiro | 239 | 76,99% | 0,00 p.p. | 2,27 p.p. | com ressalvas |
| `DY` | não financeiro sem não classificados | 238 | 77,31% | 0,00 p.p. | 2,27 p.p. | com ressalvas |
| `DY` | financeiro | 35 | 88,57% | 0,00 p.p. | 0,49 p.p. | com ressalvas |
| `PRECO` | não financeiro | 239 | 93,72% | 0,00% | 0,00% | reconstruível |
| `PRECO` | não financeiro sem não classificados | 238 | 94,12% | 0,00% | 0,00% | reconstruível |
| `PRECO` | financeiro | 35 | 94,29% | 0,00% | 0,00% | reconstruível |
| `VALOR DE MERCADO` | não financeiro | 239 | 84,52% | 0,00% | 4,70% | com ressalvas |
| `VALOR DE MERCADO` | não financeiro sem não classificados | 238 | 84,87% | 0,00% | 4,70% | com ressalvas |
| `VALOR DE MERCADO` | financeiro | 35 | 71,43% | 0,00% | 9,79% | com ressalvas |
| `LIQUIDEZ MEDIA DIARIA` | não financeiro | 239 | 91,21% | 0,28% | 1,78% | reconstruível |
| `LIQUIDEZ MEDIA DIARIA` | não financeiro sem não classificados | 238 | 91,60% | 0,28% | 1,78% | reconstruível |
| `LIQUIDEZ MEDIA DIARIA` | financeiro | 35 | 88,57% | 0,26% | 1,23% | com ressalvas |

Referências em branco no Status Invest (grupo não financeiro):

| indicador | em branco | com valor reconstruído | DY tratado como 0,00 |
| --- | --- | --- | --- |
| `DIV. LIQ. / PATRI.` | 9 | 7 | 0 |
| `MARG. LIQUIDA` | 2 | 0 | 0 |
| `MARGEM EBIT` | 2 | 0 | 0 |
| `PSR` | 3 | 0 | 0 |
| `CAGR RECEITAS 5 ANOS` | 6 | 1 | 0 |
| `CAGR LUCROS 5 ANOS` | 78 | 3 | 0 |
| `PEG RATIO` | 1 | 1 | 0 |
| `DY` | 0 | 0 | 66 |

<details>
<summary>LIQ. CORRENTE, não financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `VTRU3` | 4,48 | 2,56 | 42,86% |
| `PRIO3` | 1,34 | 1,83 | 36,57% |
| `SUZB3` | 3,38 | 2,54 | 24,85% |
| `DASA3` | 1,27 | 1,58 | 24,41% |
| `RECV3` | 2,91 | 2,28 | 21,65% |
| `CASH3` | 2,68 | 3,24 | 20,90% |
| `SMTO3` | 2,62 | 3,16 | 20,61% |
| `ALUP11` | 3,17 | 2,53 | 20,19% |
| `ALUP3` | 3,17 | 2,53 | 20,19% |
| `ALUP4` | 3,17 | 2,53 | 20,19% |

Sem valor reconstruído: `AALR3` (stale_filing), `AESB3` (no_quote), `AMBP3` (stale_filing), `AXIA6` (no_quote), `AZUL54` (no_quote), `BRFS3` (no_quote), `CPLE5` (no_quote), `CPLE6` (no_quote), `ENAT3` (no_quote), `GPIV33` (no_cvm_code), `JBSS3` (no_quote), `NEOE3` (no_quote), `NTCO3` (no_quote), `PETZ3` (no_quote), `PORT3` (no_quote), `STBP3` (no_quote), `ZAMP3` (no_quote).

</details>

<details>
<summary>LIQ. CORRENTE, financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `BMGB4` | 0,01 | 3,90 | 38900,00% |
| `BPAC11` | 0,01 | 0,07 | 600,00% |
| `BPAC3` | 0,01 | 0,07 | 600,00% |
| `BPAC5` | 0,01 | 0,07 | 600,00% |
| `ITUB3` | 0,12 | 0,32 | 166,67% |
| `ITUB4` | 0,12 | 0,32 | 166,67% |
| `ABCB4` | 0,19 | 0,40 | 110,53% |
| `BBSE3` | 1,15 | 1,83 | 59,13% |
| `SANB11` | 0,61 | 0,29 | 52,46% |
| `SANB3` | 0,61 | 0,29 | 52,46% |

Sem valor reconstruído: `BPAN4` (no_quote), `BSLI3` (stale_filing), `SULA11` (no_quote).

</details>

<details>
<summary>DIVIDA LIQUIDA / EBIT, não financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `EQPA3` | 0,32 | 2,61 | 715,62% |
| `CASH3` | 1,63 | 13,20 | 709,82% |
| `AGRO3` | -23,56 | 54,38 | 330,81% |
| `LJQQ3` | 30,40 | -26,50 | 187,17% |
| `MDNE3` | 0,37 | 0,86 | 132,43% |
| `TGMA3` | 0,08 | 0,01 | 87,50% |
| `BHIA3` | -14,38 | -2,55 | 82,27% |
| `GRND3` | -1,26 | -0,39 | 69,05% |
| `TIMS3` | -0,73 | -0,28 | 61,64% |
| `TUPY3` | -5,30 | -8,42 | 58,87% |

Sem valor reconstruído: `AALR3` (stale_filing), `AESB3` (no_quote), `AMBP3` (stale_filing), `AXIA6` (no_quote), `AZUL54` (no_quote), `BRFS3` (no_quote), `CPLE5` (no_quote), `CPLE6` (no_quote), `ENAT3` (no_quote), `GPIV33` (no_cvm_code), `JBSS3` (no_quote), `NEOE3` (no_quote), `NTCO3` (no_quote), `PETZ3` (no_quote), `PORT3` (no_quote), `STBP3` (no_quote), `ZAMP3` (no_quote).

</details>

<details>
<summary>DIVIDA LIQUIDA / EBIT, financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `WIZC3` | -0,03 | -0,14 | 366,67% |
| `BBSE3` | 0,66 | -0,14 | 121,21% |
| `BRBI11` | -264,41 | 42,70 | 116,15% |
| `PSSA3` | -4,59 | -5,63 | 22,66% |
| `QUAL3` | 2,75 | 2,67 | 2,91% |
| `CXSE3` | -0,26 | -0,26 | 0,00% |
| `ITSA3` | 0,21 | 0,21 | 0,00% |
| `ITSA4` | 0,21 | 0,21 | 0,00% |

Sem valor reconstruído: `SULA11` (no_quote).

</details>

<details>
<summary>DIV. LIQ. / PATRI., não financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `PCAR3` | 8,95 | -11,67 | 230,39% |
| `LJQQ3` | -0,16 | 0,20 | 225,00% |
| `VTRU3` | -0,54 | 0,29 | 153,70% |
| `BHIA3` | 2,15 | -0,69 | 132,09% |
| `MDNE3` | 0,09 | 0,20 | 122,22% |
| `TGMA3` | 0,03 | 0,00 | 100,00% |
| `TIMS3` | -0,39 | -0,08 | 79,49% |
| `GRND3` | -0,11 | -0,03 | 72,73% |
| `BRST3` | 0,66 | 1,07 | 62,12% |
| `CAML3` | 1,08 | 1,43 | 32,41% |

Sem valor reconstruído: `AESB3` (no_quote), `AMBP3` (stale_filing), `AXIA6` (no_quote), `BRFS3` (no_quote), `CPLE5` (no_quote), `CPLE6` (no_quote), `ENAT3` (no_quote), `GPIV33` (no_cvm_code), `JBSS3` (no_quote), `NEOE3` (no_quote), `NTCO3` (no_quote), `PETZ3` (no_quote), `PORT3` (no_quote), `STBP3` (no_quote), `ZAMP3` (no_quote).

</details>

<details>
<summary>DIV. LIQ. / PATRI., financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `WIZC3` | -0,02 | -0,09 | 350,00% |
| `BBSE3` | 0,60 | -0,05 | 108,33% |
| `PSSA3` | -0,90 | -0,97 | 7,78% |
| `BRBI11` | -1,59 | -1,60 | 0,63% |
| `CXSE3` | -0,09 | -0,09 | 0,00% |
| `ITSA3` | 0,04 | 0,04 | 0,00% |
| `ITSA4` | 0,04 | 0,04 | 0,00% |
| `QUAL3` | 0,60 | 0,60 | 0,00% |

Sem valor reconstruído: `SULA11` (no_quote).

</details>

<details>
<summary>ROE, não financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `PCAR3` | -520,29 | -932,09 | 411,80 p.p. |
| `EQPA3` | 366,78 | 29,89 | 336,89 p.p. |
| `BHIA3` | -49,14 | -159,69 | 110,55 p.p. |
| `RAIZ4` | -325,38 | -241,22 | 84,16 p.p. |
| `AZUL3` | -31,48 | 31,48 | 62,96 p.p. |
| `VTRU3` | -29,57 | 27,23 | 56,80 p.p. |
| `EMAE4` | -18,27 | -48,30 | 30,03 p.p. |
| `CASH3` | -23,45 | -0,47 | 22,98 p.p. |
| `AMER3` | 9,10 | -6,74 | 15,84 p.p. |
| `LJQQ3` | -67,62 | -82,54 | 14,92 p.p. |

Sem valor reconstruído: `AALR3` (stale_filing), `AESB3` (no_quote), `AMBP3` (stale_filing), `AXIA6` (no_quote), `AZUL54` (no_quote), `BRFS3` (no_quote), `CPLE5` (no_quote), `CPLE6` (no_quote), `ENAT3` (no_quote), `GPIV33` (no_cvm_code), `JBSS3` (no_quote), `NEOE3` (no_quote), `NTCO3` (no_quote), `PETZ3` (no_quote), `PORT3` (no_quote), `STBP3` (no_quote), `ZAMP3` (no_quote).

</details>

<details>
<summary>ROE, financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `PINE3` | 34,20 | 6143,90 | 6109,70 p.p. |
| `PINE4` | 34,20 | 6143,90 | 6109,70 p.p. |
| `BEES3` | 17,62 | 276,29 | 258,67 p.p. |
| `BEES4` | 17,62 | 276,29 | 258,67 p.p. |
| `BMEB3` | 27,16 | 219,84 | 192,68 p.p. |
| `BMEB4` | 27,16 | 219,84 | 192,68 p.p. |
| `BRSR3` | 13,87 | 72,84 | 58,97 p.p. |
| `BRSR6` | 13,87 | 72,84 | 58,97 p.p. |
| `BBAS3` | 8,27 | 30,41 | 22,14 p.p. |
| `BBSE3` | 84,38 | 68,17 | 16,21 p.p. |

Sem valor reconstruído: `ABCB4` (zero_divisor), `BMGB4` (missing_input: LL), `BPAC11` (missing_input: LL), `BPAC3` (missing_input: LL), `BPAC5` (missing_input: LL), `BPAN4` (no_quote), `BSLI3` (stale_filing), `ITUB3` (missing_input: LL), `ITUB4` (missing_input: LL), `SANB11` (missing_input: LL), `SANB3` (missing_input: LL), `SANB4` (missing_input: LL), `SULA11` (no_quote).

</details>

<details>
<summary>ROA, não financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `EQPA3` | 96,34 | 7,85 | 88,49 p.p. |
| `BHIA3` | -5,11 | -51,00 | 45,89 p.p. |
| `VTRU3` | -24,02 | 16,51 | 40,53 p.p. |
| `CASH3` | -18,87 | -0,40 | 18,47 p.p. |
| `EMAE4` | -6,52 | -17,51 | 10,99 p.p. |
| `PCAR3` | -13,68 | -19,91 | 6,23 p.p. |
| `EALT4` | 8,97 | 3,93 | 5,04 p.p. |
| `AMER3` | 2,60 | -1,92 | 4,52 p.p. |
| `TIMS3` | 3,43 | 7,56 | 4,13 p.p. |
| `AGRO3` | -2,43 | 1,44 | 3,87 p.p. |

Sem valor reconstruído: `AALR3` (stale_filing), `AESB3` (no_quote), `AMBP3` (stale_filing), `AXIA6` (no_quote), `AZUL54` (no_quote), `BRFS3` (no_quote), `CPLE5` (no_quote), `CPLE6` (no_quote), `ENAT3` (no_quote), `GPIV33` (no_cvm_code), `JBSS3` (no_quote), `NEOE3` (no_quote), `NTCO3` (no_quote), `PETZ3` (no_quote), `PORT3` (no_quote), `STBP3` (no_quote), `ZAMP3` (no_quote).

</details>

<details>
<summary>ROA, financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `BBSE3` | 41,91 | 44,25 | 2,34 p.p. |
| `PSSA3` | 6,25 | 5,98 | 0,27 p.p. |
| `BBAS3` | 0,59 | 0,47 | 0,12 p.p. |
| `BRBI11` | 0,86 | 0,78 | 0,08 p.p. |
| `ABCB4` | 1,50 | 1,46 | 0,04 p.p. |
| `PINE3` | 1,74 | 1,71 | 0,03 p.p. |
| `PINE4` | 1,74 | 1,71 | 0,03 p.p. |
| `BBDC3` | 1,06 | 1,04 | 0,02 p.p. |
| `BBDC4` | 1,06 | 1,04 | 0,02 p.p. |
| `BMEB3` | 2,19 | 2,17 | 0,02 p.p. |

Sem valor reconstruído: `BMGB4` (missing_input: LL), `BPAC11` (missing_input: LL), `BPAC3` (missing_input: LL), `BPAC5` (missing_input: LL), `BPAN4` (no_quote), `BSLI3` (stale_filing), `ITUB3` (missing_input: LL), `ITUB4` (missing_input: LL), `SANB11` (missing_input: LL), `SANB3` (missing_input: LL), `SANB4` (missing_input: LL), `SULA11` (no_quote).

</details>

<details>
<summary>ROIC, não financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `BHIA3` | -11,40 | 739,40 | 750,80 p.p. |
| `EQPA3` | 140,95 | 15,64 | 125,31 p.p. |
| `CASH3` | -32,33 | -1,80 | 30,53 p.p. |
| `VTRU3` | -29,52 | -1,87 | 27,65 p.p. |
| `BRKM3` | -40,22 | -15,88 | 24,34 p.p. |
| `BRKM5` | -40,22 | -15,88 | 24,34 p.p. |
| `EMAE4` | -81,96 | -96,94 | 14,98 p.p. |
| `TGMA3` | 12,62 | 22,70 | 10,08 p.p. |
| `AXIA3` | -7,20 | 2,01 | 9,21 p.p. |
| `AXIA7` | -7,20 | 2,01 | 9,21 p.p. |

Sem valor reconstruído: `AALR3` (stale_filing), `AESB3` (no_quote), `AMBP3` (stale_filing), `AXIA6` (no_quote), `AZUL54` (no_quote), `BRFS3` (no_quote), `CPLE5` (no_quote), `CPLE6` (no_quote), `ENAT3` (no_quote), `GPIV33` (no_cvm_code), `JBSS3` (no_quote), `NEOE3` (no_quote), `NTCO3` (no_quote), `PETZ3` (no_quote), `PORT3` (no_quote), `STBP3` (no_quote), `ZAMP3` (no_quote).

</details>

<details>
<summary>ROIC, financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `BBSE3` | 33,33 | 20,54 | 12,79 p.p. |
| `QUAL3` | 4,67 | 7,24 | 2,57 p.p. |
| `WIZC3` | 21,48 | 20,09 | 1,39 p.p. |
| `CXSE3` | 31,10 | 31,45 | 0,35 p.p. |
| `PSSA3` | 13,32 | 13,07 | 0,25 p.p. |
| `BRBI11` | -0,19 | -0,33 | 0,14 p.p. |
| `ITSA3` | 17,33 | 17,33 | 0,00 p.p. |
| `ITSA4` | 17,33 | 17,33 | 0,00 p.p. |

Sem valor reconstruído: `SULA11` (no_quote).

</details>

<details>
<summary>MARG. LIQUIDA, não financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `EQPA3` | 150,85 | 12,29 | 138,56 p.p. |
| `VTRU3` | -14,81 | 46,56 | 61,37 p.p. |
| `EMAE4` | -29,53 | -70,02 | 40,49 p.p. |
| `BHIA3` | -5,98 | -44,42 | 38,44 p.p. |
| `CASH3` | -20,31 | -0,51 | 19,80 p.p. |
| `AGRO3` | -9,73 | 5,73 | 15,46 p.p. |
| `AXIA3` | 27,05 | 41,74 | 14,69 p.p. |
| `AXIA7` | 27,05 | 41,74 | 14,69 p.p. |
| `TIMS3` | 7,22 | 15,79 | 8,57 p.p. |
| `ALUP11` | 27,68 | 19,73 | 7,95 p.p. |

Sem valor reconstruído: `AALR3` (stale_filing), `AESB3` (no_quote), `AMBP3` (stale_filing), `AXIA6` (no_quote), `AZUL54` (no_quote), `BRFS3` (no_quote), `CPLE5` (no_quote), `CPLE6` (no_quote), `ENAT3` (no_quote), `GPIV33` (no_cvm_code), `JBSS3` (no_quote), `NEOE3` (no_quote), `NTCO3` (no_quote), `PETZ3` (no_quote), `PORT3` (no_quote), `STBP3` (no_quote), `ZAMP3` (no_quote).

</details>

<details>
<summary>MARG. LIQUIDA, financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `BRBI11` | 47,73 | 43,83 | 3,90 p.p. |
| `WIZC3` | 15,57 | 16,77 | 1,20 p.p. |
| `BBAS3` | 4,66 | 3,71 | 0,95 p.p. |
| `ABCB4` | 12,60 | 13,16 | 0,56 p.p. |
| `BMEB3` | 8,94 | 8,51 | 0,43 p.p. |
| `BMEB4` | 8,94 | 8,51 | 0,43 p.p. |
| `PSSA3` | 8,64 | 8,50 | 0,14 p.p. |
| `BBDC3` | 8,59 | 8,49 | 0,10 p.p. |
| `BBDC4` | 8,59 | 8,49 | 0,10 p.p. |
| `QUAL3` | 1,93 | 1,90 | 0,03 p.p. |

Sem valor reconstruído: `BMGB4` (missing_input: LL), `BPAC11` (missing_input: LL), `BPAC3` (missing_input: LL), `BPAC5` (missing_input: LL), `BPAN4` (no_quote), `BSLI3` (stale_filing), `ITUB3` (missing_input: LL), `ITUB4` (missing_input: LL), `SANB11` (missing_input: LL), `SANB3` (missing_input: LL), `SANB4` (missing_input: LL), `SULA11` (no_quote).

</details>

<details>
<summary>MARGEM EBIT, não financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `EQPA3` | 157,82 | 19,27 | 138,55 p.p. |
| `VTRU3` | -7,31 | 28,25 | 35,56 p.p. |
| `EMAE4` | -85,26 | -119,58 | 34,32 p.p. |
| `CASH3` | -13,91 | -1,72 | 12,19 p.p. |
| `ALUP11` | 68,45 | 56,78 | 11,67 p.p. |
| `ALUP3` | 68,45 | 56,78 | 11,67 p.p. |
| `ALUP4` | 68,45 | 56,78 | 11,67 p.p. |
| `AXIA3` | 25,81 | 37,30 | 11,49 p.p. |
| `AXIA7` | 25,81 | 37,30 | 11,49 p.p. |
| `PRIO3` | 27,60 | 36,29 | 8,69 p.p. |

Sem valor reconstruído: `AALR3` (stale_filing), `AESB3` (no_quote), `AMBP3` (stale_filing), `AXIA6` (no_quote), `AZUL54` (no_quote), `BRFS3` (no_quote), `CPLE5` (no_quote), `CPLE6` (no_quote), `ENAT3` (no_quote), `GPIV33` (no_cvm_code), `JBSS3` (no_quote), `NEOE3` (no_quote), `NTCO3` (no_quote), `PETZ3` (no_quote), `PORT3` (no_quote), `STBP3` (no_quote), `ZAMP3` (no_quote).

</details>

<details>
<summary>MARGEM EBIT, financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `BPAC11` | 31,04 | 72,39 | 41,35 p.p. |
| `BPAC3` | 31,04 | 72,39 | 41,35 p.p. |
| `BPAC5` | 31,04 | 72,39 | 41,35 p.p. |
| `BRBI11` | 1,46 | -9,72 | 11,18 p.p. |
| `PINE3` | 15,86 | 14,17 | 1,69 p.p. |
| `PINE4` | 15,86 | 14,17 | 1,69 p.p. |
| `IRBR3` | 2,74 | 1,14 | 1,60 p.p. |
| `WIZC3` | 43,86 | 42,51 | 1,35 p.p. |
| `PSSA3` | 7,40 | 6,59 | 0,81 p.p. |
| `ABCB4` | 15,46 | 16,14 | 0,68 p.p. |

Sem valor reconstruído: `BPAN4` (no_quote), `BSLI3` (stale_filing), `SULA11` (no_quote).

</details>

<details>
<summary>P/L, não financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `CASH3` | -4,09 | -173,75 | 4148,17% |
| `EQPA3` | 0,57 | 6,97 | 1122,81% |
| `SIMH3` | -17,22 | 31,70 | 284,09% |
| `AGRO3` | -20,56 | 33,31 | 262,01% |
| `AMER3` | 2,22 | -3,00 | 235,14% |
| `VVEO3` | 15,38 | 38,60 | 150,98% |
| `LIGT3` | 1,65 | 3,83 | 132,12% |
| `VTRU3` | -6,09 | 1,68 | 127,59% |
| `MYPK3` | 15,83 | 35,92 | 126,91% |
| `CAML3` | 9,20 | 19,89 | 116,20% |

Sem valor reconstruído: `AALR3` (stale_filing), `AESB3` (no_quote), `AMBP3` (stale_filing), `AXIA6` (no_quote), `AZUL54` (no_quote), `BRFS3` (no_quote), `CPLE5` (no_quote), `CPLE6` (no_quote), `ENAT3` (no_quote), `GPIV33` (no_cvm_code), `JBSS3` (no_quote), `NEOE3` (no_quote), `NTCO3` (no_quote), `PETZ3` (no_quote), `PORT3` (no_quote), `STBP3` (no_quote), `ZAMP3` (no_quote).

</details>

<details>
<summary>P/L, financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `BBAS3` | 6,14 | 7,70 | 25,41% |
| `BRBI11` | 6,65 | 7,53 | 13,23% |
| `ABCB4` | 6,00 | 5,72 | 4,67% |
| `PINE4` | 4,29 | 4,48 | 4,43% |
| `PINE3` | 3,73 | 3,89 | 4,29% |
| `BBDC4` | 5,83 | 5,58 | 4,29% |
| `BBDC3` | 5,37 | 5,14 | 4,28% |
| `PSSA3` | 7,67 | 7,58 | 1,17% |
| `WIZC3` | 4,79 | 4,83 | 0,84% |
| `BMEB4` | 7,72 | 7,74 | 0,26% |

Sem valor reconstruído: `BMGB4` (missing_input: LL), `BPAC11` (missing_input: LL), `BPAC3` (missing_input: LL), `BPAC5` (missing_input: LL), `BPAN4` (no_quote), `BSLI3` (stale_filing), `ITUB3` (missing_input: LL), `ITUB4` (missing_input: LL), `SANB11` (missing_input: LL), `SANB3` (missing_input: LL), `SANB4` (missing_input: LL), `SULA11` (no_quote).

</details>

<details>
<summary>P/VP, não financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `PCAR3` | 2,88 | -3,67 | 227,43% |
| `VVEO3` | 0,48 | 1,20 | 150,00% |
| `BHIA3` | 0,22 | -0,08 | 136,36% |
| `LIGT3` | 0,43 | 0,99 | 130,23% |
| `VTRU3` | 1,80 | 0,46 | 74,44% |
| `RAIZ4` | -0,38 | -0,12 | 68,42% |
| `VIVA3` | 1,35 | 0,45 | 66,67% |
| `OPCT3` | 4,16 | 1,75 | 57,93% |
| `TIMS3` | 3,82 | 1,91 | 50,00% |
| `EALT4` | 0,68 | 0,41 | 39,71% |

Sem valor reconstruído: `AALR3` (stale_filing), `AESB3` (no_quote), `AMBP3` (stale_filing), `AXIA6` (no_quote), `AZUL54` (no_quote), `BRFS3` (no_quote), `CPLE5` (no_quote), `CPLE6` (no_quote), `ENAT3` (no_quote), `GPIV33` (no_cvm_code), `JBSS3` (no_quote), `NEOE3` (no_quote), `NTCO3` (no_quote), `PETZ3` (no_quote), `PORT3` (no_quote), `STBP3` (no_quote), `ZAMP3` (no_quote).

</details>

<details>
<summary>P/VP, financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `PINE4` | 1,47 | 275,15 | 18617,69% |
| `PINE3` | 1,28 | 239,17 | 18585,16% |
| `BEES3` | 1,23 | 19,30 | 1469,11% |
| `BEES4` | 1,30 | 20,30 | 1461,54% |
| `BMEB3` | 1,85 | 14,99 | 710,27% |
| `BMEB4` | 2,10 | 17,01 | 710,00% |
| `BRSR3` | 0,59 | 3,08 | 422,03% |
| `BRSR6` | 0,44 | 2,29 | 420,45% |
| `BBAS3` | 0,51 | 2,34 | 358,82% |
| `BNBR3` | 0,60 | 1,15 | 91,67% |

Sem valor reconstruído: `ABCB4` (zero_divisor), `BPAN4` (no_quote), `BSLI3` (stale_filing), `SULA11` (no_quote).

</details>

<details>
<summary>EV/EBIT, não financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `CASH3` | -4,33 | -37,88 | 774,83% |
| `EQPA3` | 0,86 | 7,06 | 720,93% |
| `TFCO4` | 7,40 | 42,72 | 477,30% |
| `AGRO3` | -58,52 | 138,78 | 337,15% |
| `VTRU3` | -8,65 | 4,52 | 152,25% |
| `BHIA3` | -15,82 | -2,85 | 81,98% |
| `TUPY3` | -10,99 | -18,34 | 66,88% |
| `VIVA3` | 5,97 | 2,52 | 57,79% |
| `LIGT3` | 17,61 | 25,79 | 46,45% |
| `OPCT3` | 14,12 | 8,18 | 42,07% |

Sem valor reconstruído: `AALR3` (stale_filing), `AESB3` (no_quote), `AMBP3` (stale_filing), `AXIA6` (no_quote), `AZUL54` (no_quote), `BRFS3` (no_quote), `CPLE5` (no_quote), `CPLE6` (no_quote), `ENAT3` (no_quote), `GPIV33` (no_cvm_code), `JBSS3` (no_quote), `NEOE3` (no_quote), `NTCO3` (no_quote), `PETZ3` (no_quote), `PORT3` (no_quote), `STBP3` (no_quote), `ZAMP3` (no_quote).

</details>

<details>
<summary>EV/EBIT, financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `BAZA3` | 24,74 | 113,87 | 360,27% |
| `IRBR3` | 38,30 | 124,67 | 225,51% |
| `ABCB4` | 4,89 | 11,20 | 129,04% |
| `BBSE3` | 8,04 | 15,38 | 91,29% |
| `WIZC3` | 1,67 | 1,76 | 5,39% |
| `PSSA3` | 4,37 | 4,14 | 5,26% |
| `QUAL3` | 4,12 | 4,00 | 2,91% |
| `CXSE3` | 12,60 | 12,60 | 0,00% |
| `ITSA3` | 7,80 | 7,80 | 0,00% |
| `ITSA4` | 7,80 | 7,80 | 0,00% |

Sem valor reconstruído: `BBAS3` (missing_input: DB), `BBDC3` (missing_input: DB), `BBDC4` (missing_input: DB), `BEES3` (missing_input: DB), `BEES4` (missing_input: DB), `BMEB3` (missing_input: DB), `BMEB4` (missing_input: DB), `BMGB4` (missing_input: DB), `BNBR3` (missing_input: DB), `BPAC11` (missing_input: DB), `BPAC3` (missing_input: DB), `BPAC5` (missing_input: DB), `BPAN4` (no_quote), `BRBI11` (no_quote), `BRSR3` (missing_input: DB), `BRSR6` (missing_input: DB), `BSLI3` (stale_filing), `ITUB3` (missing_input: DB), `ITUB4` (missing_input: DB), `PINE3` (missing_input: DB), `PINE4` (missing_input: DB), `SANB11` (missing_input: DB), `SANB3` (missing_input: DB), `SANB4` (missing_input: DB), `SULA11` (no_quote).

</details>

<details>
<summary>PSR, não financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `VVEO3` | 0,07 | 0,19 | 171,43% |
| `LIGT3` | 0,26 | 0,59 | 126,92% |
| `VIVA3` | 1,36 | 0,47 | 65,44% |
| `OPCT3` | 1,80 | 0,76 | 57,78% |
| `EALT4` | 0,47 | 0,28 | 40,43% |
| `BHIA3` | 0,03 | 0,02 | 33,33% |
| `PMAM3` | 1,84 | 1,29 | 29,89% |
| `BRST3` | 0,75 | 0,57 | 24,00% |
| `EMAE4` | 3,75 | 4,52 | 20,53% |
| `EGIE3` | 2,76 | 2,22 | 19,57% |

Sem valor reconstruído: `AALR3` (stale_filing), `AESB3` (no_quote), `AMBP3` (stale_filing), `AXIA6` (no_quote), `AZUL54` (no_quote), `BRFS3` (no_quote), `CPLE5` (no_quote), `CPLE6` (no_quote), `ENAT3` (no_quote), `JBSS3` (no_quote), `NEOE3` (no_quote), `NTCO3` (no_quote), `PETZ3` (no_quote), `PORT3` (no_quote), `STBP3` (no_quote), `ZAMP3` (no_quote).

</details>

<details>
<summary>PSR, financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `BPAC11` | 3,17 | 20,15 | 535,65% |
| `BPAC5` | 3,14 | 6,66 | 112,10% |
| `BPAC3` | 4,18 | 8,85 | 111,72% |
| `WIZC3` | 0,75 | 0,81 | 8,00% |
| `BBDC4` | 0,50 | 0,47 | 6,00% |
| `BMGB4` | 0,38 | 0,36 | 5,26% |
| `PINE3` | 0,40 | 0,42 | 5,00% |
| `BMEB3` | 0,61 | 0,58 | 4,92% |
| `BBDC3` | 0,46 | 0,44 | 4,35% |
| `PINE4` | 0,46 | 0,48 | 4,35% |

Sem valor reconstruído: `BPAN4` (no_quote), `BSLI3` (stale_filing), `SULA11` (no_quote).

</details>

<details>
<summary>CAGR RECEITAS 5 ANOS, não financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `MLAS3` | 6,65 | 4,98 | 1,67 p.p. |
| `BHIA3` | 1,39 | 0,20 | 1,19 p.p. |
| `VALE3` | 0,72 | 0,48 | 0,24 p.p. |
| `ABEV3` | 8,61 | 8,61 | 0,00 p.p. |
| `AGRO3` | -5,40 | -5,40 | 0,00 p.p. |
| `ALLD3` | 3,11 | 3,11 | 0,00 p.p. |
| `ALOS3` | 29,09 | 29,09 | 0,00 p.p. |
| `ALPA3` | 6,29 | 6,29 | 0,00 p.p. |
| `ALPA4` | 6,29 | 6,29 | 0,00 p.p. |
| `ALPK3` | 23,59 | 23,59 | 0,00 p.p. |

Sem valor reconstruído: `AALR3` (stale_filing), `AMBP3` (stale_filing), `AURE3` (short_history), `AXIA6` (no_quote), `AZUL54` (no_quote), `BRFS3` (no_quote), `BRST3` (short_history), `CPLE5` (no_quote), `CPLE6` (no_quote), `ENAT3` (no_quote), `JBSS3` (no_quote), `LAND3` (short_history), `NEOE3` (no_quote), `NTCO3` (no_quote), `PETZ3` (no_quote), `PORT3` (no_quote), `STBP3` (no_quote), `ZAMP3` (no_quote).

</details>

<details>
<summary>CAGR RECEITAS 5 ANOS, financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `BPAC11` | 16,36 | 8,59 | 7,77 p.p. |
| `BPAC3` | 16,36 | 8,59 | 7,77 p.p. |
| `BPAC5` | 16,36 | 8,59 | 7,77 p.p. |
| `ABCB4` | 28,96 | 28,96 | 0,00 p.p. |
| `BAZA3` | 46,47 | 46,47 | 0,00 p.p. |
| `BBAS3` | 26,49 | 26,49 | 0,00 p.p. |
| `BBDC3` | 22,38 | 22,38 | 0,00 p.p. |
| `BBDC4` | 22,38 | 22,38 | 0,00 p.p. |
| `BEES3` | 28,74 | 28,74 | 0,00 p.p. |
| `BEES4` | 28,74 | 28,74 | 0,00 p.p. |

Sem valor reconstruído: `BPAN4` (no_quote), `BSLI3` (stale_filing), `SULA11` (no_quote).

</details>

<details>
<summary>CAGR LUCROS 5 ANOS, não financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `EQPA3` | 93,04 | 16,91 | 76,13 p.p. |
| `EALT4` | 18,53 | 1,27 | 17,26 p.p. |
| `TIMS3` | 1,43 | 18,60 | 17,17 p.p. |
| `PRIO3` | 53,58 | 68,53 | 14,95 p.p. |
| `CAML3` | -19,43 | -30,94 | 11,51 p.p. |
| `AXIA3` | 13,72 | 24,05 | 10,33 p.p. |
| `AXIA7` | 13,72 | 24,05 | 10,33 p.p. |
| `VBBR3` | 5,30 | 12,72 | 7,42 p.p. |
| `ALUP11` | 8,04 | 1,81 | 6,23 p.p. |
| `ALUP3` | 8,04 | 1,81 | 6,23 p.p. |

Sem valor reconstruído: `AXIA6` (no_quote), `BRFS3` (no_quote), `BRST3` (short_history), `CPLE5` (no_quote), `CPLE6` (no_quote), `DXCO3` (non_positive_base), `ENAT3` (no_quote), `JBSS3` (no_quote), `NEOE3` (no_quote), `PETZ3` (no_quote), `PORT3` (no_quote), `SCAR3` (non_positive_base).

</details>

<details>
<summary>CAGR LUCROS 5 ANOS, financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `BRBI11` | 12,45 | 9,68 | 2,77 p.p. |
| `BBAS3` | 2,97 | 0,70 | 2,27 p.p. |
| `ABCB4` | 24,52 | 24,93 | 0,41 p.p. |
| `WIZC3` | 0,38 | 0,25 | 0,13 p.p. |
| `PSSA3` | 16,89 | 16,94 | 0,05 p.p. |
| `BBDC3` | 10,53 | 10,55 | 0,02 p.p. |
| `BBDC4` | 10,53 | 10,55 | 0,02 p.p. |
| `BAZA3` | 27,17 | 27,17 | 0,00 p.p. |
| `BBSE3` | 19,00 | 19,00 | 0,00 p.p. |
| `BEES3` | 14,84 | 14,84 | 0,00 p.p. |

Sem valor reconstruído: `BMEB3` (short_history), `BMEB4` (short_history), `BMGB4` (missing_input: LL_TTM), `BPAC11` (missing_input: LL_TTM), `BPAC3` (missing_input: LL_TTM), `BPAC5` (missing_input: LL_TTM), `BPAN4` (no_quote), `BSLI3` (stale_filing), `ITUB3` (missing_input: LL_TTM), `ITUB4` (missing_input: LL_TTM), `SANB11` (missing_input: LL_TTM), `SANB3` (missing_input: LL_TTM), `SANB4` (missing_input: LL_TTM), `SULA11` (no_quote).

</details>

<details>
<summary>PEG RATIO, não financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `CASH3` | 0,01 | 1,67 | 16600,00% |
| `SBSP3` | -0,39 | 16,24 | 4264,10% |
| `ISAE4` | -0,48 | 8,44 | 1858,33% |
| `ISAE3` | -0,65 | 11,40 | 1853,85% |
| `RECV3` | -0,25 | -4,47 | 1688,00% |
| `ALUP4` | 0,21 | -0,85 | 504,76% |
| `ALUP3` | 0,25 | -0,99 | 496,00% |
| `ALUP11` | 0,23 | -0,90 | 491,30% |
| `RADL3` | 1,97 | -5,83 | 395,94% |
| `MELK3` | -1,06 | 1,62 | 252,83% |

Sem valor reconstruído: `AALR3` (stale_filing), `AESB3` (no_quote), `AGRO3` (non_positive_base), `AMAR3` (non_positive_base), `AMBP3` (stale_filing), `AURE3` (non_positive_base), `AXIA3` (non_positive_base), `AXIA6` (no_quote), `AXIA7` (non_positive_base), `AZEV3` (non_positive_base), `AZEV4` (non_positive_base), `AZUL54` (no_quote), `BEEF3` (non_positive_base), `BHIA3` (non_positive_base), `BIOM3` (non_positive_base), `BRFS3` (no_quote), `BRKM3` (non_positive_base), `BRKM5` (non_positive_base), `CPLE5` (no_quote), `CPLE6` (no_quote), `CRPG5` (non_positive_base), `CSAN3` (non_positive_base), `CSNA3` (non_positive_base), `CVCB3` (non_positive_base), `DASA3` (non_positive_base), `ENAT3` (no_quote), `ENEV3` (non_positive_base), `EVEN3` (non_positive_base), `GPIV33` (no_cvm_code), `HAPV3` (non_positive_base), `HBSA3` (non_positive_base), `JALL3` (non_positive_base), `JBSS3` (no_quote), `LJQQ3` (non_positive_base), `MEAL3` (non_positive_base), `MLAS3` (non_positive_base), `MRVE3` (non_positive_base), `NEOE3` (no_quote), `NTCO3` (no_quote), `ONCO3` (non_positive_base), `OPCT3` (non_positive_base), `PCAR3` (non_positive_base), `PETZ3` (no_quote), `PMAM3` (non_positive_base), `PORT3` (no_quote), `POSI3` (non_positive_base), `PTBL3` (non_positive_base), `RAIZ4` (non_positive_base), `RNEW3` (non_positive_base), `RNEW4` (non_positive_base), `SCAR3` (non_positive_base), `SEQL3` (non_positive_base), `SIMH3` (non_positive_base), `STBP3` (no_quote), `TUPY3` (non_positive_base), `USIM3` (non_positive_base), `USIM5` (non_positive_base), `VVEO3` (non_positive_base), `ZAMP3` (no_quote).

</details>

<details>
<summary>PEG RATIO, financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `WIZC3` | 0,65 | 1,30 | 100,00% |
| `BBSE3` | 2,62 | 3,82 | 45,80% |
| `BMEB3` | 0,13 | 0,17 | 30,77% |
| `BRBI11` | -0,57 | -0,41 | 28,07% |
| `ABCB4` | 1,12 | 0,82 | 26,79% |
| `BMEB4` | 0,15 | 0,19 | 26,67% |
| `PSSA3` | 0,42 | 0,51 | 21,43% |
| `BBDC3` | 0,24 | 0,23 | 4,17% |
| `BBDC4` | 0,26 | 0,25 | 3,85% |
| `BAZA3` | -0,17 | -0,17 | 0,00% |

Sem valor reconstruído: `BMGB4` (missing_input: LL), `BPAC11` (missing_input: LL), `BPAC3` (missing_input: LL), `BPAC5` (missing_input: LL), `BPAN4` (no_quote), `BSLI3` (stale_filing), `ITUB3` (missing_input: LL), `ITUB4` (missing_input: LL), `PINE3` (short_history), `PINE4` (short_history), `SANB11` (missing_input: LL), `SANB3` (missing_input: LL), `SANB4` (missing_input: LL), `SULA11` (no_quote).

</details>

<details>
<summary>DY, não financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `MBRF3` | 18,41 | 0,00 | 18,41 p.p. |
| `JSLG3` | 48,30 | 34,82 | 13,48 p.p. |
| `RIAA3` | 40,52 | 53,98 | 13,46 p.p. |
| `MDNE3` | 22,31 | 11,24 | 11,07 p.p. |
| `VLID3` | 13,48 | 5,32 | 8,16 p.p. |
| `EQPA3` | 4,65 | 11,05 | 6,40 p.p. |
| `COCE5` | 0,00 | 5,30 | 5,30 p.p. |
| `PFRM3` | 2,42 | 7,13 | 4,71 p.p. |
| `PASS3` | 2,90 | 6,48 | 3,58 p.p. |
| `EQTL3` | 1,36 | 4,77 | 3,41 p.p. |

Sem valor reconstruído: `AESB3` (no_quote), `AXIA6` (no_quote), `AZUL54` (no_quote), `BRFS3` (no_quote), `CPLE5` (no_quote), `CPLE6` (no_quote), `ENAT3` (no_quote), `GPIV33` (no_cvm_code), `JBSS3` (no_quote), `NEOE3` (no_quote), `NTCO3` (no_quote), `PETZ3` (no_quote), `PORT3` (no_quote), `STBP3` (no_quote), `ZAMP3` (no_quote).

</details>

<details>
<summary>DY, financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `CXSE3` | 6,59 | 4,93 | 1,66 p.p. |
| `QUAL3` | 1,30 | 0,65 | 0,65 p.p. |
| `BEES3` | 8,94 | 9,44 | 0,50 p.p. |
| `BEES4` | 8,49 | 8,98 | 0,49 p.p. |
| `BBDC3` | 11,57 | 11,43 | 0,14 p.p. |
| `BBDC4` | 11,67 | 11,59 | 0,08 p.p. |
| `BAZA3` | 18,11 | 18,17 | 0,06 p.p. |
| `IRBR3` | 2,72 | 2,77 | 0,05 p.p. |
| `ITUB4` | 7,80 | 7,76 | 0,04 p.p. |
| `ITSA3` | 8,57 | 8,54 | 0,03 p.p. |

Sem valor reconstruído: `BPAN4` (no_quote), `SULA11` (no_quote).

</details>

<details>
<summary>PRECO, não financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `AALR3` | 2,20 | 2,20 | 0,00% |
| `ABEV3` | 15,48 | 15,48 | 0,00% |
| `AGRO3` | 18,02 | 18,02 | 0,00% |
| `ALLD3` | 5,26 | 5,26 | 0,00% |
| `ALOS3` | 25,05 | 25,05 | 0,00% |
| `ALPA3` | 10,63 | 10,63 | 0,00% |
| `ALPA4` | 12,66 | 12,66 | 0,00% |
| `ALPK3` | 3,66 | 3,66 | 0,00% |
| `ALUP11` | 32,15 | 32,15 | 0,00% |
| `ALUP3` | 11,75 | 11,75 | 0,00% |

Sem valor reconstruído: `AESB3` (no_quote), `AXIA6` (no_quote), `AZUL54` (no_quote), `BRFS3` (no_quote), `CPLE5` (no_quote), `CPLE6` (no_quote), `ENAT3` (no_quote), `GPIV33` (no_cvm_code), `JBSS3` (no_quote), `NEOE3` (no_quote), `NTCO3` (no_quote), `PETZ3` (no_quote), `PORT3` (no_quote), `STBP3` (no_quote), `ZAMP3` (no_quote).

</details>

<details>
<summary>PRECO, financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `ABCB4` | 22,74 | 22,74 | 0,00% |
| `BAZA3` | 65,96 | 65,96 | 0,00% |
| `BBAS3` | 16,49 | 16,49 | 0,00% |
| `BBDC3` | 12,69 | 12,69 | 0,00% |
| `BBDC4` | 13,77 | 13,77 | 0,00% |
| `BBSE3` | 37,90 | 37,90 | 0,00% |
| `BEES3` | 8,90 | 8,90 | 0,00% |
| `BEES4` | 9,36 | 9,36 | 0,00% |
| `BMEB3` | 46,90 | 46,90 | 0,00% |
| `BMEB4` | 53,20 | 53,20 | 0,00% |

Sem valor reconstruído: `BPAN4` (no_quote), `SULA11` (no_quote).

</details>

<details>
<summary>VALOR DE MERCADO, não financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `AZUL3` | 387468275,00 | 4238416126,00 | 993,87% |
| `TFCO4` | 1539840915,08 | 9474579601,95 | 515,30% |
| `VTRU3` | 464104198,25 | 1835409378,25 | 295,47% |
| `VVEO3` | 874700407,72 | 2195180134,40 | 150,96% |
| `LIGT3` | 3994493014,16 | 9239372035,20 | 131,30% |
| `VIVA3` | 4294075440,42 | 1492913911,86 | 65,23% |
| `OPCT3` | 4308559880,00 | 1815632750,32 | 57,86% |
| `RAIZ4` | 3105752845,20 | 1367279135,10 | 55,98% |
| `EALT4` | 263067786,77 | 156856210,00 | 40,37% |
| `PMAM3` | 1096570827,56 | 772715405,80 | 29,53% |

Sem valor reconstruído: `AESB3` (no_quote), `AXIA6` (no_quote), `AZUL54` (no_quote), `BRFS3` (no_quote), `CPLE5` (no_quote), `CPLE6` (no_quote), `ENAT3` (no_quote), `GPIV33` (no_cvm_code), `JBSS3` (no_quote), `NEOE3` (no_quote), `NTCO3` (no_quote), `PETZ3` (no_quote), `PORT3` (no_quote), `STBP3` (no_quote), `ZAMP3` (no_quote).

</details>

<details>
<summary>VALOR DE MERCADO, financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `BPAC11` | 209594339849,36 | 250468900667,22 | 19,50% |
| `BPAC3` | 209594339849,36 | 250468900667,22 | 19,50% |
| `BPAC5` | 209594339849,36 | 250468900667,22 | 19,50% |
| `PINE3` | 2299115573,76 | 2524277539,92 | 9,79% |
| `PINE4` | 2299115573,76 | 2524277539,92 | 9,79% |
| `BBDC3` | 148125541083,21 | 140123825182,08 | 5,40% |
| `BBDC4` | 148125541083,21 | 140123825182,08 | 5,40% |
| `ABCB4` | 6123081802,14 | 5926068581,94 | 3,22% |
| `SANB11` | 97255947731,47 | 96319324990,00 | 0,96% |
| `SANB3` | 97255947731,47 | 96319324990,00 | 0,96% |

Sem valor reconstruído: `BPAN4` (no_quote), `BRBI11` (no_quote), `SULA11` (no_quote).

</details>

<details>
<summary>LIQUIDEZ MEDIA DIARIA, não financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `WLMM4` | 82354,95 | 106665,57 | 29,52% |
| `AZEV3` | 8099128,19 | 9155109,86 | 13,04% |
| `CGAS5` | 149846,95 | 166245,71 | 10,94% |
| `EMAE4` | 188905,00 | 208800,57 | 10,53% |
| `AZEV4` | 3749853,62 | 4048759,52 | 7,97% |
| `TELB3` | 116076,38 | 123293,48 | 6,22% |
| `UNIP5` | 62478,68 | 59379,29 | 4,96% |
| `BHIA3` | 18613100,48 | 19467822,43 | 4,59% |
| `AZUL3` | 4280524,72 | 4084659,48 | 4,58% |
| `GOAU3` | 634462,05 | 661008,52 | 4,18% |

Sem valor reconstruído: `AESB3` (no_quote), `AXIA6` (no_quote), `AZUL54` (no_quote), `BRFS3` (no_quote), `CPLE5` (no_quote), `CPLE6` (no_quote), `ENAT3` (no_quote), `GPIV33` (no_cvm_code), `JBSS3` (no_quote), `NEOE3` (no_quote), `NTCO3` (no_quote), `PETZ3` (no_quote), `PORT3` (no_quote), `STBP3` (no_quote), `ZAMP3` (no_quote).

</details>

<details>
<summary>LIQUIDEZ MEDIA DIARIA, financeiro: 10 piores casos</summary>

| ticker | referência | reconstruído | diferença |
| --- | --- | --- | --- |
| `BNBR3` | 50200,67 | 42399,00 | 15,54% |
| `BMEB3` | 311857,75 | 353450,00 | 13,34% |
| `BAZA3` | 413686,10 | 424562,95 | 2,63% |
| `BEES3` | 124468,14 | 126003,33 | 1,23% |
| `BMEB4` | 2622792,62 | 2654844,00 | 1,22% |
| `PINE3` | 185589,86 | 187356,67 | 0,95% |
| `BEES4` | 50778,71 | 51235,05 | 0,90% |
| `ITSA3` | 4542258,81 | 4570183,14 | 0,61% |
| `SANB3` | 2552318,48 | 2564480,14 | 0,48% |
| `WIZC3` | 11249809,19 | 11303387,43 | 0,48% |

Sem valor reconstruído: `BPAN4` (no_quote), `SULA11` (no_quote).

</details>

## Dividendos

- Fonte do DY (`source`): `secondary`. Parte dos valores vem do Yahoo Finance.
- Janela: d − 365 dias < data-ex ≤ d.
- O Yahoo Finance não tem empresas que saíram da bolsa. O DY de uma empresa extinta não é medido por essa fonte, e o resultado deve ser lido com esse limite.

### Fonte por data

| data | B3 | Yahoo | sem fonte |
| --- | --- | --- | --- |
| 2026-08-07 | 238 | 8 | 0 |
| 2026-11-05 | 246 | 11 | 0 |

### Motivos de troca para o Yahoo

| data | b3_failed | b3_empty | b3_incomplete | b3_short_history |
| --- | --- | --- | --- | --- |
| 2026-08-07 | 0 | 5 | 0 | 3 |
| 2026-11-05 | 0 | 9 | 0 | 2 |

Tickers sem dividendos confirmados (`b3_empty_yahoo_failed`):

| data | tickers |
| --- | --- |
| 2026-08-07 | 0 |
| 2026-11-05 | 0 |

### Filtros e ajustes dos registros

- Registros da B3 (`dividends.b3_filters`): 16627 brutos; 482 repetidos, contados uma vez; 56 de `REST CAP DIN`, descartados.

Classes sem ISIN no ajuste de desdobramento (`dividends.split_adjust_no_isin`):

| data | classes |
| --- | --- |
| 2026-08-07 | 0 |
| 2026-11-05 | 0 |

Nomes com `/` na B3 (`dividends.slash_names`):

| tradingName | consulta | registros |
| --- | --- | --- |
| AMBEV S/A | `AMBEV SA` | 40 |
| CURY S/A | `CURY SA` | 20 |
| IMC S/A | `IMC SA` | 4 |
| KLABIN S/A | `KLABIN SA` | 219 |
| LIGHT S/A | `LIGHT SA` | 24 |
| OUROFINO S/A | `OUROFINO SA` | 25 |

## Cobertura histórica e esforço

### Primeiro ano por fonte

| fonte | primeiro ano |
| --- | --- |
| `COTAHIST` | ≤ 2005 (limite da sondagem) |
| `DFP` | 2010 |
| `FCA` | 2010 |
| `ITR` | 2011 |

### Primeiro ano por indicador

| indicador | primeiro ano |
| --- | --- |
| `LIQ. CORRENTE` | 2011 |
| `DIVIDA LIQUIDA / EBIT` | 2012 |
| `DIV. LIQ. / PATRI.` | 2011 |
| `ROE` | 2012 |
| `ROA` | 2012 |
| `ROIC` | 2012 |
| `MARG. LIQUIDA` | 2012 |
| `MARGEM EBIT` | 2012 |
| `P/L` | 2012 |
| `P/VP` | 2011 |
| `EV/EBIT` | 2012 |
| `PSR` | 2012 |
| `CAGR RECEITAS 5 ANOS` | 2015 |
| `CAGR LUCROS 5 ANOS` | 2015 |
| `PEG RATIO` | 2013 |
| `DY` | 1995 |
| `PRECO` | 2005 |
| `VALOR DE MERCADO` | 2011 |
| `LIQUIDEZ MEDIA DIARIA` | 2005 |

### Cobertura do DY

| fonte | tickers | primeira data bruta | primeiro ano | mediana do ano por ticker | retenção |
| --- | --- | --- | --- | --- | --- |
| `b3` | 252 | 1995-12-29 | 1995 | 2007 | histórico completo da empresa (GetListedCashDividends, todas as páginas) |
| `yahoo` | 12 | 2016-05-02 | 2016 | 2026 | a partir de 2000-01-01 (period1 da requisição) |

O DY começa em 1995, pela fonte `b3`.

### Esforço da reconstrução completa

- Total projetado: 187 s. Carga: 166 s. Compute: 20 s (0,11 s por data × 180 datas).
- Premissa: compute por data com o universo de hoje.

| tipo | segundos | bytes medidos | bytes completos | segundos projetados |
| --- | --- | --- | --- | --- |
| `CADASTRO` | 0,02 | 1494679 | 1494679 | 0,02 |
| `COTAHIST` | 15,58 | 179489589 | 616660205 | 53,54 |
| `DFP` | 12,29 | 90623836 | 175412433 | 23,79 |
| `FCA` | 0,03 | 6646755 | 6646755 | 0,03 |
| `ITR` | 17,99 | 86594608 | 428000310 | 88,92 |

### Anos lidos na reconstrução completa

| fonte | primeiro ano | último ano |
| --- | --- | --- |
| `COTAHIST` | 2011 | 2026 |
| `DFP` | 2010 | 2026 |
| `FCA` | 2010 | 2026 |
| `ITR` | 2011 | 2026 |

## Entradas

### Arquivos brutos

| arquivo | URL | sha256 | bytes |
| --- | --- | --- | --- |
| COTAHIST_A2025.ZIP | https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A2025.ZIP | `6cc0ad851c8f605ca30418acac48be81f1767f6ed05b4f563772d06ae448047d` | 89056788 |
| COTAHIST_A2026.ZIP | https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A2026.ZIP | `4df3625cbb2dcbf12308c0acaae4acff1887427b620854072bb56cf6d63d60b0` | 90432801 |
| b3_cash_3TENTOS_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiIzVEVOVE9TIn0= | `6d6ca4e79fcc2af2e3930f86e044a8765eea377ed24be31eaa000932a3a02ead` | 1696 |
| b3_cash_ABC_BRASIL_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJBQkMgQlJBU0lMIn0= | `73c7877465e42a546732fecbd7b06c549af8715c6896b126a1e954caaf5a3ed9` | 23305 |
| b3_cash_ACO_ALTONA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJBQ08gQUxUT05BIn0= | `8fd62bd6397406fb3b42ea50d1c1e9ae3524ea26f3f3e3ace75f3c53886a386e` | 36795 |
| b3_cash_ALLIAR_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJBTExJQVIifQ== | `d8f491696f24779fb703d14daea1647289438945634c7a492a2cac156c511236` | 734 |
| b3_cash_ALLIED_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJBTExJRUQifQ== | `4bd4aaea104ee02e8ce02ac2947f73b3107b978d401ef84aecc63c2f7488ce15` | 5014 |
| b3_cash_ALLOS_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJBTExPUyJ9 | `3c42567ae10249d43a775d9e582459cf0398d2a0d1f3a01396626aa7c3644371` | 13952 |
| b3_cash_ALPARGATAS_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJBTFBBUkdBVEFTIn0= | `7da8ef6addbbdddf9be51681a589bb58f59efe31b9e483945c05fbeb33b5bdde` | 39249 |
| b3_cash_ALPARGATAS_p2.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MiwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJBTFBBUkdBVEFTIn0= | `320c6cf89002bdea194affc4c14f38fd288fdb46f292109d6476a6733281b3af` | 21549 |
| b3_cash_ALUPAR_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJBTFVQQVIifQ== | `15f7c563b03d56cbf05f5fe2f1cf8b29c86a50ea57f95f3ceefffc98e939ae6a` | 34108 |
| b3_cash_AMAZONIA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJBTUFaT05JQSJ9 | `0111f8878adedc6e45087867d5da858b188e97197e2dc03ca78cbc48ba954eb1` | 25579 |
| b3_cash_AMBEV_SA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJBTUJFViBTQSJ9 | `fd31601c74a13eec7f248b132486c796134b7fb057e9a3c467dce45fce6c5d71` | 12880 |
| b3_cash_AMBIPAR_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJBTUJJUEFSIn0= | `e28989f27b385724ba488f643cb1a5f4a309b91fddabde7d12d52958e12e9fc1` | 1047 |
| b3_cash_AMERICANAS_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJBTUVSSUNBTkFTIn0= | `d5f676e8fea1b8bafaa65470528dffd6ed7e8fca1ff24b4aaf72b04885e8c201` | 2014 |
| b3_cash_ANIMA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJBTklNQSJ9 | `753ecba21ea7d8c0cb6a6a9da306276e0615ed4527a0ae7306c2666b49e8ad0e` | 2966 |
| b3_cash_ARMAC_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJBUk1BQyJ9 | `0b510c903b76bb433d08aed2c80fc000dda9cffacdc7b7619ad79382fb08117d` | 7304 |
| b3_cash_ASSAI_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJBU1NBSSJ9 | `c6efbcdfdb3878b25230980c2bd4c93145fba0f38e3edea2aa16a722dfd99a8e` | 2703 |
| b3_cash_AUREN_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJBVVJFTiJ9 | `81d26b6ef0a3df00cbbc522f3197c28a04e6a2e28c382b36d6530cae41d85dfa` | 2000 |
| b3_cash_AXIA_ENERGIA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJBWElBIEVORVJHSUEifQ== | `d74c20c435e6c756ba552e40ea956f06d377f6fc1eeb58d42a216bb0fbb3307f` | 39327 |
| b3_cash_AXIA_ENERGIA_p2.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MiwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJBWElBIEVORVJHSUEifQ== | `50b7f726fc4e253628b06bb747f8ef96c6194b186def423e261eba0db307347b` | 22032 |
| b3_cash_AZEVEDO_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJBWkVWRURPIn0= | `1401c98d5d05f218f1733042d29cf27cec2221954e229a0728d4126ceb51eafb` | 9255 |
| b3_cash_AZUL_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJBWlVMIn0= | `420b1d65d2f16949a60710187a020c7d68cac20ce47892d0d03f3287a59ad5d5` | 85 |
| b3_cash_AZZAS_2154_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJBWlpBUyAyMTU0In0= | `5f07915b189b46b56780cbabe6c91de5fcafd13ce295244b1a815defbf3b962e` | 20671 |
| b3_cash_B3_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJCMyJ9 | `9fc6af398b3c138954eef70e976eaf90fe88aee913230467421d717fe2b0f97a` | 37228 |
| b3_cash_BANCO_BMG_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJCQU5DTyBCTUcifQ== | `0d7345769cbde027fa3eabb566b147539051e2dbec096671465ee4da7ac3cb93` | 8669 |
| b3_cash_BANESTES_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJCQU5FU1RFUyJ9 | `d2bdf9897ee92d6eb205540d3c35b23f33979301ddbf40c094dbeebd1d784faf` | 39573 |
| b3_cash_BANESTES_p2.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MiwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJCQU5FU1RFUyJ9 | `8f0c0f662ee6a46369773d09e8263809a801699e894db4de1f2f527508585d0b` | 39595 |
| b3_cash_BANESTES_p3.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MywicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJCQU5FU1RFUyJ9 | `2015d4291330f056ea657351230b7420f6bd9a419f406af998ac24d018a60b2b` | 39420 |
| b3_cash_BANESTES_p4.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6NCwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJCQU5FU1RFUyJ9 | `820aeb1c33ff5ffd33f268331c431b9d18cc8754e4f3e9b244d6f00118aff860` | 39641 |
| b3_cash_BANESTES_p5.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6NSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJCQU5FU1RFUyJ9 | `a123523fde164f339ec3ddafc621b4d6383f8aaf484747a7c8551ffc932e8516` | 21723 |
| b3_cash_BANRISUL_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJCQU5SSVNVTCJ9 | `a7d59a0cc213a534d0c5595f11a2a814845bbb5c4b6c3f3d354bc4041355225b` | 39335 |
| b3_cash_BANRISUL_p2.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MiwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJCQU5SSVNVTCJ9 | `dfb5d91b04f781b0c31d110703963d66030c289a6e0f5593ce5e6f26315b61f8` | 39423 |
| b3_cash_BANRISUL_p3.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MywicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJCQU5SSVNVTCJ9 | `3c81915a20e3cae59f1a415ded78b1e78e9348b641240c4ab36646eac4f8124f` | 38716 |
| b3_cash_BBSEGURIDADE_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJCQlNFR1VSSURBREUifQ== | `53be448a73e4df66e0c67ea083e671d24b9b7cbe46d685963c7917d7c2d953e3` | 15969 |
| b3_cash_BEMOBI_TECH_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJCRU1PQkkgVEVDSCJ9 | `13354e64a7ce7eac744bef8063b9bc595ae4d3a6949a6a255f0caaa3dd23251d` | 3985 |
| b3_cash_BIOMM_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJCSU9NTSJ9 | `420b1d65d2f16949a60710187a020c7d68cac20ce47892d0d03f3287a59ad5d5` | 85 |
| b3_cash_BLAU_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJCTEFVIn0= | `400c82567283cfcb2c86c71281fcc48b31ecb0bc70bb852fb11635f9be8298c0` | 7982 |
| b3_cash_BOA_SAFRA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJCT0EgU0FGUkEifQ== | `3820e4109f0b416b8236e002f24228ca6abfd2337c24a3dc603722076c1c52b1` | 2709 |
| b3_cash_BRADESCO_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJCUkFERVNDTyJ9 | `db840651d875d3efa60397b49cb227b88a23927715da2dc3711fded51778b1ea` | 39543 |
| b3_cash_BRADESCO_p2.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MiwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJCUkFERVNDTyJ9 | `9c1a514715fba8aa2fbd0f24499743616252372141799a3cc6350edb523bcc05` | 39385 |
| b3_cash_BRADESCO_p3.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MywicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJCUkFERVNDTyJ9 | `24dcf3a2c2647f599d602a7f17e36e1dcc755f5e8456d966577e766972e92b3b` | 39182 |
| b3_cash_BRADESCO_p4.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6NCwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJCUkFERVNDTyJ9 | `399a8ed77ab87c41dff72664b6f6b27249fab2aa4adb88aeb5d97ab384db50c3` | 39207 |
| b3_cash_BRADESCO_p5.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6NSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJCUkFERVNDTyJ9 | `0191b84bcae2987633c0bf36b5feffe210573a5a714ab2441bbf3048f6ec5019` | 39546 |
| b3_cash_BRADESCO_p6.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6NiwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJCUkFERVNDTyJ9 | `7988c1f835d3ce8323064c1f4654217f880c1854992d14d791dbedeca91fd7c2` | 39261 |
| b3_cash_BRADESCO_p7.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6NywicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJCUkFERVNDTyJ9 | `5a729be93b6da96acb78c93fcc6536b4c4b0422c9ea5668b17cdf3dbc5749358` | 39509 |
| b3_cash_BRADESCO_p8.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6OCwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJCUkFERVNDTyJ9 | `2ba2547d0263451d3a89aa154670b852056ffddb322cf2a09f155b6e4818017a` | 22114 |
| b3_cash_BRADESPAR_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJCUkFERVNQQVIifQ== | `64571fdf9b8fac6f4159414efea5b54dfd8b60225a69b01ed18117f33c8c4fb9` | 39212 |
| b3_cash_BRADESPAR_p2.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MiwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJCUkFERVNQQVIifQ== | `fab098c7649e54c0b7a24d2832c8ac20077b503a4a70cc0dce6d54698f67fb59` | 6619 |
| b3_cash_BRADSAUDE_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJCUkFEU0FVREUifQ== | `802d98b74816fc6c4cdff3cf3902a693bbbe3d9e262a019c69215cf9cbaf7b0c` | 39171 |
| b3_cash_BRADSAUDE_p2.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MiwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJCUkFEU0FVREUifQ== | `d8cff35de23d8d9329c29301d1bbde913f004fd7f32012706481314d481e65f9` | 1705 |
| b3_cash_BRASILAGRO_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJCUkFTSUxBR1JPIn0= | `65a90f94ad6feb2008f8974c4679b0ee7d3e4a0b220fc948b45ffda2ea86f01c` | 5246 |
| b3_cash_BRASIL_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJCUkFTSUwifQ== | `bc1b44d552293394635c497640d4ffd1c5615b749ced692776d4aaaac4e62e1d` | 39583 |
| b3_cash_BRASIL_p2.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MiwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJCUkFTSUwifQ== | `255a6404dd5579571350e2949dedb05dfa9279af30341c9198ff2352f50c1721` | 39262 |
| b3_cash_BRASIL_p3.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MywicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJCUkFTSUwifQ== | `7f2a7e8aef3c6e096c8b0dbb1ae6a02f54bc9bbffa1322152f4d30958b3a8423` | 11865 |
| b3_cash_BRASKEM_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJCUkFTS0VNIn0= | `e010ca293c5a5e580308e3d03e123654087d2f2d6be955caa3c216c5623b4b74` | 14376 |
| b3_cash_BRAVA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJCUkFWQSJ9 | `b79a1d0fb4a3d5f671fedf7a7657142145e653db17131192bb9949a2cc207192` | 734 |
| b3_cash_BRB_BANCO_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJCUkIgQkFOQ08ifQ== | `4a7fd8c40a7ffd6c404c77c37304af726b10dd31faccc6e1fc7e233034b8d303` | 38925 |
| b3_cash_BRB_BANCO_p2.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MiwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJCUkIgQkFOQ08ifQ== | `f46729a7bfcb58cb9902be82a6f2ad303e12b344c4954f2146ea31b5feb472f1` | 24672 |
| b3_cash_BRISANET_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJCUklTQU5FVCJ9 | `9e698bae2d6f620f5140d0ba61879cd198bc8e6a8089dfefb32f77812e255713` | 1050 |
| b3_cash_BR_PARTNERS_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJCUiBQQVJUTkVSUyJ9 | `d53578dd3ea7e721d34a7400c873d56e6b05c59bf540dc70eb6de511c4ed8d1d` | 16614 |
| b3_cash_BTGP_BANCO_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJCVEdQIEJBTkNPIn0= | `7d985e5cd3988513a559e9fd661c7bf6753676cc7b56a2f1a58d76593b84c18c` | 34404 |
| b3_cash_CAIXA_SEGURI_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJDQUlYQSBTRUdVUkkifQ== | `8d38da0ee374ef4b86d27fd9bfa66fdeba58953a216399d8c9f24f0d0ac020b7` | 6466 |
| b3_cash_CAMBUCI_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJDQU1CVUNJIn0= | `7945f9a1f0f6420d5c41c630da5547120b7341cb873607b697af908cd16eae67` | 10082 |
| b3_cash_CAMIL_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJDQU1JTCJ9 | `767d98cf9df965a454b02c22d2fbb6d347019b58ae0f95d11edc88710f62a903` | 14788 |
| b3_cash_CASAS_BAHIA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJDQVNBUyBCQUhJQSJ9 | `6f1d129b93ba84518b77ae3460203c8333d420929e4f61ee23306c8612fe050a` | 14990 |
| b3_cash_CBA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJDQkEifQ== | `c61270b1e6efa39150a9cdb1ec04da92fd68e026dc0cad086868ab1320f69890` | 1706 |
| b3_cash_CEA_MODAS_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJDRUEgTU9EQVMifQ== | `7c536ed6728926ac99b57b0af971d7b6a7f7259544c9369f280f2258cdbe49ab` | 1390 |
| b3_cash_CEB_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJDRUIifQ== | `9b3d07eb81556e798cbbe452292583a04a68c3a0ad380df15795a48e42a12b93` | 29744 |
| b3_cash_CELESC_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJDRUxFU0MifQ== | `c41dc3b12ecb9948e0ac4e5fee4c3998d8a2a97d514056e0513c07421ff607b9` | 39345 |
| b3_cash_CELESC_p2.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MiwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJDRUxFU0MifQ== | `a3b0627ecab715a20b3407263947ae45f23fe24ccfeee495be3987d40d03fbff` | 28054 |
| b3_cash_CEMIG_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJDRU1JRyJ9 | `889a78bd0b028cb13617df19ae7316eb5a10cdf9dd538de0c1a64342beade93f` | 39470 |
| b3_cash_CEMIG_p2.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MiwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJDRU1JRyJ9 | `38610c3a955be71ac4cd2ec375b0ece04f95f0b7f16737d1ce27c6f73819fdce` | 39547 |
| b3_cash_CEMIG_p3.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MywicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJDRU1JRyJ9 | `c1a39e65f1492e1f08826ab0b082da3747e4a54a274a0e74336206a4cfb82f99` | 17924 |
| b3_cash_COELBA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJDT0VMQkEifQ== | `43a28a529a835768f8c5008d340432faafe65613fa5272a124504764bdda4c8b` | 39480 |
| b3_cash_COELBA_p2.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MiwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJDT0VMQkEifQ== | `4262f4643ce6d7c810c52a7cb40648042692b3c28a4bf79221e70a2fd72795d2` | 39141 |
| b3_cash_COELBA_p3.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MywicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJDT0VMQkEifQ== | `0969a793c059364677a0760756879e34a5b2f452b51f4e8b21fdf9eb2ae2ab3a` | 26178 |
| b3_cash_COELCE_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJDT0VMQ0UifQ== | `1e3632101c720f9fdd97e1b208b7e7461168d653cd52f1af5675c70bc34474e0` | 39251 |
| b3_cash_COELCE_p2.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MiwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJDT0VMQ0UifQ== | `4d83e09f274dc1d3dee44f05a25942660e5e32b9202a8e20960668cc4b42220b` | 17291 |
| b3_cash_COGNA_ON_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJDT0dOQSBPTiJ9 | `06531630a80facfd82e434ec636a19077718e467f172ab4a12e90c1a664c76aa` | 12653 |
| b3_cash_COMGAS_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJDT01HQVMifQ== | `2227b5cb9b37f13ddd93b08175a7a87af1fd1313fddab326fb27cb6623da32d6` | 39469 |
| b3_cash_COMGAS_p2.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MiwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJDT01HQVMifQ== | `58a7b1e3312e19b523e5dfcfe77d29ca34983186dc60f11f43901acbe59385ac` | 17162 |
| b3_cash_COMPASS_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJDT01QQVNTIn0= | `0af9da4a89b9826d26075cb078bf7944910956176bbecff11b728cca27196bc7` | 12697 |
| b3_cash_COPASA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJDT1BBU0EifQ== | `9e3967b91f87525143534eba18e009f6c618a0525026b3b8aca17324de56df5e` | 30901 |
| b3_cash_COPEL_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJDT1BFTCJ9 | `e3aa33a138b35ede009da0d16eaa8e70639a3676afd3ad4f602a8374aa0ae130` | 39028 |
| b3_cash_COPEL_p2.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MiwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJDT1BFTCJ9 | `155ef38d1f165cc97548572933241b7c2cb1eda506417fe5779184e06f3105c9` | 24233 |
| b3_cash_COSAN_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJDT1NBTiJ9 | `8eca8b32293ede054df9a43ac4392762a7d369846b4ecc789e33f4f517c17973` | 8476 |
| b3_cash_CPFL_ENERGIA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJDUEZMIEVORVJHSUEifQ== | `b662df6b8745ef2faa2c2a2e7fa619a836c221104fd368ce900c94509fd40e4e` | 12364 |
| b3_cash_CRISTAL_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJDUklTVEFMIn0= | `7dc9c166d45c31384274a4804bc02925587b218491b4cbde54a7d850810cc945` | 38924 |
| b3_cash_CRISTAL_p2.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MiwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJDUklTVEFMIn0= | `8c742d8d943d164493032827d32eb0f335afeecc63d167fd8a49a95d4a501c02` | 1333 |
| b3_cash_CRUZEIRO_EDU_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJDUlVaRUlSTyBFRFUifQ== | `a6bdb152160c32b26fef4d2fd74ade84d05b89b364e8269b434ac9b718226895` | 2673 |
| b3_cash_CSNMINERACAO_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJDU05NSU5FUkFDQU8ifQ== | `795748dc5baaac37861a759d69569f61c141756422388c868665ba4d7156e6b0` | 7917 |
| b3_cash_CSU_DIGITAL_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJDU1UgRElHSVRBTCJ9 | `55c7851709b2e830dcdf30af6277316ce1ce41ae1b4b555c1d7fe5d97ec13427` | 16113 |
| b3_cash_CURY_SA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJDVVJZIFNBIn0= | `52a33933d104f915465d992398cdb1a13d6a5f0bd06683c819949581c66225f6` | 6561 |
| b3_cash_CVC_BRASIL_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJDVkMgQlJBU0lMIn0= | `6da43354e26ca1302b558589e8e307092612108a0045cc21786eb77616f3849e` | 4316 |
| b3_cash_CYRELA_REALT_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJDWVJFTEEgUkVBTFQifQ== | `778ca2d357fb52f7e39b6297145a76c1e3c76fd41a63af1275456374744d1303` | 15439 |
| b3_cash_D1000VFARMA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJEMTAwMFZGQVJNQSJ9 | `420b1d65d2f16949a60710187a020c7d68cac20ce47892d0d03f3287a59ad5d5` | 85 |
| b3_cash_DASA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJEQVNBIn0= | `ec9d011d379789d1b1d044ac7d77e4e813817e3781349dd436bc830d47f29ec4` | 5629 |
| b3_cash_DESKTOP_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJERVNLVE9QIn0= | `0d9ffa76875a5501223930d40266cadfa5668fa46d54c9ed1c2ca9a28a9861f3` | 2351 |
| b3_cash_DEXCO_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJERVhDTyJ9 | `3d9372a7a2e5621a7545f7a6000ad9c38625148b53fbb3234597f468f07632a5` | 11869 |
| b3_cash_DEXXOS_PAR_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJERVhYT1MgUEFSIn0= | `07fce071cd5eb95bc2129dcfdf01c5b7d861299edbbd3c9798807b375dd20d3a` | 9651 |
| b3_cash_DIMED_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJESU1FRCJ9 | `9081fd7000a18da702a7f05f9cb6051f8f1c9f4cee0d0cd4aa48f9f59b053e9a` | 39400 |
| b3_cash_DIMED_p2.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MiwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJESU1FRCJ9 | `71672d32f7b079990fb2330dd9ea986b2acd43578d9d3d82eb92be94c0552bf1` | 31328 |
| b3_cash_DIRECIONAL_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJESVJFQ0lPTkFMIn0= | `c58a4c98729c278e9ce4b03023a7da77b137b8fb03425bd127869524ab34b348` | 7076 |
| b3_cash_ECORODOVIAS_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJFQ09ST0RPVklBUyJ9 | `e73d713aed61fa52d31d97f26ebfaec767cfac044a1089d9eada85b62c3aa3c3` | 8486 |
| b3_cash_EMAE_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJFTUFFIn0= | `63163bc263538d0ab94e3b0c25f46f2ebc67436892eadb3b4a91389c14069e29` | 16707 |
| b3_cash_EMBRAER_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJFTUJSQUVSIn0= | `a0e54aa8968283e058eeed98f3ff1fa463c8396b5c5d572210c14e049101d228` | 38873 |
| b3_cash_EMBRAER_p2.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MiwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJFTUJSQUVSIn0= | `ba8a9b816d9ad82d40b3383cf9f3de88227e989fce3e2150f9761a52a6736cf9` | 2011 |
| b3_cash_ENERGISA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJFTkVSR0lTQSJ9 | `a47a19cde9ddcd541dcaac58d834bb402782ad05c68e0e00af13ee4fd596a2bd` | 36788 |
| b3_cash_ENEVA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJFTkVWQSJ9 | `e4b9de496f3772aebf972c396e2a566fd4db46d0fa5a05d90ab9d9c43da98e4b` | 403 |
| b3_cash_ENGIE_BRASIL_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJFTkdJRSBCUkFTSUwifQ== | `df91fef33c6551af727d41c448e4dd31e44421f92cef05746fed7323fe47f905` | 33724 |
| b3_cash_EQTL_PARA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJFUVRMIFBBUkEifQ== | `ce10264fba74703d929343bfc564be8aecc85c69cab5c6de9e96a805357c2105` | 38374 |
| b3_cash_EQTL_PARA_p2.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MiwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJFUVRMIFBBUkEifQ== | `98126e38d45c3ebdc28e506ed17b4dff36e78316920b0193daf90b2153a39d25` | 10682 |
| b3_cash_EQUATORIAL_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJFUVVBVE9SSUFMIn0= | `423d08c2eff46f73218d375c7375d90b7c6fd46844c7958423b4ff26e81cd18b` | 13546 |
| b3_cash_ESPACOLASER_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJFU1BBQ09MQVNFUiJ9 | `f556bf52e25bb23c030e5a0cc638922f70dd51e3b9dedd9497cb8f0d5a5cd455` | 2357 |
| b3_cash_ESTAPAR_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJFU1RBUEFSIn0= | `8a7be229ce81cb54a94e3625dcbaeab095362b3d291b47f3d3ba4624515b4593` | 408 |
| b3_cash_ETERNIT_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJFVEVSTklUIn0= | `f3913ee3176ff6a0d2d942f1672e6b5a0809ce5de6be4dd518a05c19515eac42` | 38575 |
| b3_cash_ETERNIT_p2.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MiwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJFVEVSTklUIn0= | `d2539e5efa0a280007ea190db9085bd9e804b1a62e7ba43f146add5d490a7633` | 19996 |
| b3_cash_EUCATEX_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJFVUNBVEVYIn0= | `defe11d2ff13c106299a8e23fa0bf2abfbd597be93b4c1e745dd5cbbc3224714` | 13604 |
| b3_cash_EVEN_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJFVkVOIn0= | `13c55220f884ca1d4f9ab4428c1aa7a33928951de61af3aa1ba76f7c71509fda` | 7165 |
| b3_cash_EZTEC_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJFWlRFQyJ9 | `480ad489627527d0f1997c9d5a1dcf40f0a88188652bc66862b917f1e8d1d27a` | 11084 |
| b3_cash_FERBASA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJGRVJCQVNBIn0= | `f91f886d984399cc18adccb6c71745ef23e22261b791d3111dba2e46bbeed290` | 39314 |
| b3_cash_FERBASA_p2.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MiwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJGRVJCQVNBIn0= | `9e0abc69a702db4f725a25b38e0cdc226844073e0ee97e07436de1fb596b486a` | 12402 |
| b3_cash_FLEURY_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJGTEVVUlkifQ== | `bb8e6f35be94b43c2a1dcff87baeeef43732b32cc25dd8ddd3bc86017fdff9b9` | 15117 |
| b3_cash_FRAS_LE_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJGUkFTLUxFIn0= | `195f9b650d76fbf26bc1bf658e59c29892b7580d33ceb459ea1f41cc5c049d2d` | 5276 |
| b3_cash_GERDAU_MET_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJHRVJEQVUgTUVUIn0= | `0349b9cda80acaa57c6ab2b5b6c28ac641f02ceb4479c52f0833c1a43fa2bd81` | 38322 |
| b3_cash_GERDAU_MET_p2.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MiwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJHRVJEQVUgTUVUIn0= | `b50a790ca5d604dbb86e8c7a776743a9c46397e31198ab08f1c7e9c6e28e4941` | 30130 |
| b3_cash_GERDAU_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJHRVJEQVUifQ== | `e53a9b7dc92487f1d62d20f34d102aadf0f816a268448e5311a0dbdb2ab7965b` | 38262 |
| b3_cash_GERDAU_p2.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MiwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJHRVJEQVUifQ== | `0cbd6bfd5820ff17a42dd3c752c855107e732e54064bb21ac1ed9a61aae483ef` | 22419 |
| b3_cash_GPS_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJHUFMifQ== | `97eb0568b88603d874fffcffe433883c6579853a5d34755e6b1172c9d0798dd3` | 1707 |
| b3_cash_GRAZZIOTIN_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJHUkFaWklPVElOIn0= | `0ce088513b0ec632450c54093dded356993fe8b3db8e8daa7b7015a848addfa1` | 34560 |
| b3_cash_GRENDENE_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJHUkVOREVORSJ9 | `478966876b4dc55be0a43d3425f7cfc8151dbb4a938069c3fc492d98dd63be99` | 34858 |
| b3_cash_GRUPO_MATEUS_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJHUlVQTyBNQVRFVVMifQ== | `6521c4392b2078d89127fbf494352dfdb54c7a3d1de4e8c9ee01fbce0ae4570b` | 2715 |
| b3_cash_GRUPO_SBF_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJHUlVQTyBTQkYifQ== | `5732ed3a46f70ec3dcd7d796809ef78153978301b577cfea2de59a3015802c69` | 1708 |
| b3_cash_HAPVIDA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJIQVBWSURBIn0= | `9569aa18d238ed688b4943a3041d71fbcf2c152f213a82e28619e70d9d774523` | 3693 |
| b3_cash_HBR_REALTY_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJIQlIgUkVBTFRZIn0= | `aba050c017a038c7262275bde797d8194b64f863b003065d4614bb0c184dad68` | 733 |
| b3_cash_HELBOR_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJIRUxCT1IifQ== | `cf95756dda4372e73685bdef87f9c23204951c402c540b81e2da6d966522506a` | 5246 |
| b3_cash_HIDROVIAS_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJISURST1ZJQVMifQ== | `420b1d65d2f16949a60710187a020c7d68cac20ce47892d0d03f3287a59ad5d5` | 85 |
| b3_cash_HYPERA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJIWVBFUkEifQ== | `4b7a80420a9f52ad03e2420ac754ccf9a64d53c98a75bc1a0dcec74620fb7347` | 17269 |
| b3_cash_IGUATEMI_S_A_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJJR1VBVEVNSSBTLkEifQ== | `37d86a3fa8ac3c5789683233fbc8e61370cc85d7dc5a9a91659a2702b61795bd` | 34865 |
| b3_cash_IMC_SA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJJTUMgU0EifQ== | `db93001ea1ddaac753f07ee2d0a5eb2c489a9a47f8bd063f96818005600835f8` | 1376 |
| b3_cash_INTELBRAS_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJJTlRFTEJSQVMifQ== | `eb49d594f3b717ecb5c2718dba8fcf4a3b4d7b652e4544f95de87f5e34adcbca` | 6295 |
| b3_cash_IOCHP_MAXION_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJJT0NIUC1NQVhJT04ifQ== | `1963067f5a5287f4a7e00d7d65426692928239ed5086eb22aa7398c4b74d7806` | 14684 |
| b3_cash_IRANI_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJJUkFOSSJ9 | `3e7e2f0457a294c17d3904fb1b17180e3539acc67ac245e7627c4b97d199ae87` | 32695 |
| b3_cash_IRBBRASIL_RE_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJJUkJCUkFTSUwgUkUifQ== | `0f76445ab135815dc08bc963375adc24eacb222992f3f484190d7476e9ff1c80` | 7286 |
| b3_cash_ISA_ENERGIA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJJU0EgRU5FUkdJQSJ9 | `5804ccfd82aaefb7c69eeec52a15289f35e9886e0bcd6e75702a4e56c272db6c` | 38974 |
| b3_cash_ISA_ENERGIA_p2.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MiwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJJU0EgRU5FUkdJQSJ9 | `cfd76da0ba453339badd428e0c3fb90cec73782c76ebe825c3e567b9529074fd` | 33166 |
| b3_cash_ITAUSA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJJVEFVU0EifQ== | `fc56014aeb592ce52793481924380b945967d5d722d542e2d3348b4b2b8bdf11` | 38654 |
| b3_cash_ITAUSA_p2.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MiwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJJVEFVU0EifQ== | `c697b4b16627d11bb4ff4f6cad9166ab592cbcb6c3c8921febb3c5855f6a47a0` | 38680 |
| b3_cash_ITAUSA_p3.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MywicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJJVEFVU0EifQ== | `373831293292634c7c41b4222b27eef901ea2ffb658f40b8f28094187cf77c07` | 38608 |
| b3_cash_ITAUSA_p4.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6NCwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJJVEFVU0EifQ== | `267af8e8e9f62cfdd8361e11727a3a24f385a4903acd2760d9c53f2ab3cf8333` | 38641 |
| b3_cash_ITAUSA_p5.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6NSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJJVEFVU0EifQ== | `601b110c70c93a99500221dc5a7b6abf87271198bf4c13faca23aa1a56d87a8a` | 8389 |
| b3_cash_ITAUUNIBANCO_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJJVEFVVU5JQkFOQ08ifQ== | `f8d93aea5c9d85041581a4f089b102b9ae805df7f558822322cd9ad2a1584750` | 38850 |
| b3_cash_ITAUUNIBANCO_p2.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MiwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJJVEFVVU5JQkFOQ08ifQ== | `290c5bf41dee85bed66a4369ad123a44eb0a001fe56094dd6c4b3c83593ce1e3` | 38273 |
| b3_cash_ITAUUNIBANCO_p3.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MywicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJJVEFVVU5JQkFOQ08ifQ== | `ceee6617d568272ca69b1eaa2ff1a4ec4981cfe6c8436282c2d86b864129e6c1` | 38494 |
| b3_cash_ITAUUNIBANCO_p4.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6NCwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJJVEFVVU5JQkFOQ08ifQ== | `d7db348079a8adfa20221994f5b8b5c8b5a6938c52bec1b319669d98824d6fd9` | 39110 |
| b3_cash_ITAUUNIBANCO_p5.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6NSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJJVEFVVU5JQkFOQ08ifQ== | `d84653034ef41d281c6d29556a59b5638adbf471e0ed9755c5e955a2e7caf47a` | 38850 |
| b3_cash_ITAUUNIBANCO_p6.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6NiwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJJVEFVVU5JQkFOQ08ifQ== | `ba8e4fceddb5d193e1eb3030c3096e389b4fbbe4b9fea2381bb84ae096fe32cb` | 38273 |
| b3_cash_ITAUUNIBANCO_p7.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6NywicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJJVEFVVU5JQkFOQ08ifQ== | `816c07d2ae795df094ceaac41657115f8cb955af027609ada5357a460944db0e` | 38494 |
| b3_cash_ITAUUNIBANCO_p8.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6OCwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJJVEFVVU5JQkFOQ08ifQ== | `436d7de26a7af553117a691148aa380faacee49494ba006d53bc9aee08a82db3` | 39136 |
| b3_cash_JALLESMACHAD_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJKQUxMRVNNQUNIQUQifQ== | `44a3d47f4e64c4638ab682e1112b5241253d48d20f23c185c69a4ffd0c9444b0` | 2026 |
| b3_cash_JHSF_PART_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJKSFNGIFBBUlQifQ== | `c3b7509a9ecb9ce0bc20106344052d275cb5297bff3df8064edcc74e6e9e42b6` | 20780 |
| b3_cash_JSL_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJKU0wifQ== | `2be7e993f5f0bb4ed26463beaed77c1441fc22fdd1634ecda76ded2b5d7dfd4b` | 7894 |
| b3_cash_KEPLER_WEBER_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJLRVBMRVIgV0VCRVIifQ== | `b732bea2638a15548a57536db393f05c77dfb87b5ac1d8d779a6b34f5fe87b66` | 17467 |
| b3_cash_KLABIN_SA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJLTEFCSU4gU0EifQ== | `d83616dd93b49e5d6c03ba2bb1ddf008e791b172148ad828b40e885501dde942` | 38856 |
| b3_cash_KLABIN_SA_p2.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MiwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJLTEFCSU4gU0EifQ== | `e8574454ae77ba49facafe7805300180c0a8ece2cb3ec19f7a475473e4052eb3` | 32049 |
| b3_cash_LAVVI_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJMQVZWSSJ9 | `0b33f80aebe468911e24812db96c127c14f70b1f72dee83e7e34d5ea773eae07` | 6891 |
| b3_cash_LIGHT_SA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJMSUdIVCBTQSJ9 | `d2c90b9601117188a6da7220947fefa6a3f1755b137bb50ffb6bf8d32a23618d` | 7762 |
| b3_cash_LOCALIZA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJMT0NBTElaQSJ9 | `e9ed42ace366e114b663c30a00dd1c3304556291ac0a4ac1f96929538371adc6` | 34469 |
| b3_cash_LOG_COM_PROP_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJMT0cgQ09NIFBST1AifQ== | `1446b6dc3330b3cfa71b23d1013374bd1808490e2fc694d8a43d63af78d786ae` | 4631 |
| b3_cash_LOG_IN_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJMT0ctSU4ifQ== | `94c53b66bd973ad4c998c14f56a77513470d31423a0c4163326c4ae6c0cb491f` | 1054 |
| b3_cash_LOJAS_MARISA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJMT0pBUyBNQVJJU0EifQ== | `1ad44efb37d58c89766820c1f49e5d1452acb8dc5a27d7cb5f63569367183c30` | 3991 |
| b3_cash_LOJAS_RENNER_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJMT0pBUyBSRU5ORVIifQ== | `03dccd4333fed8554f9be6e205b90177fae83aa869a9303455d445e39d0c0e66` | 32145 |
| b3_cash_LOPES_BRASIL_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJMT1BFUyBCUkFTSUwifQ== | `b5094e3a388244f30897d349b4c888c8ac32d7356842681beff729c0db97d413` | 6224 |
| b3_cash_LWSA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJMV1NBIn0= | `27dfaee6d2f8f89030e9503648c9a922c25d9fc8516ee18850f4d34501ea0022` | 2023 |
| b3_cash_MAGAZ_LUIZA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJNQUdBWiBMVUlaQSJ9 | `4f3546b9395f5ed2bec7c4061a46986232bda75206f1e6af5a3520d1d8df9c4f` | 5630 |
| b3_cash_MARCOPOLO_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJNQVJDT1BPTE8ifQ== | `002a3c63aa5823623802e898f2006c32506398013aeca6886421695524dec9b1` | 38539 |
| b3_cash_MARCOPOLO_p2.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MiwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJNQVJDT1BPTE8ifQ== | `db026b84a98e9455b07ddf39146806c22db878b5a5cda86b92d716167d52b882` | 14230 |
| b3_cash_MARFRIG_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJNQVJGUklHIn0= | `cbeba2dbf975a6dd0aa33d15528b94324c58d45cd3e15189a2bbbdf94dd3df05` | 4289 |
| b3_cash_MATER_DEI_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJNQVRFUiBERUkifQ== | `cb3b210b4c1dfc1bbc9470d1e8cad3f17469f7cdd5abf163abba4fda6349c55a` | 1705 |
| b3_cash_MELIUZ_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJNRUxJVVoifQ== | `901721c04a75f64126b517054c6d3bed2f907d2b935d75f68bbe2dfe80b8ac61` | 1061 |
| b3_cash_MELNICK_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJNRUxOSUNLIn0= | `867496c4634a20c507a137d8bfaa6d52fbb0c2202e5adda9eb185336d7fe2782` | 5909 |
| b3_cash_MERCANTIL_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJNRVJDQU5USUwifQ== | `89ea55c16e236317d5b885e5967496102622021445b3f4a92aa5e377c2f98fb0` | 38468 |
| b3_cash_METAL_LEVE_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJNRVRBTCBMRVZFIn0= | `07bb5beaa4bd941d524e257de557daf6cd9760e8cde56e9123c90cedeb692204` | 33315 |
| b3_cash_METISA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJNRVRJU0EifQ== | `13803d704f79ba0680118042da069fcb2fe0c2b6e25751e8a453eeb5f1c95481` | 37662 |
| b3_cash_MILLS_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJNSUxMUyJ9 | `b8d3f28b97403f122f8dfd0ef36bc5da14e4e458d699752f3556df56f18230c9` | 11207 |
| b3_cash_MINERVA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJNSU5FUlZBIn0= | `516f1cab8ff0a9fc7c35ff948c74874cb025e33ee735da2296302f21db5b582d` | 4935 |
| b3_cash_MITRE_REALTY_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJNSVRSRSBSRUFMVFkifQ== | `6570e2b1808e69e2e12670b2a5638d7235cd88e8af82f4c8683c4e084063f87e` | 12720 |
| b3_cash_MOTIVA_SA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJNT1RJVkEgU0EifQ== | `c2042b3293ea67ea2fbcee34983905bf21fb0d548a629fd4f0fda7b7887e4390` | 16152 |
| b3_cash_MOURA_DUBEUX_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJNT1VSQSBEVUJFVVgifQ== | `6e218510058b81f6de54ba214b03e604ba210406d861f28aaaad9b685393aa11` | 3316 |
| b3_cash_MOVIDA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJNT1ZJREEifQ== | `cdf7529b2950ea16bce52b19a94efaf34bd974f9f835e92e1ce5fb45ade603e2` | 7943 |
| b3_cash_MRV_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJNUlYifQ== | `e7441e2d48d8dff088c3ea90e9107e38f30e71d100cdaf0174517b4a18802fdd` | 7805 |
| b3_cash_MULTILASER_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJNVUxUSUxBU0VSIn0= | `af23b6593894f5879f46be6dfd403626669f9ff347a5056f6c048786bce8d46e` | 730 |
| b3_cash_MULTIPLAN_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJNVUxUSVBMQU4ifQ== | `c27d27f804ac81c58900b783ab72d5acafb96694951e4cf217ab0f8f9ff0af7e` | 15228 |
| b3_cash_M_DIASBRANCO_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJNLkRJQVNCUkFOQ08ifQ== | `903f0c1ba2d1c4ecd23047b2867f4fb8a7e784f4a5aa909d9c9238d25bad2bd3` | 19041 |
| b3_cash_NATURA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJOQVRVUkEifQ== | `76e988e72ed70e2d8f12c5a4fb44684c5ca60c21909b15a24a1853ac081bb004` | 17954 |
| b3_cash_NEOGRID_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJORU9HUklEIn0= | `cbf5f1f33f50d1aade93ba8cca69f0cc382ba698ac84ae9418ff02bcbc313473` | 1367 |
| b3_cash_NORD_BRASIL_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJOT1JEIEJSQVNJTCJ9 | `5ed8c2ebc393ab66f375109d7cbf41a46a0a2ae2f6513bd1a3738386e1b49f98` | 39346 |
| b3_cash_NORD_BRASIL_p2.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MiwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJOT1JEIEJSQVNJTCJ9 | `096c2633cfefc4ef0f95b41c62af5b8dd387a40c8deb94c94a064f7043919a20` | 16101 |
| b3_cash_OCEANPACT_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJPQ0VBTlBBQ1QifQ== | `645208890ae8e3062929afd61ef1f94e30311c1c4e5b5abf2e42234937ab2a8e` | 409 |
| b3_cash_ONCOCLINICAS_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJPTkNPQ0xJTklDQVMifQ== | `420b1d65d2f16949a60710187a020c7d68cac20ce47892d0d03f3287a59ad5d5` | 85 |
| b3_cash_ORIZON_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJPUklaT04ifQ== | `420b1d65d2f16949a60710187a020c7d68cac20ce47892d0d03f3287a59ad5d5` | 85 |
| b3_cash_OUROFINO_SA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJPVVJPRklOTyBTQSJ9 | `24daf814af1981b125bb5a07add407743d3f7ee25216bfa5d53b500f6f087996` | 8202 |
| b3_cash_PAGUE_MENOS_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJQQUdVRSBNRU5PUyJ9 | `54a21f04edf6a5f004cba78e9e763a466b58ae5b66dfb43adc824278e17cf6a4` | 1400 |
| b3_cash_PARANAPANEMA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJQQVJBTkFQQU5FTUEifQ== | `65d4ca44c6a1ff45f7a3df8234a53536060dc3f250337301e54b7921291cbe25` | 728 |
| b3_cash_PETROBRAS_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJQRVRST0JSQVMifQ== | `cbcf452d3ff0a94f837dd074cc436190f82843e602d2dfc622f21218b2420bb7` | 38767 |
| b3_cash_PETROBRAS_p2.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MiwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJQRVRST0JSQVMifQ== | `3721b776aaf5aed45dd7de25f24824491b5dca5d5f3cde9f77f6305ac4be59eb` | 38808 |
| b3_cash_PETROBRAS_p3.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MywicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJQRVRST0JSQVMifQ== | `35f61de52dcc9e96419ed9dd4dbf6ee8bb17393cf4b754d343117806be2b899e` | 33135 |
| b3_cash_PETRORECSA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJQRVRST1JFQ1NBIn0= | `de8082aec0c2c8a0d197955a79ab2bd0c893da952b009d0f065091c173bc73bf` | 4299 |
| b3_cash_PINE_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJQSU5FIn0= | `e7c34733bd380775f71216aea28cc699cec8ad544cd8ea38597a769399b70809` | 38350 |
| b3_cash_PINE_p2.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MiwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJQSU5FIn0= | `a0104f4a7851ee317d4919d6dbbcc9e7f8516a492236bafaea574be51366c6c2` | 13119 |
| b3_cash_PLANOEPLANO_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJQTEFOT0VQTEFOTyJ9 | `0d84f86304576af9574a735140ac75d6b5df334608e93844e61c065b1f5cf89d` | 2028 |
| b3_cash_PORTOBELLO_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJQT1JUT0JFTExPIn0= | `0c7912567790c16f2d5e1ab10d8b71a44e209fb7a0edf259ea8cd710d55388cb` | 11332 |
| b3_cash_PORTO_SEGURO_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJQT1JUTyBTRUdVUk8ifQ== | `fce56e24c10f7744cd7c946278b9d737433b9889188b0cbb660dcf60441d4c42` | 23615 |
| b3_cash_POSITIVO_TEC_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJQT1NJVElWTyBURUMifQ== | `ce62311f39aecb99acfcd2f920f7016cd24e63a897d4022f4de31a48586fdf89` | 8804 |
| b3_cash_PRINER_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJQUklORVIifQ== | `29f940ceeff216f06efa23f7528d0644d5317c8ef4865aba6b17fd6f53bae094` | 2335 |
| b3_cash_PRIO_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJQUklPIn0= | `84fc4d3a38bb1ed0d29671ff2b46d2a022606550a9129e643f154793ca62517a` | 405 |
| b3_cash_PROFARMA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJQUk9GQVJNQSJ9 | `f3c45cd82c4633f90e9bf24b629168896c1688a9285c0a6e1bc3b7b5cd47be39` | 5954 |
| b3_cash_P_ACUCAR_CBD_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJQLkFDVUNBUi1DQkQifQ== | `63b4d498205ca93bb2216c2ca0331c000bb4637b1d068c60c8ecddf9c6dfa448` | 35152 |
| b3_cash_QUALICORP_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJRVUFMSUNPUlAifQ== | `018e44385ffccd5ea74998942df7e4a18d6cbbf46decb346b87cc29234e8c871` | 7208 |
| b3_cash_QUERO_QUERO_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJRVUVSTy1RVUVSTyJ9 | `accc88f9c33dae96b3b68778cd24f7bdfda801341db0a24fd54cbb9bafa6a4ed` | 1728 |
| b3_cash_RAIADROGASIL_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJSQUlBRFJPR0FTSUwifQ== | `62a1436f6eeaf0b3db6c3a5fb016576a8b2044ac3ce94d5f2d50a1aeadab4323` | 39120 |
| b3_cash_RAIADROGASIL_p2.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MiwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJSQUlBRFJPR0FTSUwifQ== | `1f156d3c96f55babbe3e4dcb4ce1ca9fea02ac7d7af182acdb18fe752dcaa4f8` | 3502 |
| b3_cash_RAIZEN_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJSQUlaRU4ifQ== | `83cea7b755d0db60eadee2de7444ddbf446b9c5b719b214bf06ae57f3781ed78` | 4002 |
| b3_cash_RANDON_PART_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJSQU5ET04gUEFSVCJ9 | `538dd331fca02e66ef278b5b7e07a8495da6bee0d17d6581befd26c912dac45f` | 38217 |
| b3_cash_REDE_D_OR_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJSRURFIEQgT1IifQ== | `f5315da62563a3f27c17e09621e0fea8afd776576de741e53c95980bbe1073cb` | 8336 |
| b3_cash_RENOVA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJSRU5PVkEifQ== | `420b1d65d2f16949a60710187a020c7d68cac20ce47892d0d03f3287a59ad5d5` | 85 |
| b3_cash_RIACHUELO_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJSSUFDSFVFTE8ifQ== | `c82a604106a24345b41e5c5729c4f3c518d91f87aebc22b9d77bdeb71df221ca` | 30876 |
| b3_cash_ROMI_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJST01JIn0= | `9b8e59ce356b984f11aed45fbbba2e3de8ff68682230f0b2d102d21f8e1c53bb` | 31164 |
| b3_cash_RUMO_S_A_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJSVU1PIFMuQS4ifQ== | `8d408da48972cdfb6843f2909b5d16cff524aa9437cc2a552e8ebed9b72cc805` | 11436 |
| b3_cash_SABESP_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJTQUJFU1AifQ== | `25bb2d65cbef80fc377f893000c8cb41ed3b3e52dcebe2c31a6ee73147b75f3c` | 15791 |
| b3_cash_SANEPAR_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJTQU5FUEFSIn0= | `0759acdadbc4b591fed2374aa3ac672782a869c54e342ba620637731f7e09832` | 39246 |
| b3_cash_SANEPAR_p2.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MiwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJTQU5FUEFSIn0= | `98492b3e3094dcd911cad48c4af78c93cacc54b550acbe8387c6ac95f09ffd36` | 2730 |
| b3_cash_SANTANDER_BR_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJTQU5UQU5ERVIgQlIifQ== | `fe0008ceef1467e7c87749e7248ebac5c1a1d373754f9e0d629222c93190f0b6` | 39443 |
| b3_cash_SANTANDER_BR_p2.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MiwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJTQU5UQU5ERVIgQlIifQ== | `b746f031ad85c971e030e15897720fccd2b60c7071c5062d2e091151b7894e8a` | 39463 |
| b3_cash_SANTANDER_BR_p3.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MywicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJTQU5UQU5ERVIgQlIifQ== | `ed14cd8aa80dd4dfef20e20ce03cd95e8b7eaf38dc80cd3e1165b83ad9597f03` | 20797 |
| b3_cash_SAO_CARLOS_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJTQU8gQ0FSTE9TIn0= | `14fb05ac1649b876e926ddb84e4b9c56c91d1e79c8eca24630a14d3f77653058` | 12431 |
| b3_cash_SAO_MARTINHO_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJTQU8gTUFSVElOSE8ifQ== | `066e88767029d0d2aac64db80742b69dcd75d167d486d5538b2548bccc20c023` | 9512 |
| b3_cash_SCHULZ_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJTQ0hVTFoifQ== | `af7543f5bad145ec966f3579bf8374fb0923da0cbfc0121bca10223b4bd470e9` | 38446 |
| b3_cash_SCHULZ_p2.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MiwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJTQ0hVTFoifQ== | `12d0ba2282488a3be00e5434ed0e33cf5b47c56e212da32f451d568959924d35` | 3655 |
| b3_cash_SEQUOIA_LOG_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJTRVFVT0lBIExPRyJ9 | `420b1d65d2f16949a60710187a020c7d68cac20ce47892d0d03f3287a59ad5d5` | 85 |
| b3_cash_SER_EDUCA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJTRVIgRURVQ0EifQ== | `1c9e848f9b9d96daa73cb28320a11bcf68048a7846958cbdddaea3fb528791f9` | 7524 |
| b3_cash_SID_NACIONAL_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJTSUQgTkFDSU9OQUwifQ== | `965cb716eb3dfdaa85c419837fb8542eb2dc1bb353af3da1a84bc9077804d892` | 22223 |
| b3_cash_SIMPAR_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJTSU1QQVIifQ== | `163190842f334819142731fa950a383a3774d982e169ed3a5af5a6646f9dc713` | 2682 |
| b3_cash_SLC_AGRICOLA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJTTEMgQUdSSUNPTEEifQ== | `00376daaaaddf1dd02cdba45c9cae2466b1dddb12122d0dcd8e2f9eabdad7705` | 8157 |
| b3_cash_SMART_FIT_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJTTUFSVCBGSVQifQ== | `5a5318ee105051b0f442354e302c527ad4a114122b6ab7b68674acbd7bef6fd9` | 4954 |
| b3_cash_SUZANO_S_A_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJTVVpBTk8gUy5BLiJ9 | `db8232245ffd0d8b197bc0189fa1fea04740b66405c0c9046cc56e4f6dda08dd` | 23006 |
| b3_cash_SYN_PROP_TEC_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJTWU4gUFJPUCBURUMifQ== | `7a62ae0372847786a20ffd9083dc89811358a352a85293ae76a85feb60a055b8` | 7207 |
| b3_cash_TAESA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJUQUVTQSJ9 | `a18c925ffc56f9bbd34fa950ccb88ec97ecfa08572c531af7ca7af67a7bede0a` | 38990 |
| b3_cash_TAESA_p2.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MiwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJUQUVTQSJ9 | `7d4360dfe28a3e5f2429e24bbf7c03c7c0078ce5bc21a9a54d164a8660b6db51` | 38934 |
| b3_cash_TAESA_p3.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MywicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJUQUVTQSJ9 | `804c3d05c7424a66d4217aa657b05236eacf61804fb37915c777c5bc71461cd4` | 18722 |
| b3_cash_TAURUS_ARMAS_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJUQVVSVVMgQVJNQVMifQ== | `da91ce86fd953aeb168f48e69f66ce2b3f609a7b9f96632bff6245c84f19ffb5` | 21086 |
| b3_cash_TECHNOS_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJURUNITk9TIn0= | `fd81f2beeb70d6e4f73b1e420d52124ed19664573688e27a0fb841b76b5c8795` | 3939 |
| b3_cash_TEGMA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJURUdNQSJ9 | `085e44f102c19133bf09f06abdcb1e3f146b52f70e5d5b0a84085f33213c4d1a` | 26963 |
| b3_cash_TELEBRAS_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJURUxFQlJBUyJ9 | `c3d481c42aa7d01d9bfce41c71f78001f71ccfa620547d6afd78ba228654fa72` | 5323 |
| b3_cash_TELEF_BRASIL_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJURUxFRiBCUkFTSUwifQ== | `8d5aac59140893010654e2f03ba866bd4966911a5c8e94dcac87adb8d206acee` | 39629 |
| b3_cash_TELEF_BRASIL_p2.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MiwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJURUxFRiBCUkFTSUwifQ== | `beb298e00d486b32b904d2e2303893710d6c7c8c3f5f21c8324aae55c8708a01` | 39600 |
| b3_cash_TELEF_BRASIL_p3.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MywicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJURUxFRiBCUkFTSUwifQ== | `45ec3583ccfd80c25bc4720b6890777a4c2cf1347fe76cb388ebc57fc523d359` | 8378 |
| b3_cash_TENDA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJURU5EQSJ9 | `5060d0020a5b3f27a15820230a2320cf38f694d325a5066547e8d39f4d77c73f` | 4280 |
| b3_cash_TERRASANTAPA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJURVJSQVNBTlRBUEEifQ== | `08a2f4671d7694b60e91fcd81b9ba5eafbec0f4f4fc6a84902b3147e191a87cc` | 1700 |
| b3_cash_TIM_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJUSU0ifQ== | `8b12f3f4c1484968c11b377402bc47b66d3d90209922d7b2ae9d7168837d0478` | 11564 |
| b3_cash_TOTVS_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJUT1RWUyJ9 | `da85dd04bd83f1622f00cbfcfde3b43240736d78d9ca759640e702bd6731ac12` | 16571 |
| b3_cash_TRACK_FIELD_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJUUkFDSyBGSUVMRCJ9 | `2b37f4fe62d80887cb1fd4c7e9773c39ae9685ca3e2cf7f92a848d496072e676` | 10515 |
| b3_cash_TRISUL_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJUUklTVUwifQ== | `72c57d151cd4c0b748ece726359a91b50c50426d3b2083572f510feaf562b18b` | 11039 |
| b3_cash_TRIUNFO_PART_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJUUklVTkZPIFBBUlQifQ== | `c8e27c306d4170ed475943f7d3ba3f7e9f9e6724a40cc900bbb5732b52e26e4d` | 5849 |
| b3_cash_TUPY_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJUVVBZIn0= | `9d6101271ca70cd4b558893dfada5730f6c8a51c4b5bc64ca2d456e24f7ff124` | 20033 |
| b3_cash_ULTRAPAR_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJVTFRSQVBBUiJ9 | `4fd7525ca34093a397fb1da920e6638b65a52b2552d746cb6dbfd090e97af40f` | 26178 |
| b3_cash_UNIFIQUE_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJVTklGSVFVRSJ9 | `0c5a78eee9943de4d256f6ef0423d894547b1284f2d2bec44edc5d6b4d56f97e` | 6901 |
| b3_cash_UNIPAR_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJVTklQQVIifQ== | `3114feb8d2424f5b9764d21b38a1af41874c1a75ecf573296263e49109e3eb04` | 38749 |
| b3_cash_UNIPAR_p2.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MiwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJVTklQQVIifQ== | `077a5d1badd3d2acf2e4463cb6ae6ff379b56d1d9247ad5b91dc5c8732475884` | 38639 |
| b3_cash_UNIPAR_p3.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MywicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJVTklQQVIifQ== | `ed4f486797dd09f03c5f6cc6fce6bc37a7251d608479559b1a463da21c524cea` | 2617 |
| b3_cash_USIMINAS_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJVU0lNSU5BUyJ9 | `638f7308ddd68a71413c750e1322706f8183190d5426684b27bf2178599b8c19` | 38907 |
| b3_cash_USIMINAS_p2.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MiwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJVU0lNSU5BUyJ9 | `7dc7844e70ece5d4326281fd5520142b0cc3cd6e5461bf5c4d3d4bb353704047` | 12696 |
| b3_cash_VALE_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJWQUxFIn0= | `1bbf5fab3a30247e500079a1a35224af7ddb2085aa4a4361ccc69920fb6fc079` | 39188 |
| b3_cash_VALE_p2.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MiwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJWQUxFIn0= | `98ec87ac1c73bf4e920b89b1436e36b39ee9f9e3e6fcb150b5b4c40d64bf4bca` | 10483 |
| b3_cash_VALID_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJWQUxJRCJ9 | `b5303c63ab834df73373f306e9ab2c9f8f5c0f3fa30fd5560dda8ef4644906f9` | 30439 |
| b3_cash_VAMOS_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJWQU1PUyJ9 | `f732498e1d5b5f880dcda6478c85a081671625404d0f7aa565fd80ef796e3543` | 2372 |
| b3_cash_VESTE_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJWRVNURSJ9 | `949af2b2d5381886d94484b9d8f10e733d7d88c243b05522a82324687d7ebeda` | 5207 |
| b3_cash_VIBRA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJWSUJSQSJ9 | `87866b6e0402fdef1bf05eefb742f7f888a5a77d4749da80ff238de3e24e7f3d` | 19974 |
| b3_cash_VITRUEDUCA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJWSVRSVUVEVUNBIn0= | `802fa10b25ccdce514dc2bda2497bcb73300b8672568186359bdd1ec271fc35e` | 726 |
| b3_cash_VITTIA_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJWSVRUSUEifQ== | `86f81b28a930633b002ebbeaa206c686b4178493f6cbe9798935f24828864da7` | 4675 |
| b3_cash_VIVARA_S_A_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJWSVZBUkEgUy5BLiJ9 | `000c7bb195832098a5a581cb88bf2c40ac0652376baaf6696740a70558e273e4` | 3015 |
| b3_cash_VIVEO_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJWSVZFTyJ9 | `26384145ae0069289948b3dbe5a45903b61e68d9189b0a5734631f477a1c9cd5` | 1386 |
| b3_cash_VULCABRAS_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJWVUxDQUJSQVMifQ== | `14f3ed1de6aec2d8961fbcfa75a3d24c93f45b7f7bdf45ea34c2cec313df40e4` | 12162 |
| b3_cash_WEG_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJXRUcifQ== | `dbf534221036ab09c1ad093690405b32512738fd9b383ad9c70d1da197ba6a84` | 39216 |
| b3_cash_WEG_p2.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MiwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJXRUcifQ== | `32503ecdbe52b12205aeb1a4f215e232a49143dea8a79f4dafd5e57f16d9f771` | 9452 |
| b3_cash_WHIRLPOOL_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJXSElSTFBPT0wifQ== | `9c267b459af38af4fa333efe2e2aa67689ef4d1dd1d4d8603889083951195230` | 31432 |
| b3_cash_WIZ_CO_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJXSVogQ08ifQ== | `0efb9c4bdbda373b5d34699cc62621eab9f7f328b076201e88aaea535b1932a2` | 9145 |
| b3_cash_WLM_IND_COM_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJXTE0gSU5EIENPTSJ9 | `ee95b93798fc58bd51799223a7dc8e26397553ec72fbf16ea9439cd78aefef8a` | 35502 |
| b3_cash_YDUQS_PART_p1.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwidHJhZGluZ05hbWUiOiJZRFVRUyBQQVJUIn0= | `97eb300f248411e338dc5d79ff0772eb618c3403a9836214e85d12798f1f24cc` | 8097 |
| b3_company_AALR.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkFBTFIiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `e452a10df8c67c071243957618c00a0e61fa4f4dd8d54eaafe127fb6ad906272` | 941 |
| b3_company_ABCB.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkFCQ0IiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `c0690acbc681fb91cc83a463abfbd9cc5e5c5cea76602cb36867de51b3c21cc7` | 1489 |
| b3_company_ABEV.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkFCRVYiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `0387bb19ba04630530bc1cc31cf9307e4d7cdd5e133dd078c5407ef9240d6188` | 1601 |
| b3_company_AESB.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkFFU0IiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `5c928a8a08b72f52e174d20dfff221a24c4bd70ad9b0e649daed41dc0b00b37c` | 436 |
| b3_company_AGRO.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkFHUk8iLCJsYW5ndWFnZSI6InB0LWJyIn0= | `4c171acee34a091bf7a1c40fda397cc063b47922e3ed280b209b151801b8b302` | 825 |
| b3_company_ALLD.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkFMTEQiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `04d4fafc6574983a2c716e80cb63d44d8b8292b1f5674f7154b07b51ebbffad4` | 1339 |
| b3_company_ALOS.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkFMT1MiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `1e3525539e0a93a19fe7b997c6530ba37dcfe060ba2731b53e83123dde60a0bc` | 3308 |
| b3_company_ALPA.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkFMUEEiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `72ae3ef16d0d0fea1fb0b3f9aa2455d3cf71350ba24971e78e9482a695a0a6ff` | 3428 |
| b3_company_ALPK.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkFMUEsiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `bd2d72cb3ef3d0b6278fadf709b8fa6d4c800d4773018563274f9fae37aec640` | 943 |
| b3_company_ALUP.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkFMVVAiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `4344859a45b86ce6428281f7fdae5d0a66f1d3e5173fd020496b1169264900fd` | 5237 |
| b3_company_AMAR.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkFNQVIiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `f841cc3e5e20e57946d3008ec6975de15365ee72569628765e4d4b894f62e48e` | 1050 |
| b3_company_AMBP.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkFNQlAiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `b6c4aa2cfbd784f9b586a834cb6b4d47b7b7a800f2118dea9fe1c13359c7c474` | 614 |
| b3_company_AMER.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkFNRVIiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `78fb7e7f972d4692cfac5c499d2dd25f71b0478d92d77cc15e0cff2c5cd730b5` | 1081 |
| b3_company_ANIM.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkFOSU0iLCJsYW5ndWFnZSI6InB0LWJyIn0= | `e1b8cc32d551ca177f02bf5714f8f5830911080884fb448b3a938d3f9fa3f098` | 811 |
| b3_company_APER.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkFQRVIiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `f70a6868375c6869512fc66b70884986fda87b1d327e2b34c0f330a674ea0116` | 449 |
| b3_company_ARML.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkFSTUwiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `f3e2e6d0a8a3f7cd03f6038e0266190baa386d478461d9a7eb816232d43f2c03` | 890 |
| b3_company_ASAI.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkFTQUkiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `d335f4e162e832242ba8792fc3ec8a1014be3133b2b5746abc26fbb03eef193e` | 854 |
| b3_company_AURE.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkFVUkUiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `2afb1a31dcbe079b04594ab77fbbca27da08c98ef2967ee895e811200072d38c` | 626 |
| b3_company_AXIA.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkFYSUEiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `c38bd436b522076a62e840e15b76b7c38b970fdb5fb54d7521ddaaa5b3f3be81` | 2799 |
| b3_company_AZEV.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkFaRVYiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `a684b9a3bf00c7a2c18d0c0c51cdbce6f3ddfde81022632c3f636bba413b5615` | 3118 |
| b3_company_AZUL.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkFaVUwiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `b249d036f3e504407a8f327a3d603cbb251a9aaec44df5002a4f4defa75972b8` | 1908 |
| b3_company_AZZA.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkFaWkEiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `893ea41b48b22b7d25c9c3151842c0564ed0419590194a2f0b544afeacb2ee98` | 868 |
| b3_company_B3SA.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkIzU0EiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `1f682fdaee8a6e217dffa6ba5ba7f006f7772105b63e0d557fb16f8261a51dfc` | 2930 |
| b3_company_BAZA.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkJBWkEiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `00eef5f2c60c789a5be0e419d24392581f954a1a9535f7e14af7f9e24c528332` | 1925 |
| b3_company_BBAS.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkJCQVMiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `6dd58623893be3c5b3bbde815c1c71022e33217f6d1f76e9776fb6d75bf5b4ff` | 10674 |
| b3_company_BBDC.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkJCREMiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `1c4d2b89f81b1f387c4c76378eee620fdf99928aa322c55ed2d4eee2bca71e9c` | 10063 |
| b3_company_BBSE.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkJCU0UiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `dfa8d7d5835ec59f29e51fc1d9fbddc97ffa8c2cb564431cf7444f2c6487eac1` | 1100 |
| b3_company_BEEF.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkJFRUYiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `5a5a6fe8a3d17966a103ffcb07e057905979ee661cd053d0fe6547173ac3b8d0` | 1330 |
| b3_company_BEES.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkJFRVMiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `c93952a3243ef26f94563d247fd8b2a7d56d0a6e5c72778017e4b8a065750eba` | 9832 |
| b3_company_BHIA.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkJISUEiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `50091da3f6d478add2d3461c904ae1ecadbfbd3317233db3b11e64d048dac6dc` | 632 |
| b3_company_BIOM.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkJJT00iLCJsYW5ndWFnZSI6InB0LWJyIn0= | `2ddf829ba3991465c708952bb2c765f29671b2548958f8ff4ea9e16b2db61a4a` | 731 |
| b3_company_BLAU.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkJMQVUiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `967778db6f4cc4a93a021c6117d0a2ab255fea3a14f4aec02fdf5d8eb8dd40c1` | 1763 |
| b3_company_BMEB.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkJNRUIiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `947570cc2c999acbc13b331f032c3c5e2cc1eb7a9e4f343766546f3a5d9dd9b0` | 3262 |
| b3_company_BMGB.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkJNR0IiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `de7ef5078595abd9b392b349b8b7161c2d6ac63399e8e894f449cae6ff6f7cc4` | 3359 |
| b3_company_BMOB.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkJNT0IiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `49e5292f8bf6e52cd56f3340c239e1a5451a435e5f822ec6c313b6c53254bc6a` | 1575 |
| b3_company_BNBR.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkJOQlIiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `64475ab64fe48823729d77cfc8509689e97417c0793d1733e6c4ba9551b2664b` | 2149 |
| b3_company_BPAC.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkJQQUMiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `ff9c22e8f876e49a8966a9fe41c13a2dc898a97250ee79f0a2c522b4fdb882d7` | 3488 |
| b3_company_BPAN.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkJQQU4iLCJsYW5ndWFnZSI6InB0LWJyIn0= | `99fd8eaf8eee2d9f3b77c3948b9b6bcc282145dfa30e5a277cddc309e1298df8` | 513 |
| b3_company_BRAP.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkJSQVAiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `a017eeaf47b28021f1ccb6f560c8f4dca735d65cda98d74cf0f8865894666c0c` | 4216 |
| b3_company_BRAV.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkJSQVYiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `a24529b1cb8a294ae8d3fe195a9c661991435fad0b342ec96f809f57f72fd8c1` | 649 |
| b3_company_BRBI.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkJSQkkiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `660d737c9112b92bae7e8263864fb25bc12f7c0a3efd316f405a9a1621dcb73b` | 2557 |
| b3_company_BRFS.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkJSRlMiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `f59d9e494662072dce1ea849811bb98f2a66690639c46d719e9a372fdfdc2738` | 792 |
| b3_company_BRKM.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkJSS00iLCJsYW5ndWFnZSI6InB0LWJyIn0= | `957fe783d50326f21a09d713505eff95df16d52879e15417e71745b96b3da6d6` | 2350 |
| b3_company_BRSR.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkJSU1IiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `30bf8c5d23db1b2ae1c5830bf48246ec550d79be946014aca1c7d608a073c5a6` | 6052 |
| b3_company_BRST.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkJSU1QiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `4da08a566d58535cf3bb65d72cc0da09409299e19c42a4e2c0b081f8566c29c3` | 675 |
| b3_company_BSLI.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkJTTEkiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `dabdade287f9029a9493899c6c17a819215c0b23a3209a0243cb437b6a11fb07` | 2380 |
| b3_company_CAMB.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkNBTUIiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `6110e8cf93617c1a2916a6c332e8b9c16f8f620917aaaa74b9cc28eb30522715` | 4716 |
| b3_company_CAML.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkNBTUwiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `bc66a6544f290453c6b5db5cb91109401aa3cfe309b9d75f012a58ce24c9d57f` | 1595 |
| b3_company_CASH.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkNBU0giLCJsYW5ndWFnZSI6InB0LWJyIn0= | `66cfe51218f081ffa3c04c1b2a714463e3729ae732ff2d4dbf21f7f8c408cab6` | 1071 |
| b3_company_CBAV.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkNCQVYiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `69b6ae5d15e5f12e3c7653e06373880a8758aa0e2b85683e7bdad3d7a0d02bde` | 945 |
| b3_company_CEAB.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkNFQUIiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `bebff1af05246d76603e3e02ef864c06e134f664b63a0360b101a16d707b3723` | 675 |
| b3_company_CEBR.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkNFQlIiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `489f81ac69f916b11518b08c9a0bdfd12b8063bde2d46ec51a55c8f0d29eb6ac` | 4665 |
| b3_company_CEEB.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkNFRUIiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `4839228ec9af68df65cd8eaf287eb8e76f2f095fd217d04b15e8da865f166437` | 5149 |
| b3_company_CGAS.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkNHQVMiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `232dd4b5913b8adda0bacf6163309519c810aef26658250cf82abb87fce848ea` | 3196 |
| b3_company_CGRA.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkNHUkEiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `2776a5be441a143fe4d944bd6080e9c9a4bf057801c3caddd7ec5df8aefb1b99` | 3711 |
| b3_company_CLSA.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkNMU0EiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `a0c394012125885ac5c46022f45b0bccff683538104584a3f8cf2d966a621d5f` | 430 |
| b3_company_CLSC.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkNMU0MiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `f56f28fdf3ff557e4fc27a9d92a076096daec7f52891c0c2b7ff4a47c28aa0c9` | 8037 |
| b3_company_CMIG.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkNNSUciLCJsYW5ndWFnZSI6InB0LWJyIn0= | `1bfcd6a5d1c5208cdc8dad44b29bbfeed3eff51a8fd7677d6de09a32ad1e92c9` | 6650 |
| b3_company_CMIN.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkNNSU4iLCJsYW5ndWFnZSI6InB0LWJyIn0= | `50c993fdb998197151f10b7f96e3824d98711b5cd8f80fbce66d285d54770225` | 1548 |
| b3_company_COCE.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkNPQ0UiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `03efb8767a8ee2d7780c20f3e046dc2a877471d10a9f0b670661745b936d178f` | 4088 |
| b3_company_COGN.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkNPR04iLCJsYW5ndWFnZSI6InB0LWJyIn0= | `1efb1970d6cce7a1d2d21f5228db671b276806a186fd73afb90b5446288179ae` | 1294 |
| b3_company_CPFE.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkNQRkUiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `1b78f1925d2975827d1b036ebc2b9dec2d6b8fc2f6077dd5e9d46845a866bbf3` | 1158 |
| b3_company_CPLE.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkNQTEUiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `e90f34a2ff13e1f1914bfe624f301d638d8cbdcb64f9b37cf1c1972842ae0175` | 3205 |
| b3_company_CRFB.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkNSRkIiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `415e330ce288b9707efaa816f9b068f39575f2539dfdc2dccfe8f6be364431c9` | 437 |
| b3_company_CRPG.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkNSUEciLCJsYW5ndWFnZSI6InB0LWJyIn0= | `3b64083d7659b608b431266621fc2d48d3c98e6788f0dd09bbaf6d491db7d644` | 985 |
| b3_company_CSAN.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkNTQU4iLCJsYW5ndWFnZSI6InB0LWJyIn0= | `a13cd4a9a1e25eca5a38a3f44a11dae83a7b96f58b95b22ee89a924e03664c82` | 1409 |
| b3_company_CSED.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkNTRUQiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `142635d54e35bac9ee6c0f6bd86ad57b150e347cab813d8f8845e055273d8538` | 903 |
| b3_company_CSMG.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkNTTUciLCJsYW5ndWFnZSI6InB0LWJyIn0= | `4f3dea82569aed85148b95eb3304f6d47026950e6fe8a3556f1fb4ed55c254e7` | 2510 |
| b3_company_CSNA.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkNTTkEiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `83c79dc771ceb3fa1a5e2bebd2256fdbf66c11d7e7ce36c33c7bc02308730dfb` | 1396 |
| b3_company_CSUD.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkNTVUQiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `1f5867a68c334b811e0246fd3786cbe1de6846488f93067e01949e1cdfe2787d` | 1601 |
| b3_company_CURY.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkNVUlkiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `4959084b078000c3b49ed00c44526198f28bdb49bbd3599c25afec3c3c0d65be` | 1576 |
| b3_company_CVCB.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkNWQ0IiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `0a67cab65e5a461eb194e2e52465ca0325d74168626897da974eba70f8fa6ea5` | 779 |
| b3_company_CXSE.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkNYU0UiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `bbaecd43abf95b3e2c9dfe796d0585971eb0314620bc03b3021b390ccc03f765` | 1593 |
| b3_company_CYRE.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkNZUkUiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `3336fcc87061ddb96212c76a1de22041097112b97e521129b474077032a40fbe` | 1980 |
| b3_company_DASA.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkRBU0EiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `639bd4e78ab12223329985d48034a665a8c7ada1e2d56cc7e7c907c32c08daed` | 1080 |
| b3_company_DESK.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkRFU0siLCJsYW5ndWFnZSI6InB0LWJyIn0= | `9d99a9f0299bd9844db43c0f42af8aae5aa288966ec5611d7e8edeef42552684` | 1393 |
| b3_company_DEXP.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkRFWFAiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `274807c859973b74b50e2353638f6c88578f4504e5f2144b8b941091cb13c01f` | 1601 |
| b3_company_DIRR.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkRJUlIiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `487a4652713a918aae7395aaece198d91cba74845f7453c1388fda35f27890c1` | 830 |
| b3_company_DMVF.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkRNVkYiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `25c098475ca9c30f7fd404547ec24168328cf75165ac23c863291de9f4ddde60` | 448 |
| b3_company_DXCO.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkRYQ08iLCJsYW5ndWFnZSI6InB0LWJyIn0= | `94ebea6826dac839629d18beab484c8d9af31973b137a75fad0ba34e723d8862` | 601 |
| b3_company_EALT.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkVBTFQiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `4562adc4776a08790bf373cbe17bcdeab2d73b790d930a25b0eeb71762845b43` | 5005 |
| b3_company_ECOR.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkVDT1IiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `cac2cd22a451b448921faf956ef13a2e240cddc6d5509b3ff8337a7255bbbe6c` | 648 |
| b3_company_EGIE.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkVHSUUiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `2ad0f772c87039fcf23c7b1c0889c333fb1e7cd1f9e6762c289c27df0cd9da50` | 1305 |
| b3_company_EMAE.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkVNQUUiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `9ce769975f5db9e422776a5063f39780ac3835d700a500c63b45f0abe78e3358` | 1514 |
| b3_company_EMBJ.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkVNQkoiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `7fdcc9b11aa209ad2d42879180bcba8098459da17c5ab30371c8ce9060974496` | 1640 |
| b3_company_ENAT.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkVOQVQiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `4288c0e9150fd283d07b0416a2f10323c02542208ae229867408bf65829b1863` | 583 |
| b3_company_ENEV.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkVORVYiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `48bec09203110b6295364717e429cdd04aacf754a36e78396722e58b54502066` | 1058 |
| b3_company_ENGI.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkVOR0kiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `51065ee893113a78000ba31e05e3c717c942d50182bda79ca9dc136c904c4acb` | 3542 |
| b3_company_EQPA.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkVRUEEiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `f69cbaf535fc839e6e032a99b34ed27eb3e92e35b99a038aaaedabf0b9cf1eed` | 3113 |
| b3_company_EQTL.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkVRVEwiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `f28e3232c16fccfe8623671b64ec9787c0d508fdd01da128b00be105cc78b342` | 1664 |
| b3_company_ESPA.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkVTUEEiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `13624cee59de0b4df14168276691144103bfe64ba974569358e62c1e739e33eb` | 1554 |
| b3_company_ETER.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkVURVIiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `d93d530f6c8b5efbb98d31fe200d0113b99d529f92442daa803fc957e623ec87` | 1659 |
| b3_company_EUCA.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkVVQ0EiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `b62fba4ddf22b8b1be341a034611b8fb841d92a31867c11919b5052964da6a3d` | 4530 |
| b3_company_EVEN.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkVWRU4iLCJsYW5ndWFnZSI6InB0LWJyIn0= | `cafb166d269064e2569868fd865d87b1dd8989bbd3ff62f2decfea91f3c25479` | 1161 |
| b3_company_EZTC.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkVaVEMiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `00f06cc1cccdae0f35f043d3506f946454b5ac891d5d594151612b3fc70e5beb` | 1512 |
| b3_company_FESA.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkZFU0EiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `ceaf601b9abd588ad9a620d3596acac38f5a6e29d06225651e43b9116c504f73` | 3077 |
| b3_company_FIQE.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkZJUUUiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `b94b40934560fdaa72d25deb918cee7f891dc7da94cd1cb71071db977b1015e7` | 2387 |
| b3_company_FLRY.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkZMUlkiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `79ca1bbfc812e66c600e3f589b68d606cb9d58e50425bbe4a9c92fe6b64b107a` | 2153 |
| b3_company_FRAS.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkZSQVMiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `24b2506f79ca5de20d7e95288a6376be6781b983b28f550c11849b6cf9d1e07f` | 1431 |
| b3_company_GGBR.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkdHQlIiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `980b253f587dd162f86a2c4517c142ed26782ffc50d6a7e06d1c5e89ea8eb8b0` | 3025 |
| b3_company_GGPS.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkdHUFMiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `8895a05a0c283560712f4d2276e716d2c848e69abcc7612d381c09bac3c66efb` | 828 |
| b3_company_GMAT.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkdNQVQiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `f06f032469ec09f59cdbd7e229b8d7417e715bcc4788eca5ff17de0cd52c0dd3` | 942 |
| b3_company_GOAU.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkdPQVUiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `8206a4a2388b12d4396983b8b4fce0a20fdfbe9deea12d0335798ade44608d0d` | 4510 |
| b3_company_GPIV.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkdQSVYiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `48181d26e5b3243ba4c1e26947ccb81eae0189d0d358755b6b21d809fa88db01` | 413 |
| b3_company_GRND.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkdSTkQiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `68cc25828367ba8109e89e3d93da7443e349c3afeabf21932d25f4ac6f41e38b` | 3840 |
| b3_company_HAPV.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkhBUFYiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `861b0ed9262e90b7f18630b73cceb0624a5576c0466a89a11750e0fe5e926c0d` | 806 |
| b3_company_HBOR.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkhCT1IiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `9e04c0f3c8a9911e98e080aabed05ee927b4afc12d6e8862c3c1cf66901369de` | 1655 |
| b3_company_HBRE.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkhCUkUiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `b189e46a8423e75d7f9c6d23142ea33289f942ba079fd7234c63aa1b11dc3e1a` | 872 |
| b3_company_HBSA.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkhCU0EiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `c96f50b93cb99e9e80c7147c3a74350ede25fa3f23c99e5c67cfd5494696087b` | 907 |
| b3_company_HYPE.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkhZUEUiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `6a8609a8ae894df9fb8e5b360ba376405fcf7430f0731b90bb303bfd2fdbdcb4` | 1824 |
| b3_company_IGTI.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IklHVEkiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `eca1f7a107db9df32f830e2a2787e42c6b7b4296e6c2832d818ae8fc38aace1b` | 3636 |
| b3_company_INTB.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IklOVEIiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `1ec74bc4d1a4cc2a0b1176ebdbe338d2ccb3280da06974cf9d7e15f10313e46a` | 1137 |
| b3_company_IRBR.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IklSQlIiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `c7a698be29f5ec839943c24cd88dd22efdfe97dc4fd41409f0ab144af6afe173` | 2198 |
| b3_company_ISAE.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IklTQUUiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `4ff46f04940adfc60972917736f5217c56d0d960af96b9297c14b50330da7c66` | 4514 |
| b3_company_ITSA.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IklUU0EiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `3fe45778c1e45be344d9062250198fb0d2105050c322ac8988f91dc12d8e4ede` | 5367 |
| b3_company_ITUB.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IklUVUIiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `3c561a6b1540b54b4570f546e7e6203a26f00132ad998f4c67684a3c647e100c` | 9512 |
| b3_company_JALL.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkpBTEwiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `e671b1059e051843f575167da3b4d6d55cbc83f5e98ed68784deec566a6527d1` | 945 |
| b3_company_JBSS.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkpCU1MiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `aadf4d587d3bb08cd718d7ffb96d25b7d15c8d37d5b1986f99493bf25ddb0688` | 1082 |
| b3_company_JHSF.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkpIU0YiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `69aa6e86dfe842bd627bc3b14b2945831bea13d92be6e02eb0572b7419dec1a9` | 3366 |
| b3_company_JSLG.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkpTTEciLCJsYW5ndWFnZSI6InB0LWJyIn0= | `9961050812786f93adcb380efd57faa9cd7f9569624c0284d6677ff18bd547a1` | 1042 |
| b3_company_KEPL.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IktFUEwiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `0d50bf987e9625926d4fa573ebeae019483729132a115bea9f0ac30003084dfd` | 2041 |
| b3_company_KLBN.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IktMQk4iLCJsYW5ndWFnZSI6InB0LWJyIn0= | `865a83b6a22d6c614b12639adaf6384775ceb750ac78aa1fd684a13f3282fe3b` | 5770 |
| b3_company_LAND.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkxBTkQiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `745f6eeff62a19805f5802ffe47aa2a906b546e2f49b805b2ad0d67ddf45ebbe` | 446 |
| b3_company_LAVV.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkxBVlYiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `5e1635e4681d2f070a69ba861bf789b0fe0b834e37d9febf2af25c6a8fb2ec39` | 1110 |
| b3_company_LEVE.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkxFVkUiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `dd382cf8272476068aa178f01eeb45dfa0b8622766c6941f49f06b35d6a562eb` | 2703 |
| b3_company_LIGT.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkxJR1QiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `8221035b498de38d898dffaad729255d670fe1691441a1862fe1907adfb1cf58` | 1046 |
| b3_company_LJQQ.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkxKUVEiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `dc9a1bc19d27b7a126c185a8a1e17f1098d312113f60d072c710e52640fa1e07` | 724 |
| b3_company_LOGG.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkxPR0ciLCJsYW5ndWFnZSI6InB0LWJyIn0= | `59b5b00328d135379fc7b5b06d2f449b342c775e492a959b8468a482a302ba69` | 1863 |
| b3_company_LOGN.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkxPR04iLCJsYW5ndWFnZSI6InB0LWJyIn0= | `28d195e3fe49e033d168e6b4b8d56fed15b7733f3d046578663e90820f9f8ec8` | 906 |
| b3_company_LPSB.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkxQU0IiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `903dc88772efd908c26d7045017a1f14521adaf8def8a793d76a810f59041666` | 1345 |
| b3_company_LREN.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkxSRU4iLCJsYW5ndWFnZSI6InB0LWJyIn0= | `c0459c1c41a9db9552cad5f13de77afff0f9673c956242c4abc799940fb2bcb7` | 3162 |
| b3_company_LWSA.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IkxXU0EiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `11f1965cb9bdc65bc170b8502a38a00e1a0b2f7b40d0628e22003d7b56fa2c2d` | 1081 |
| b3_company_MATD.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6Ik1BVEQiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `c7e48d495a95e853341eeba3a3b88173b76a55ee0c3ad223a417ef04759ef984` | 831 |
| b3_company_MBRF.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6Ik1CUkYiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `4ef27f46a8be7feadecd8c509a86eedc04a23087871f88365212a252bbb0581c` | 440 |
| b3_company_MDIA.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6Ik1ESUEiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `22bba1b121ff7339a1806cefd56e4d11f0b121abe843baf916cef950778680ca` | 3503 |
| b3_company_MDNE.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6Ik1ETkUiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `95137984677f937790609227d138d1f90fa02ec2480596404a4ed36a268260f0` | 2188 |
| b3_company_MEAL.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6Ik1FQUwiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `3de3de45a66ddb1e0c572c43d50880abdebd8c78cc19b973dd695b9b60487dda` | 707 |
| b3_company_MELK.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6Ik1FTEsiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `7cf2ad83b6c5c5651e9cf04d2ebad8f1d2f3c8fa39775877174ae50fd0f8015a` | 1131 |
| b3_company_MGLU.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6Ik1HTFUiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `742100ac89abae1a0ea712bfbcfaf8373269b47e9a58c0a917914f76d018c6da` | 1464 |
| b3_company_MILS.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6Ik1JTFMiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `b5b9b62ea0ba4474cc1af5782af49155b668f8b5a2dfba26a03e6647a643babd` | 1187 |
| b3_company_MLAS.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6Ik1MQVMiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `8e3db0f0676480c8b0b61307bfe96053b885f96fc59b98bc524e821fa0e5cece` | 669 |
| b3_company_MOTV.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6Ik1PVFYiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `08ecb4066f166e1372fe04647e2a7b4557fbb58ebb0e687361ffe3e09216f444` | 1113 |
| b3_company_MOVI.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6Ik1PVkkiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `48eea770dcb95d4b3bdfe8d069380ef12d6aa8df0ca39a13a825dbe5ee3db983` | 1205 |
| b3_company_MRVE.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6Ik1SVkUiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `825b20e0bc6b7d0fec5decce7d445a07d4b378928172bf106cf0771c16ece6b7` | 971 |
| b3_company_MTRE.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6Ik1UUkUiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `55e8dc94a97a6b372a842651e1c93111e15b4d737fa3b920e76b3615e39138a4` | 2676 |
| b3_company_MTSA.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6Ik1UU0EiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `7153d68d1b632bdb30e609f6d629825390ff7349f2488ba29ba6ec1aad4bda79` | 3048 |
| b3_company_MULT.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6Ik1VTFQiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `f9067f5235c3917d5bcfdf48a3d02ed24c42f24fc731effde7fb09c6f5fff2c0` | 1846 |
| b3_company_MYPK.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6Ik1ZUEsiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `e69472a6ecf923289ee591b68888ad5f3c469b9b1ac80a57c010534ec8c5639e` | 4830 |
| b3_company_NATU.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6Ik5BVFUiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `1e23b770da2a46d7fb8b756e85d977506aa74a54ee65f8090f81ed76e22ea7a6` | 957 |
| b3_company_NEOE.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6Ik5FT0UiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `bed3277c5299ed42bc1f1c0c441d94de940b4fba46c65f7600bbe1a8bc0e2c2c` | 1508 |
| b3_company_NGRD.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6Ik5HUkQiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `bc52d66277e3d22bd11c8330a97827feb0f9046d9ca0e35e87ba262cc1840a7a` | 426 |
| b3_company_NTCO.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6Ik5UQ08iLCJsYW5ndWFnZSI6InB0LWJyIn0= | `a1b91cdc299eb503b61f76b30a1160d755c99de77ee7569e8e65049ec7a7e983` | 437 |
| b3_company_OFSA.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6Ik9GU0EiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `e2a1e67e5997b8e0661dd5c35fddfcb1bc4d6435ee9e73e2e8bf9e000f56f10e` | 1078 |
| b3_company_ONCO.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6Ik9OQ08iLCJsYW5ndWFnZSI6InB0LWJyIn0= | `16db27ebcd6ad3d880b52e623695fedc54f6fbde1b5a9d5fa85dd6f8ddb8cebe` | 732 |
| b3_company_OPCT.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6Ik9QQ1QiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `de40151fdf394080aebc1ffecc8828b09e7e5687e51144906883caff17f861e6` | 995 |
| b3_company_ORVR.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6Ik9SVlIiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `d8886698e27f6ac5273467d18d957727a64e847ca195c3a8d8829e1de9d7d68e` | 954 |
| b3_company_PASS.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlBBU1MiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `4df9840fdc8136195c712ce33f4943b7eb56e2fb72401fdd07fc9530bdd45e65` | 2801 |
| b3_company_PCAR.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlBDQVIiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `58d3501bd0b73f1e797deb58918ba08df926d795115eb70291311ff4f4894cec` | 2929 |
| b3_company_PETR.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlBFVFIiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `9640204b919e1b8050e2370e4d007510b44745a43bda35b453af25c3ca3df283` | 7148 |
| b3_company_PETZ.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlBFVFoiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `dd70ed21e6c6710d2ced4c44ea79a414cbc648c0a2b007874817623852f448b4` | 1218 |
| b3_company_PFRM.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlBGUk0iLCJsYW5ndWFnZSI6InB0LWJyIn0= | `0f72b81bfac463cbab5d4808d8463fcfb48c44208c3b3433f3309af3c53829a8` | 1161 |
| b3_company_PGMN.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlBHTU4iLCJsYW5ndWFnZSI6InB0LWJyIn0= | `a0af19070b3b168f33d489262828fdc3d55e7052fd664f0ae5a4f2912e14f1c6` | 930 |
| b3_company_PINE.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlBJTkUiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `3ad35e064b4fa36f4bd3cdb98f3e0b261041177c81bd896f78a7e1429aaf8771` | 3734 |
| b3_company_PLPL.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlBMUEwiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `8fcb72e4098726e74c444431c15fdc9aeba42f095bd63ab3b0ea696e5f6ac950` | 646 |
| b3_company_PMAM.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlBNQU0iLCJsYW5ndWFnZSI6InB0LWJyIn0= | `703587118006f4991bacfde09e3821ec4e69aa1faffb90046595f946cb793b98` | 4869 |
| b3_company_PNVL.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlBOVkwiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `45da80f372075342a3cdfac774e3e44d4210f3a734505b7076ad919378c45c10` | 4783 |
| b3_company_POMO.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlBPTU8iLCJsYW5ndWFnZSI6InB0LWJyIn0= | `2c7430ecf0a72ccab3f4c84375a57b6daa75d54c2f0010f7526b18a98953b179` | 3469 |
| b3_company_PORT.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlBPUlQiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `8229bfcc136da708e4dadb8def4841978f72c149fb222b02cd31c30f8a1f1ef0` | 453 |
| b3_company_POSI.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlBPU0kiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `5baa31b41fcfc8f4c326e2e6c3992a00c2b674eabf4240e1d61aba08b3b80a3d` | 648 |
| b3_company_PRIO.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlBSSU8iLCJsYW5ndWFnZSI6InB0LWJyIn0= | `d0c0a4d49e8c3c8e54a5452477ec09a43f041aa73b274ab1c16b618564188d89` | 776 |
| b3_company_PRNR.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlBSTlIiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `85a0b79d7400684400de480ff6db6d18702b2bc22adbd8ccdee7588a6c0be4c9` | 1155 |
| b3_company_PSSA.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlBTU0EiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `1044e5334ae6ba855f5dbef21bb725abfd4acd464ee4b560efcca3754ad74e17` | 1954 |
| b3_company_PTBL.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlBUQkwiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `682567a9b98ed455d0773a4f0f2dc461ded60cc0631a56802d4e63c0906eab93` | 1696 |
| b3_company_QUAL.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlFVQUwiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `dbefb4a6246989239d00bdeffbca2e6e5344939d38bd7957e6c31a94e766cb55` | 667 |
| b3_company_RADL.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlJBREwiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `585c42ecb42ca8c883197f2113620b7fa4680cb7699a6acd5d4a77227c27b166` | 2197 |
| b3_company_RAIL.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlJBSUwiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `d463ee764891b7a403e7d0a339abb6e5023406874c87143af3d470ee041e4ddb` | 683 |
| b3_company_RAIZ.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlJBSVoiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `38ebc06b1ce6ab94cc1eed52ff4cca875d96feb0ed1bf220bfcd655170deee54` | 1354 |
| b3_company_RANI.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlJBTkkiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `8911da629284149873e51aec3c39e92afef5c33993a82df5e11ef3f50430bc6e` | 3189 |
| b3_company_RAPT.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlJBUFQiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `8702270d8037a9028d78ba2c2f06a845ab6fc7a53d99c37ca6dd4127793a3dec` | 1709 |
| b3_company_RDOR.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlJET1IiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `9e41ae628fe0724e0b77220431365596f8b5faf4dcefdb754d2d6a23b4b614fb` | 1823 |
| b3_company_RECV.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlJFQ1YiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `bbd06b2fab7da6e3fce177439d6cff33b28b6c722a78140011acd34979adb440` | 1556 |
| b3_company_RENT.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlJFTlQiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `80b0f321784984e9987967abd75f08a3b7fad3f1d2967a1172f45c4e798a70df` | 2734 |
| b3_company_RIAA.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlJJQUEiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `b795b59de5933cc0a4794c5302449c9270321ea3d966e7be466792cfdcd1fda9` | 1392 |
| b3_company_RNEW.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlJORVciLCJsYW5ndWFnZSI6InB0LWJyIn0= | `f12bc373b686684b0e16f90f9cf6f9c1a154bfaad7a111a1d68ff48221105c55` | 1533 |
| b3_company_ROMI.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlJPTUkiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `1c81f185b3c9796d5e6a3f7b0a408d8c31f1a14e15b87c10213d01b9d7df8452` | 2536 |
| b3_company_SANB.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlNBTkIiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `df91abda096be136a0c2596c5b269f9663ce1553693e9367d9c10532fe36b9c5` | 6524 |
| b3_company_SAPR.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlNBUFIiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `88f3085e05f9b6d2aae913569fcbb5914f89c8d5705ef8bf43c1bc80d5eec1d6` | 2577 |
| b3_company_SAUD.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlNBVUQiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `a9ff7bfe1f1556e51cae1bfe32a0f36b4a71a987e5a35a4a62415dd0f3b518f1` | 1117 |
| b3_company_SBFG.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlNCRkciLCJsYW5ndWFnZSI6InB0LWJyIn0= | `63316c534c28457c5a9c842574ce1571f8e5210edb0656e8eb23ce158fbcfeaa` | 648 |
| b3_company_SBSP.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlNCU1AiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `037b72a06c1927b7bc2522069a4e235c23ddb13aa4dcc6b0a73bc3a222a8d6cc` | 1853 |
| b3_company_SCAR.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlNDQVIiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `c189ba01cc73bb12eb88c4afd96cc676985fd560a9c13a6bf7139b6daf4e9eaa` | 1542 |
| b3_company_SEER.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlNFRVIiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `f2d79bda952b7a814b2afe0fafdfc6f9c59e3ad76dccf765b5c148ddce9f887e` | 1614 |
| b3_company_SEQL.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlNFUUwiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `e226bf3de26f58db8ce0a18241ae8a4d0a4ebce2c086026bc2bc24b737c9e0f8` | 1208 |
| b3_company_SHUL.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlNIVUwiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `938a1576c5f45f2aa22c68c95a6609098107f4e786cf0921579a62ebfa06bb76` | 4658 |
| b3_company_SIMH.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlNJTUgiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `9a212fa0b6e15514fb16b49bc8f36061336d8a720c7409c31b6961d839965ab4` | 1260 |
| b3_company_SLCE.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlNMQ0UiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `26740e82e7bdb2f1be97ece891f33667a8653f3cf8dc963d97ff413cc6c3dbb7` | 1236 |
| b3_company_SMFT.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlNNRlQiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `d887fd3286f9b12cc8eb66aeba8808f46974399645686fe0e7d7cda4d11f1caa` | 1961 |
| b3_company_SMTO.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlNNVE8iLCJsYW5ndWFnZSI6InB0LWJyIn0= | `ed9dd79f56cc8bf68feeebb3a4baf52c40109c57d58bfb25c9b020930436f37f` | 1122 |
| b3_company_SOJA.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlNPSkEiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `7a4eb8b3015b1f08c52359a614baa0850caaabc69363ccaa93e75674cb6c3d7b` | 844 |
| b3_company_SOMA.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlNPTUEiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `38e7478eb9e0876a8ad9749ec37d2cac08a61f087cee5d1b647d4570ee6c02bd` | 432 |
| b3_company_STBP.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlNUQlAiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `9e75085ebe2727c50c60da0f3b5b1dada7b4cedbe07de1bd82be1e660caaffca` | 625 |
| b3_company_SULA.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlNVTEEiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `42c834fd53fa6f2be906d37f504dd64b3992e136e6d9245eeb5c0bf58ce850d0` | 474 |
| b3_company_SUZB.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlNVWkIiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `6ea6f67bca458dee17613994a78d1203705ab125f12ee421341c82c86f126ad1` | 2270 |
| b3_company_SYNE.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlNZTkUiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `498379cfe4f247a4169159dd2c45ff96407a8695a3aa4ea40c84565c9a115791` | 868 |
| b3_company_TAEE.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlRBRUUiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `9ab86c56b8a0283e930ed4398d2c82633ca0e8df223e3b8841b2a2eb2539deb5` | 5881 |
| b3_company_TASA.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlRBU0EiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `4814043c213ecd177a8de6a07ea59f4ed157d06ab5765926539339b816bf2f35` | 1248 |
| b3_company_TECN.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlRFQ04iLCJsYW5ndWFnZSI6InB0LWJyIn0= | `38176b314fb0d8388535b7aa82b8d0f668441210e5ed59c0b6d38fb04386c05a` | 880 |
| b3_company_TELB.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlRFTEIiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `9a5c83ad47467b08aa12a830d4a0585d6d253dd5853d2aa88595a7f90edd4a54` | 6214 |
| b3_company_TEND.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlRFTkQiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `9b8d4353af1380f262e22056da074637bdaac97f11805346b9d4bdb90723fbab` | 1230 |
| b3_company_TFCO.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlRGQ08iLCJsYW5ndWFnZSI6InB0LWJyIn0= | `e26cc2c521a14c3acaa9eef6daad52b05164da7193424390fe3992bd018e27c8` | 2763 |
| b3_company_TGMA.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlRHTUEiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `5e7cdc95fa02f665ef011e36c65d4a783088777ba7cc28abeff732f3c7c6aa9b` | 1111 |
| b3_company_TIMS.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlRJTVMiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `5a36dda5525fb46c4ecc348da1771db31fcca7148d39aaff46653ac6b9f00521` | 1925 |
| b3_company_TOTS.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlRPVFMiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `27477956f7d2293ab636a5230c723546e36ae4a46aa771dec32efab5b2ab11e8` | 1846 |
| b3_company_TPIS.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlRQSVMiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `fd4a41de0854d95f03842b0ba4f69fafc6a58a36c07e5dcf1afd88db07eef436` | 1099 |
| b3_company_TRIS.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlRSSVMiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `463f3a991f8786481d70ee92fbba3fac7133897a273694a1e3d1beb0bc404e8f` | 1664 |
| b3_company_TTEN.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlRURU4iLCJsYW5ndWFnZSI6InB0LWJyIn0= | `daaf4591a804615ceca888853ba8a9fdd0ca61d5b26185e3b9de8582d3b1eaef` | 643 |
| b3_company_TUPY.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlRVUFkiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `5e117e16b9053c91248d8b5db7ebecf7da49a9f2e217328e76d38ed77b1def8c` | 1934 |
| b3_company_UGPA.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlVHUEEiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `225d7cbb4ff0d41e8f9c899d2d0cf2c6a505c7abac3b5357c5fb59c3e2a83514` | 2056 |
| b3_company_UNIP.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlVOSVAiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `4f37080839008f7a55449dccd2efcaeaaa2f3a8fc668f8e2797989e3849237ad` | 3375 |
| b3_company_USIM.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlVTSU0iLCJsYW5ndWFnZSI6InB0LWJyIn0= | `385cf9807fd1788ddcd95c643d23d1838f15988c2032f0730a67ee497bc0a1e2` | 2669 |
| b3_company_VALE.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlZBTEUiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `cc2d0f5b3c00cb08f6180eb60eeb241b35e84d1109bd4338e5aa768d22ee7dc2` | 4336 |
| b3_company_VAMO.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlZBTU8iLCJsYW5ndWFnZSI6InB0LWJyIn0= | `c3e05f3e73077c0444514ee76e4f1bceb7b1627d1395c5a280d859107b764818` | 1463 |
| b3_company_VBBR.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlZCQlIiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `5ea7cb11f13a9f62bc17cf9575650dbab7faa29eec96ae6d1be2398c7e84c665` | 1551 |
| b3_company_VITT.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlZJVFQiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `7244d18607edc5bc6e27f4a127943c40bf9c4ad8b0ee0dc94713a0ac51a35ef0` | 1297 |
| b3_company_VIVA.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlZJVkEiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `5b998a02b1ec161092b3457b7c9e0cd295ad615112ae78c92d47dc33ae923b2b` | 656 |
| b3_company_VIVT.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlZJVlQiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `2a4e8a08b826d13cfa0d12e24dca5edc0d7f8f09c487819bef6db163cff808c0` | 3580 |
| b3_company_VLID.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlZMSUQiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `8ebcb5adcaf85c9d1760fc5529e2ba03e7feb70211810ff66b75b782fa91303e` | 2235 |
| b3_company_VSTE.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlZTVEUiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `54efe0e0998854a9ec97abcaf53d5ae0fd721cd2e7cebd08ab980177368837dd` | 923 |
| b3_company_VTRU.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlZUUlUiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `8ced23545e5399b59f9a8901d82728f12649c4770981d87d5809ad36b9ee5eda` | 669 |
| b3_company_VULC.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlZVTEMiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `37f87295f01d07c36ea7340c7c30a536e6014f97c2b4c3a25e319ab7fa938736` | 2153 |
| b3_company_VVEO.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlZWRU8iLCJsYW5ndWFnZSI6InB0LWJyIn0= | `055335c01d337a565c28add7928f4425250091f6e24c49897ec4ecad8eaee858` | 711 |
| b3_company_WEGE.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IldFR0UiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `27aa18180c21c4a9a53dc2bcc5d9cc0ad653e2846850f3b2ddb414625eb902f7` | 2631 |
| b3_company_WHRL.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IldIUkwiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `49fc7a78b9b8238f32536097e6284adc63aecb13fb4ce408cfe5e5787a74b878` | 1345 |
| b3_company_WIZC.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IldJWkMiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `f2c0d7c4006018cc4fae789c305c08994e0d51e5fe7a6e9dbb3de08df38bf7b5` | 886 |
| b3_company_WLMM.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IldMTU0iLCJsYW5ndWFnZSI6InB0LWJyIn0= | `1d5fd07d00d38ae018f583e6a15fa01351ed9663be5eaaafd9fff887bb6bb523` | 3628 |
| b3_company_YDUQ.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IllEVVEiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `5f1012426cfe1f84fee802d1bbfb6d5de5da58db70934c3d7a77b037b2548985` | 675 |
| b3_company_ZAMP.json | https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/eyJpc3N1aW5nQ29tcGFueSI6IlpBTVAiLCJsYW5ndWFnZSI6InB0LWJyIn0= | `a5ea2ba9a659b94e47840a038c6d0026a8a838474469acb7b1501e15a2b6a531` | 709 |
| cad_cia_aberta.csv | https://dados.cvm.gov.br/dados/CIA_ABERTA/CAD/DADOS/cad_cia_aberta.csv | `4364c52c1102d94db8e99ecc67a038350e5936c9aa298802fcebb76349a58121` | 1494679 |
| dfp_cia_aberta_2019.zip | https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP/DADOS/dfp_cia_aberta_2019.zip | `bc82a1162b44686e8953b521e3424a5938b6661ff4655f832db4ea9488296a0d` | 11185526 |
| dfp_cia_aberta_2020.zip | https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP/DADOS/dfp_cia_aberta_2020.zip | `8fb97d1887aed1e1e2d85e54cb133baf9f65b9f46f60fa8720f09c1f7a74362e` | 12637854 |
| dfp_cia_aberta_2021.zip | https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP/DADOS/dfp_cia_aberta_2021.zip | `64b8693b709538a0357c5179ae02231cb380d0000956ee57a873a21d4efda2e4` | 13283026 |
| dfp_cia_aberta_2022.zip | https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP/DADOS/dfp_cia_aberta_2022.zip | `151d605dc4c2bbe324ecbcfda29730017a6891314382ee3ebd9554df765c2f2a` | 13449824 |
| dfp_cia_aberta_2023.zip | https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP/DADOS/dfp_cia_aberta_2023.zip | `344e37b704a588b3863c5dd4ad6ee54b3dc9a2ee18cf88e0e4e35f5a641cdc2d` | 13563493 |
| dfp_cia_aberta_2024.zip | https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP/DADOS/dfp_cia_aberta_2024.zip | `b6a485b540728dcd4e7ed030bad8e749471044594ec17866fff4f0e94c54120a` | 13397599 |
| dfp_cia_aberta_2025.zip | https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP/DADOS/dfp_cia_aberta_2025.zip | `a0aaf1ed8be9f7a644d1ec7eb6c17f2d0860f40416c0e3151c2c690329eeb343` | 12863166 |
| dfp_cia_aberta_2026.zip | https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP/DADOS/dfp_cia_aberta_2026.zip | `eda1116c2434715051b7fcaa165ff8e53f7c791fc44233282f3d00203f44e159` | 243348 |
| fca_cia_aberta_2010.zip | https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FCA/DADOS/fca_cia_aberta_2010.zip | `9a9b933fa02c96f7cf632dbb1f7b004bb78696c81c989792b1eb60f70b001021` | 337720 |
| fca_cia_aberta_2011.zip | https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FCA/DADOS/fca_cia_aberta_2011.zip | `f5c7aaa2a4464b4232a05d77c288d33bca5474edd72199a9295d6fa2a34012e8` | 381161 |
| fca_cia_aberta_2012.zip | https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FCA/DADOS/fca_cia_aberta_2012.zip | `ba68b8e8c12b74fc56bf76a75ae582aa53e5adef1d61d940d5191c55b9be2e06` | 381058 |
| fca_cia_aberta_2013.zip | https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FCA/DADOS/fca_cia_aberta_2013.zip | `d62ef489d8c1208ef3d41cfd25a093b8fcb5f6b47640e59288aece45d4962cce` | 373007 |
| fca_cia_aberta_2014.zip | https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FCA/DADOS/fca_cia_aberta_2014.zip | `1f70b5e3a544535affb47b04a0c60a618bef872bc783e6909940d2e7dad89e3f` | 384748 |
| fca_cia_aberta_2015.zip | https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FCA/DADOS/fca_cia_aberta_2015.zip | `74326a566c405d703b56af257c9fade687f112f372ae68b8624719becc8215f2` | 380898 |
| fca_cia_aberta_2016.zip | https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FCA/DADOS/fca_cia_aberta_2016.zip | `229b2cfc8c0ac5e60083932381364bbdb6666f028422c13a1ac5f6dccc9e3e6c` | 376154 |
| fca_cia_aberta_2017.zip | https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FCA/DADOS/fca_cia_aberta_2017.zip | `48127cf15d8baf4d31b9544b266f9511962db158d591fc9803e335a8ab081d45` | 375132 |
| fca_cia_aberta_2018.zip | https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FCA/DADOS/fca_cia_aberta_2018.zip | `b20056b00927c5261ba587b668e52e4cdfe2d989e7d794491b2bb127fbd0a8cd` | 381642 |
| fca_cia_aberta_2019.zip | https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FCA/DADOS/fca_cia_aberta_2019.zip | `09f84e5f5485217c8b36f8379c36f718dbdcd6d6908b4e96b68861394e76fe97` | 371770 |
| fca_cia_aberta_2020.zip | https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FCA/DADOS/fca_cia_aberta_2020.zip | `ccf2ac081747da9ccefa50349a5494c263763d5dd8bb961565fd2ae64c264b20` | 405892 |
| fca_cia_aberta_2021.zip | https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FCA/DADOS/fca_cia_aberta_2021.zip | `65ba4b0225e9b2106901cdf348215933be8affba79925fc71361e2f6818c7999` | 465704 |
| fca_cia_aberta_2022.zip | https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FCA/DADOS/fca_cia_aberta_2022.zip | `c30912a49163e54503b5664c118cf088212e8715fcdbc7cad76bb4944b36c6a2` | 458295 |
| fca_cia_aberta_2023.zip | https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FCA/DADOS/fca_cia_aberta_2023.zip | `04364dea9a0bfca8816315792720f640e971f4b3c5ab67d5cb918803173bbc71` | 434052 |
| fca_cia_aberta_2024.zip | https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FCA/DADOS/fca_cia_aberta_2024.zip | `469bccc738b4e71f94d130ac10757123b5b1ef9f7170e1df2f8330e3926ab481` | 391984 |
| fca_cia_aberta_2025.zip | https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FCA/DADOS/fca_cia_aberta_2025.zip | `62877519e3543a568b64f8bd5ea8c956327ae9b21bb0cb0e7af901e94aa1f926` | 383613 |
| fca_cia_aberta_2026.zip | https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FCA/DADOS/fca_cia_aberta_2026.zip | `b24da0981f891532bb7d3068bf714f93d76c3a0cdf0ecb16badbab52b7081917` | 363925 |
| itr_cia_aberta_2024.zip | https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/ITR/DADOS/itr_cia_aberta_2024.zip | `bdfa4c83bcc2bfac8f78dbca85494eea209c018ce429ac2ec15eb8d900762d7a` | 32725967 |
| itr_cia_aberta_2025.zip | https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/ITR/DADOS/itr_cia_aberta_2025.zip | `d29c6e378b60f61b1fb9978e5d407a8f13c245ab268194430227ebf392bba61f` | 31684923 |
| itr_cia_aberta_2026.zip | https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/ITR/DADOS/itr_cia_aberta_2026.zip | `115227a5d0872e23a532c0f17c17683f7218ac225013eea2901844a92b3fe4a9` | 22183718 |
| listing_dfp.html | https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP/DADOS/ | `6951d9486073da7cd6ea2583756a84a501e472734677aa776e561f830f09f921` | 3972 |
| listing_fca.html | https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FCA/DADOS/ | `c1ede9df2449bf9e8426c70c2dc0ae7dfdc627e165ac3832003892a2b5f0174a` | 3972 |
| listing_itr.html | https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/ITR/DADOS/ | `6c21a1fa4c9b80d423b140ceaa5688a3132c2dca8c38e4128d9c0cbb89633472` | 3856 |
| yahoo_ALPK3_2026-11-07.json | https://query1.finance.yahoo.com/v8/finance/chart/ALPK3.SA?period1=946684800&period2=1794009600&interval=1d&events=div | `dfc05ae9b379b0445464535b3f7d87dd5d1ad03a33538b11c7ad01b456ea72e5` | 170750 |
| yahoo_AZUL3_2026-11-07.json | https://query1.finance.yahoo.com/v8/finance/chart/AZUL3.SA?period1=946684800&period2=1794009600&interval=1d&events=div | `bce2d32dfc2b09e795975d4987578b7ed715b60dbcee073b58d7e2d4b83625df` | 20489 |
| yahoo_BIOM3_2026-11-07.json | https://query1.finance.yahoo.com/v8/finance/chart/BIOM3.SA?period1=946684800&period2=1794009600&interval=1d&events=div | `607cd71227d3c2bc7c0cf25fa2f5b1e7ef54f3cbad9b83e83bb43aef761b4e78` | 715370 |
| yahoo_DMVF3_2026-11-07.json | https://query1.finance.yahoo.com/v8/finance/chart/DMVF3.SA?period1=946684800&period2=1794009600&interval=1d&events=div | `c29d33f899e1de38d8973affa0cbda52a13c47e2e44aa07462587b96f7112b5d` | 161390 |
| yahoo_HBRE3_2026-11-07.json | https://query1.finance.yahoo.com/v8/finance/chart/HBRE3.SA?period1=946684800&period2=1794009600&interval=1d&events=div | `33304bac960b194e72ad4e4f4fdb1f26b54b28936e85a8178a5c301cac84bbcd` | 152392 |
| yahoo_HBSA3_2026-11-07.json | https://query1.finance.yahoo.com/v8/finance/chart/HBSA3.SA?period1=946684800&period2=1794009600&interval=1d&events=div | `abdf59704c0e2d04406f655049ebe49b37b5e849028dfb1dd63d1c44f2f8c43e` | 164460 |
| yahoo_ONCO3_2026-11-07.json | https://query1.finance.yahoo.com/v8/finance/chart/ONCO3.SA?period1=946684800&period2=1794009600&interval=1d&events=div | `4938bdd28ec283b5e751783d16182efae30d3faf8ddc1ff8342fe97233652d5a` | 139030 |
| yahoo_OPCT3_2026-11-07.json | https://query1.finance.yahoo.com/v8/finance/chart/OPCT3.SA?period1=946684800&period2=1794009600&interval=1d&events=div | `4219bd183f376d87c5bb76c5f9c39feb350377b697aa810675a64e0125dd8e8e` | 152463 |
| yahoo_ORVR3_2026-11-07.json | https://query1.finance.yahoo.com/v8/finance/chart/ORVR3.SA?period1=946684800&period2=1794009600&interval=1d&events=div | `0ba63604225701f86739c90977bbdb55e4481044ab8e1fd4247ead50a8b6ecce` | 153966 |
| yahoo_RNEW3_2026-11-07.json | https://query1.finance.yahoo.com/v8/finance/chart/RNEW3.SA?period1=946684800&period2=1794009600&interval=1d&events=div | `011b6028dbdf6a13a72de5d298f95ce8a2040bfe844be9fd08b71b7e493c67b4` | 300778 |
| yahoo_RNEW4_2026-11-07.json | https://query1.finance.yahoo.com/v8/finance/chart/RNEW4.SA?period1=946684800&period2=1794009600&interval=1d&events=div | `3bd3e42ce23b0e1bd209905d74522658211b75f2776415856492dfaf733aa670` | 246261 |
| yahoo_SEQL3_2026-11-07.json | https://query1.finance.yahoo.com/v8/finance/chart/SEQL3.SA?period1=946684800&period2=1794009600&interval=1d&events=div | `875db8ddd11139c5d0718040b5b7a3eff9bed35f871903f460d22f6faffa1bef` | 130861 |

### Status Invest

| data | coletado em | caminho | sha256 |
| --- | --- | --- | --- |
| 2026-08-07 | 2026-08-07 | data/raw/status_invest_fundamentals.csv | `ceeed0b172069899e0b877f050c938a56230f7ff1b459793397f59d3cbbd3b10` |
| 2026-11-05 | 2026-11-06 | /home/caio/Documents/projetos/TCC/snapshots/data/selectors-2026-11-06/stocks.csv | `87c15cb54209966bc44224e49939db9be9b9a5993061155feaf465387ac5044e` |

### Data de referência (as-of)

**2026-08-07**: 269 elegíveis com cotação. Sessão de melhor correspondência: `2026-08-07`.

| sessão | tickers com o mesmo preço |
| --- | --- |
| 2026-08-06 | 11 |
| 2026-08-07 | 245 |

**2026-11-05**: 274 elegíveis com cotação. Sessão de melhor correspondência: `2026-11-05`.

| sessão | tickers com o mesmo preço |
| --- | --- |
| 2026-11-04 | 11 |
| 2026-11-05 | 255 |
| 2026-11-06 | 8 |

